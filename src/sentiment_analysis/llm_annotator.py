"""Multi-LLM batch sentiment annotator with checkpointing and structured JSON output."""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.sentiment_analysis.sentiment import LABELS, normalize_label

PENDING_LABEL = "pending"
GEMINI_503_BACKOFF_SECONDS = (15, 30, 60, 120, 180)
GEMINI_MAX_CONCURRENCY = 1  # Request dikirim sekuensial (1 batch pada satu waktu)


class ServiceUnavailableError(RuntimeError):
    """Raised when an LLM endpoint (e.g. Gemini HTTP 503) stays unavailable after all retries."""

SYSTEM_PROMPT = """Anda adalah anotator ahli analisis sentimen berbahasa Indonesia untuk komentar YouTube terkait wacana Pilpres 2029.

Tugas Anda: klasifikasikan SETIAP komentar ke tepat SATU label: "positif", "netral", atau "negatif".

=== PEDOMAN LABEL (WAJIB DIIKUTI SECARA KETAT & KONSISTEN) ===

1. "positif":
   - Dukungan, pujian, harapan baik, doa/aamiin, ajakan mendukung, rekomendasi tokoh/pasangan yang pantas dipilih, atau pembelaan terhadap tokoh/isu.
   - Menyebut nama tokoh/nomor disertai dukungan, pilihan, slogan, atau tahun (mis. "KDM RI 1", "kdm susi pastinya", "no 1 lah", "yang pas adalah X", "yang sangat pantas X dan Y", "mending X sih").
   - Doa/harapan positif murni (mis. "aamiin ya robal alamin", "ayo pak kami menunggu sosok bapak").
   - Doa/dukungan dengan catatan syarat ringan tanpa hujatan (mis. "insya allah X jadi presiden asal tidak dicurangi") -> "positif".

2. "negatif":
   - Kritik, penolakan, keluhan, kekecewaan, kemarahan, ejekan, makian, hujatan, pesimisme, atau ketidakpercayaan terhadap tokoh, pemerintah, partai, survei, atau sistem politik.
   - Sarkasme atau sindiran politik Indonesia (mis. "pejabat konoha", "survey termul", "kandidat terkuat di yaman", "angka 58 sudah ditentukan", "ikut paket C dulu", "hebat menata kata", "jangan narkoboy", "amit amit") -> WAJIB "negatif".
   - Penolakan/sindiran dalam bahasa daerah seperti Jawa/Sunda (mis. "sing ameh milih sopo", "wong 3 ora ana sing tak pilih") -> WAJIB "negatif".
   - Pertanyaan retoris yang menyindir/meremehkan (mis. "emang masih laku?", "emang masih ada yang percaya?", "sakit ginjal ya") -> WAJIB "negatif".
   - ATURAN KOMENTAR CAMPURAN (WAJIB): Jika komentar memuji satu tokoh TAPI juga mengandung ejekan, hinaan, atau serangan tajam terhadap tokoh/pihak lain (mis. "X layak, tidak seperti Y yang ijazahnya dipertanyakan") -> WAJIB "negatif".

3. "netral":
   - Pertanyaan informatif murni tanpa sindiran (mis. "ini video tahun berapa?", "hasil survei lembaga mana?").
   - Pernyataan fakta, analisis objektif, atau prediksi/sebutan pasangan tanpa nada pujian, dukungan, maupun serangan emosi (mis. "dedi mulyadi berpasangan mahfud md", "prabowo dulu belum jadi presiden sekarang sudah pasti beda strateginya").
   - Komentar sikap terbuka/tidak memihak ("siapapun yg jadi bolehlah").
   - Komentar super pendek/ambigu tanpa konteks sikap politik yang jelas atau spam tidak bermakna (mis. "yah benar", "ngga lah", "yaman ya", "crot crot croot", atau hanya menyebut 1 nama tokoh tanpa konteks).

=== CONTOH KALIBRASI (FEW-SHOT) ===
- "saya dari jambi dukung kdm menjadi ri1 2029" -> "positif"
- "kdm ri 1" -> "positif"
- "aamiin ya robal allamin kobul ya allah" -> "positif"
- "yang sangat pantas ada dua yusril dan ahy" -> "positif"
- "insya allah kdm jadi ri 1 tahun 2029 asal tidak diganggu elit yang curang" -> "positif"
- "dedi muluadi berpassmgan mahfud md" -> "netral"
- "prabowo blm jd presiden sekarang dah jd presiden ya pasti bedalah" -> "netral"
- "saya pilih no 4 atau no 2 bolehlah siapapun yg jadi" -> "netral"
- "yah benar" -> "netral"
- "survey termul" -> "negatif"
- "baswedan kandidat terkuat di yaman" -> "negatif"
- "wong 3 ora ana sing tk pilih" -> "negatif"
- "emang msh laku" -> "negatif"
- "ahy lebih layak daripada si gibran bocah solo yg ijazahnya dipertanyakan" -> "negatif"

=== FORMAT OUTPUT ===
Kembalikan HANYA objek JSON valid tanpa teks tambahan di luar JSON, dengan format:
{"results": [{"comment_id": "<id>", "label": "positif|netral|negatif"}, ...]}
Pastikan SEMUA comment_id dari input muncul tepat satu kali di output."""


def _build_user_prompt(batch: list[dict[str, str]]) -> str:
    payload = [{"comment_id": item["comment_id"], "text": item["text"]} for item in batch]
    return (
        "Klasifikasikan sentimen komentar berikut, kembalikan HANYA JSON `results`:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _endpoint_url(annotator: dict[str, Any]) -> str:
    base_url = annotator["base_url"].rstrip("/")
    if annotator["provider"].lower() == "gemini":
        return f"{base_url}/models/{annotator['model']}:generateContent"
    return f"{base_url}/chat/completions"


def _sanitize_error_text(text: str, api_key: str) -> str:
    cleaned = text[:300]
    if api_key:
        cleaned = cleaned.replace(api_key, "***")
    return cleaned


def _parse_duration_seconds(raw: str | None) -> float | None:
    """Parse seconds from Retry-After or duration strings like '14.3s', '1m12.5s', '500ms'."""
    if not raw:
        return None
    s = raw.strip().lower()
    try:
        val = float(s)
        return val if val >= 0 else None
    except ValueError:
        pass
    m_min_sec = re.search(r"(?:(\d+(?:\.\d+)?)m)?\s*(\d+(?:\.\d+)?)s\b", s)
    if m_min_sec:
        mins = float(m_min_sec.group(1) or 0.0)
        secs = float(m_min_sec.group(2))
        return mins * 60.0 + secs
    m_ms = re.search(r"(\d+(?:\.\d+)?)ms\b", s)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    return None


def _extract_retry_after(resp: requests.Response | None) -> float | None:
    """Extract server-requested wait time (seconds) from HTTP 429 headers or error body."""
    if resp is None:
        return None
    for header in ("Retry-After", "retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        wait = _parse_duration_seconds(resp.headers.get(header))
        if wait is not None and wait > 0:
            return wait
    body = resp.text or ""
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?m)?\s*\d+(?:\.\d+)?(?:ms|s))", body, flags=re.IGNORECASE)
    if m:
        return _parse_duration_seconds(m.group(1))
    return None


def _get_runtime_profile(
    annotator: dict[str, Any],
    requested_batch_size: int | None,
    requested_delay: float | None,
) -> dict[str, Any]:
    """Read batch size, delay, text length, and TPM budget from annotator config (with optional CLI override)."""
    batch_size = requested_batch_size if requested_batch_size is not None else int(annotator.get("batch_size", 40))
    min_delay = requested_delay if requested_delay is not None else float(annotator.get("request_delay", 3.0))
    return {
        "batch_size": max(1, batch_size),
        "min_delay": max(0.0, min_delay),
        "tpm_budget": int(annotator.get("tpm_budget", 0)),
        "max_text_chars": int(annotator.get("max_text_chars", 450)),
        "concurrency": GEMINI_MAX_CONCURRENCY,
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from LLM response, stripping markdown fences if present."""
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _parse_batch_response(raw_text: str, expected_ids: set[str]) -> dict[str, str]:
    data = _extract_json_object(raw_text)
    items = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(items, list):
        raise ValueError("JSON output does not contain a list under 'results'.")
    parsed: dict[str, str] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("comment_id", "")).strip()
        lbl = normalize_label(entry.get("label"))
        if cid in expected_ids and lbl in LABELS:
            parsed[cid] = lbl
    return parsed


def _call_gemini_api(
    annotator: dict[str, Any],
    user_prompt: str,
    batch_len: int = 10,
    timeout: int = 120,
) -> tuple[str, int]:
    url = _endpoint_url(annotator)
    api_key = annotator["api_key"]
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": min(4096, max(512, batch_len * 45 + 200)),
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Some Gemini models reject thinkingBudget=0; retry once without thinkingConfig
    if resp.status_code == 400 and "thinking" in resp.text.lower():
        payload["generationConfig"].pop("thinkingConfig", None)
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        text = "".join(p.get("text", "") for p in parts)
    total_tokens = int(body.get("usageMetadata", {}).get("totalTokenCount", 0) or 0)
    return text, total_tokens


def _call_openai_compatible_api(
    annotator: dict[str, Any],
    user_prompt: str,
    batch_len: int = 20,
    timeout: int = 120,
) -> tuple[str, int]:
    url = _endpoint_url(annotator)
    headers = {
        "Authorization": f"Bearer {annotator['api_key']}",
        "Content-Type": "application/json",
    }
    max_tokens = min(3500, max(512, batch_len * 50 + 250))
    payload: dict[str, Any] = {
        "model": annotator["model"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Fallback bila provider/model menolak response_format=json_object
    if resp.status_code == 400 and "response_format" in resp.text.lower():
        payload.pop("response_format", None)
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"API returned no choices: {body}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty message content from {annotator['name']}: {body}")
    total_tokens = int(body.get("usage", {}).get("total_tokens", 0) or 0)
    return content, total_tokens


def call_llm_batch(
    annotator: dict[str, Any],
    batch: list[dict[str, str]],
    max_retries: int = 5,
    base_backoff: float = 3.0,
    return_usage: bool = False,
) -> dict[str, str] | tuple[dict[str, str], int]:
    """Send a batch of comments sequentially (concurrency=1) with retry + backoff + Retry-After.

    - HTTP 503 (Gemini): backoff 15, 30, 60, 120, 180 detik + random jitter, maks 5 retry.
    - HTTP 429 (Groq/dll): patuhi `Retry-After` / waktu tunggu server (+ margin), fallback exponential backoff.
    - Logging endpoint & model ID tanpa pernah mencetak API key.
    """
    provider = annotator["provider"].lower()
    endpoint = _endpoint_url(annotator)
    model_id = annotator["model"]
    expected_ids = {item["comment_id"] for item in batch}
    user_prompt = _build_user_prompt(batch)

    last_error: Exception | None = None
    last_status: int = 0
    for retry_idx in range(max_retries + 1):
        try:
            if provider == "gemini":
                raw_text, total_tokens = _call_gemini_api(annotator, user_prompt, batch_len=len(batch))
            elif provider == "openai_compatible":
                raw_text, total_tokens = _call_openai_compatible_api(annotator, user_prompt, batch_len=len(batch))
            else:
                raise ValueError(f"Provider tidak dikenal: {provider} (pakai 'gemini' atau 'openai_compatible')")
            parsed = _parse_batch_response(raw_text, expected_ids)
            if not parsed:
                raise ValueError("Tidak ada label valid yang berhasil diparse dari respons LLM.")
            return (parsed, total_tokens) if return_usage else parsed
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else 0
            last_status = status
            detail = _sanitize_error_text(
                exc.response.text if exc.response is not None else str(exc),
                annotator.get("api_key", ""),
            )
            if status in (401, 403):
                raise RuntimeError(
                    f"[{annotator['name']}] Autentikasi gagal (HTTP {status}) | "
                    f"endpoint={endpoint} | model={model_id}\nDetail: {detail}"
                ) from exc
            if retry_idx >= max_retries:
                break
            if status == 503:
                base_wait = GEMINI_503_BACKOFF_SECONDS[min(retry_idx, len(GEMINI_503_BACKOFF_SECONDS) - 1)]
                jitter = random.uniform(1.0, 5.0)
                wait = base_wait + jitter
                print(
                    f"  [retry {retry_idx + 1}/{max_retries}] {annotator['name']} HTTP 503 | "
                    f"model={model_id} | endpoint={endpoint}\n"
                    f"    Detail: {detail}\n"
                    f"    Tunggu {wait:.1f}s (base {base_wait}s + jitter {jitter:.1f}s)..."
                )
            elif status == 429:
                server_wait = _extract_retry_after(exc.response)
                if server_wait is not None:
                    jitter = random.uniform(1.0, 3.0)
                    wait = server_wait + jitter
                    reason = f"Retry-After server {server_wait:.1f}s + jitter {jitter:.1f}s"
                else:
                    base_wait = GEMINI_503_BACKOFF_SECONDS[min(retry_idx, len(GEMINI_503_BACKOFF_SECONDS) - 1)]
                    jitter = random.uniform(1.0, 5.0)
                    wait = base_wait + jitter
                    reason = f"exponential backoff {base_wait}s + jitter {jitter:.1f}s"
                print(
                    f"  [retry {retry_idx + 1}/{max_retries}] {annotator['name']} HTTP 429 | "
                    f"model={model_id} | endpoint={endpoint}\n"
                    f"    Detail: {detail}\n"
                    f"    Tunggu {wait:.1f}s ({reason})..."
                )
            else:
                base_wait = base_backoff * (2 ** retry_idx)
                jitter = random.uniform(0.5, 3.0)
                wait = base_wait + jitter
                print(
                    f"  [retry {retry_idx + 1}/{max_retries}] {annotator['name']} HTTP {status} | "
                    f"model={model_id} | endpoint={endpoint}\n"
                    f"    Detail: {detail}\n"
                    f"    Tunggu {wait:.1f}s..."
                )
            time.sleep(wait)
        except Exception as exc:
            last_error = exc
            if retry_idx >= max_retries:
                break
            wait = base_backoff * (retry_idx + 1) + random.uniform(0.5, 2.5)
            print(
                f"  [retry {retry_idx + 1}/{max_retries}] {annotator['name']} error | "
                f"model={model_id} | endpoint={endpoint}: {exc}. Tunggu {wait:.1f}s..."
            )
            time.sleep(wait)

    if last_status in (503, 429):
        raise ServiceUnavailableError(
            f"[{annotator['name']}] HTTP {last_status} masih gagal setelah {max_retries} kali retry "
            f"(model={model_id}, endpoint={endpoint})."
        ) from last_error
    raise RuntimeError(
        f"[{annotator['name']}] Gagal setelah {max_retries} kali retry "
        f"(model={model_id}, endpoint={endpoint}): {last_error}"
    ) from last_error


def load_cache(cache_path: Path, include_pending: bool = False) -> dict[str, str]:
    """Load cached comment_id -> label mapping from checkpoint CSV.

    Secara default hanya mengembalikan komentar yang sudah berhasil dilabeli
    (positif/netral/negatif), sehingga komentar `pending` akan otomatis dicoba ulang saat resume.
    """
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path, dtype=str, encoding="utf-8-sig")
    if "comment_id" not in df or "label" not in df:
        return {}
    cache: dict[str, str] = {}
    for cid, lbl in zip(df["comment_id"], df["label"]):
        if not cid or pd.isna(cid):
            continue
        raw = str(lbl).strip().lower() if lbl is not None and not pd.isna(lbl) else ""
        if raw == PENDING_LABEL:
            if include_pending:
                cache[str(cid)] = PENDING_LABEL
            continue
        norm = normalize_label(raw)
        if norm in LABELS:
            cache[str(cid)] = norm
    return cache


def save_cache(cache_path: Path, cache: dict[str, str], model_name: str) -> None:
    """Persist checkpoint CSV atomically (menyimpan label berhasil maupun status pending)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        [
            {
                "comment_id": cid,
                "label": lbl,
                "status": PENDING_LABEL if lbl == PENDING_LABEL else "ok",
                "model": model_name,
            }
            for cid, lbl in cache.items()
        ]
    )
    tmp_path = cache_path.with_suffix(".tmp.csv")
    out.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(cache_path)


def annotate_with_llm(
    df: pd.DataFrame,
    annotator: dict[str, Any],
    cache_path: Path,
    batch_size: int | None = None,
    request_delay: float | None = None,
) -> dict[str, str]:
    """Annotate all rows in `df` sequentially (concurrency=1) with per-batch CSV checkpointing.

    - Komentar yang sudah berhasil di cache tidak dikirim ulang.
    - Komentar yang sebelumnya `pending` atau belum ada otomatis diproses ulang saat resume.
    - Jika HTTP 503 tetap gagal setelah 5 retry, komentar yang tersisa ditandai `pending`,
      disimpan ke CSV, dan proses dilanjutkan ke anotator berikutnya tanpa crash.
    """
    if not annotator.get("api_key") or "masukkan_api_key" in annotator["api_key"].lower():
        raise EnvironmentError(
            f"API key untuk Annotator {annotator['id']} ({annotator['name']}) belum diisi di .env."
        )

    profile = _get_runtime_profile(annotator, batch_size, request_delay)
    eff_batch_size = profile["batch_size"]
    eff_min_delay = profile["min_delay"]
    tpm_budget = profile["tpm_budget"]
    max_text_chars = profile["max_text_chars"]
    endpoint = _endpoint_url(annotator)

    full_cache = load_cache(cache_path, include_pending=True)
    completed_ids = {cid for cid, lbl in full_cache.items() if lbl in LABELS}
    target_ids = [str(cid) for cid in df["comment_id"]]
    todo_df = df[~df["comment_id"].astype(str).isin(completed_ids)].copy()

    done_before = sum(1 for cid in target_ids if cid in completed_ids)
    pending_before = sum(1 for cid in target_ids if full_cache.get(cid) == PENDING_LABEL)
    print(
        f"\n[Annotator {annotator['id']}: {annotator['name']} | model={annotator['model']}]\n"
        f"  endpoint={endpoint} | concurrency={profile['concurrency']} | "
        f"batch_size={eff_batch_size} | min_delay={eff_min_delay:.1f}s"
        + (f" | tpm_budget={tpm_budget}" if tpm_budget else "")
        + f"\n  Selesai: {done_before}/{len(df)} | Pending (resume): {pending_before} | Akan diproses: {len(todo_df)}"
    )
    if todo_df.empty:
        return full_cache

    records = [
        {"comment_id": str(row.comment_id), "text": str(row.text_light)[:max_text_chars]}
        for row in todo_df[["comment_id", "text_light"]].itertuples(index=False)
    ]

    total_batches = (len(records) + eff_batch_size - 1) // eff_batch_size
    for idx in range(0, len(records), eff_batch_size):
        batch_num = idx // eff_batch_size + 1
        batch = records[idx : idx + eff_batch_size]
        t0 = time.monotonic()
        try:
            parsed, tokens_used = call_llm_batch(annotator, batch, max_retries=5, return_usage=True)
            full_cache.update(parsed)

            # Retry missing items in smaller sub-batch once if LLM skipped any IDs
            missing = [item for item in batch if item["comment_id"] not in parsed]
            if missing:
                print(f"  Batch {batch_num}: {len(missing)} komentar terlewat, mengulang khusus yang terlewat...")
                time.sleep(eff_min_delay)
                try:
                    retry_parsed, retry_tokens = call_llm_batch(
                        annotator, missing, max_retries=5, return_usage=True
                    )
                    tokens_used += retry_tokens
                    full_cache.update(retry_parsed)
                except ServiceUnavailableError as exc:
                    for item in missing:
                        if full_cache.get(item["comment_id"]) not in LABELS:
                            full_cache[item["comment_id"]] = PENDING_LABEL
                    save_cache(cache_path, full_cache, annotator["model"])
                    print(f"  [PENDING] {exc} Komentar yang terlewat ditandai pending; lanjut ke anotator lain.")
                    return full_cache

            # Any IDs still missing after sub-batch retry are marked pending so next resume retries them
            for item in batch:
                if full_cache.get(item["comment_id"]) not in LABELS:
                    full_cache[item["comment_id"]] = PENDING_LABEL

            save_cache(cache_path, full_cache, annotator["model"])
            done_now = sum(1 for cid in target_ids if full_cache.get(cid) in LABELS)
            pending_now = sum(1 for cid in target_ids if full_cache.get(cid) == PENDING_LABEL)
            tok_info = f" | tokens={tokens_used}" if tokens_used else ""
            print(
                f"  Batch {batch_num}/{total_batches} tersimpan{tok_info} -> "
                f"{done_now}/{len(df)} selesai (pending: {pending_now})."
            )
            if batch_num < total_batches:
                elapsed = time.monotonic() - t0
                wait_sec = eff_min_delay
                if tpm_budget > 0 and tokens_used > 0:
                    needed_for_tpm = (tokens_used / tpm_budget) * 60.0 - elapsed
                    wait_sec = max(wait_sec, needed_for_tpm)
                if wait_sec > 0:
                    time.sleep(wait_sec)
        except ServiceUnavailableError as exc:
            # Mark failed batch and remaining unannotated items as pending, save immediately, continue pipeline
            for item in records[idx:]:
                if full_cache.get(item["comment_id"]) not in LABELS:
                    full_cache[item["comment_id"]] = PENDING_LABEL
            save_cache(cache_path, full_cache, annotator["model"])
            done_now = sum(1 for cid in target_ids if full_cache.get(cid) in LABELS)
            pending_now = sum(1 for cid in target_ids if full_cache.get(cid) == PENDING_LABEL)
            print(
                f"  [PENDING] {exc}\n"
                f"  -> {done_now} selesai & {pending_now} ditandai 'pending' di {cache_path.name}.\n"
                f"  -> Melanjutkan proses model anotator lainnya (jalankan ulang nanti untuk resume)."
            )
            return full_cache

    return full_cache
