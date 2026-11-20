"""
sentiment_analysis.py
=====================
Tahap 8 — Analisis sentimen berbasis 3 LLM Anotator + Fleiss' Kappa + Majority Vote:
  1) Gemini 3.5 Flash (Google)
  2) Llama 3.3 70B Instruct (Meta via Cloudflare Workers AI)
  3) Qwen 3.8 27B (Alibaba via Groq)

Cara pakai:
    python scripts/sentiment_analysis.py --pilot           # uji pilot 150 komentar
    python scripts/sentiment_analysis.py                   # full run 6.601 komentar
    python scripts/sentiment_analysis.py --annotator 1     # jalankan 1 anotator saja (1|2|3)
    python scripts/sentiment_analysis.py --aggregate-only  # hitung Fleiss' Kappa & Majority Vote dari cache

Input :
    data/processed/preprocessing/comments_clean.csv (filter: keep_sentiment == True)

Output:
    data/processed/sentiment_analysis/llm_cache/annotator_{1,2,3}_*.csv  (checkpoint per batch)
    data/processed/sentiment_analysis/sentiment_predictions.csv
        kolom: comment_id, video_id, author_id, annotator_1_label, annotator_2_label,
               annotator_3_label, sentiment, sentiment_source
    outputs/sentiment_analysis/sentiment_agreement.csv
        kolom: n_comments, fleiss_kappa, percent_agreement, interpretation, meets_target_85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.sentiment_analysis import llm_annotator as llm
from src.sentiment_analysis import sentiment as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tahap 8: Analisis Sentimen Multi-LLM + Fleiss' Kappa")
    parser.add_argument("--pilot", action="store_true", help="Jalankan uji pilot untuk cek Fleiss' Kappa")
    parser.add_argument("--pilot-size", type=int, default=config.SENTIMENT_SAMPLE_SIZE, help="Jumlah sampel pilot")
    parser.add_argument("--annotator", type=int, choices=[1, 2, 3], help="Jalankan 1 anotator tertentu saja")
    parser.add_argument("--aggregate-only", action="store_true", help="Hanya agregasi dari cache tanpa panggil API")
    parser.add_argument("--batch-size", type=int, default=None, help="Override ukuran batch per request")
    parser.add_argument("--delay", type=float, default=None, help="Override jeda detik antar batch")
    return parser.parse_args()


def load_comments() -> pd.DataFrame:
    path = config.PREPROCESSING_PROCESSED_DIR / "comments_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}. Jalankan preprocess.py terlebih dahulu.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["keep_sentiment"] == True].copy()  # noqa: E712
    df["comment_id"] = df["comment_id"].astype(str)
    df["text_light"] = df["text_light"].fillna("").astype(str)
    return df[["comment_id", "video_id", "author_id", "text_light"]].reset_index(drop=True)


def cache_file_for(annotator: dict) -> Path:
    return config.SENTIMENT_LLM_CACHE_DIR / f"annotator_{annotator['id']}_{annotator['name']}.csv"


def select_pilot_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=config.SENTIMENT_SEED).reset_index(drop=True)


def aggregate_and_save(df: pd.DataFrame, caches: dict[int, dict[str, str]], is_pilot: bool) -> None:
    annotator_cols = ["annotator_1_label", "annotator_2_label", "annotator_3_label"]

    out = df[["comment_id", "video_id", "author_id"]].copy()
    for idx, col in enumerate(annotator_cols, start=1):
        out[col] = out["comment_id"].map(caches.get(idx, {}))

    missing_counts = {
        col: int((out[col].isna() | (out[col] == llm.PENDING_LABEL)).sum())
        for col in annotator_cols
    }
    if any(missing_counts.values()):
        print("\n[INFO RESUME] Sebagian komentar masih pending / belum teranotasi:")
        for col, cnt in missing_counts.items():
            if cnt:
                print(f"  - {col}: {cnt} komentar pending")
        print("  -> Jalankan ulang perintah yang sama nanti untuk resume khusus yang pending.")

    # Resolusi Majority Vote
    sentiment_series, _ = st.resolve_majority_vote(
        out, annotator_cols, tie_breaker=config.SENTIMENT_TIE_BREAKER
    )
    out["sentiment"] = sentiment_series
    out["sentiment_source"] = "llm_majority_vote"

    # Metrik reliabilitas Fleiss' Kappa
    summary = st.compute_agreement_summary(
        out,
        annotator_cols,
        target_kappa=config.SENTIMENT_FLEISS_TARGET,
    )

    prefix = "pilot_" if is_pilot else ""
    if is_pilot:
        pilot_pred_path = config.SENTIMENT_OUTPUT_DIR / "pilot_predictions.csv"
        out.to_csv(pilot_pred_path, index=False, encoding="utf-8-sig")

        valid_all = out[out[annotator_cols].isin(st.LABELS).all(axis=1)]
        disagreements = valid_all[valid_all[annotator_cols].nunique(axis=1) > 1].merge(
            df[["comment_id", "text_light"]], on="comment_id", how="left"
        )
        dis_path = config.SENTIMENT_OUTPUT_DIR / "pilot_disagreements.csv"
        disagreements.to_csv(dis_path, index=False, encoding="utf-8-sig")
        print(f"\n[PILOT] Prediksi -> {pilot_pred_path} | Tidak kompak ({len(disagreements)} baris) -> {dis_path}")
    else:
        pred_path = config.SENTIMENT_PROCESSED_DIR / "sentiment_predictions.csv"
        out.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"\nPrediksi sentimen disimpan -> {pred_path} ({len(out)} baris)")

    agr_path = config.SENTIMENT_OUTPUT_DIR / f"{prefix}sentiment_agreement.csv"
    pd.DataFrame([summary]).to_csv(agr_path, index=False, encoding="utf-8-sig")

    print("\n=== HASIL FLEISS' KAPPA ===")
    print(f"Jumlah komentar dievaluasi : {summary['n_comments']}")
    if summary["n_comments"] > 0:
        print(f"Fleiss' Kappa              : {summary['fleiss_kappa']:.4f} ({summary['interpretation']} - Landis & Koch, 1977)")
        print(f"Percent Agreement          : {summary['percent_agreement'] * 100:.2f}%")
        status = "MEMENUHI SYARAT (>= 0.85)" if summary["meets_target_85"] else "BELUM MEMENUHI SYARAT (< 0.85)"
        print(f"Status Target (>= 0.85)    : {status}")
    else:
        print("Fleiss' Kappa belum dapat dihitung (menunggu ketiga anotator memiliki hasil selesai).")


def main() -> None:
    args = parse_args()
    df = load_comments()
    if args.pilot:
        df = select_pilot_sample(df, args.pilot_size)
        print(f"Mode PILOT aktif: {len(df)} komentar sampel acak (seed={config.SENTIMENT_SEED}).")
    else:
        print(f"Mode FULL RUN: {len(df)} komentar (keep_sentiment == True).")

    annotators = config.LLM_ANNOTATORS
    if args.annotator is not None:
        annotators = [a for a in annotators if a["id"] == args.annotator]

    caches: dict[int, dict[str, str]] = {}
    for a in config.LLM_ANNOTATORS:
        caches[a["id"]] = llm.load_cache(cache_file_for(a), include_pending=True)

    if not args.aggregate_only:
        for a in annotators:
            try:
                caches[a["id"]] = llm.annotate_with_llm(
                    df=df,
                    annotator=a,
                    cache_path=cache_file_for(a),
                    batch_size=args.batch_size,
                    request_delay=args.delay,
                )
            except Exception as exc:
                print(
                    f"\n[PERINGATAN] Annotator {a['id']} ({a['name']}) berhenti sementara: {exc}\n"
                    "  -> Melanjutkan proses model anotator lainnya."
                )
                caches[a["id"]] = llm.load_cache(cache_file_for(a), include_pending=True)

    if args.annotator is not None and not args.aggregate_only:
        print(
            f"\nAnnotator {args.annotator} selesai di-cache. "
            "Jalankan annotator lainnya atau jalankan --aggregate-only setelah ketiganya selesai."
        )
        return

    aggregate_and_save(df, caches, is_pilot=args.pilot)


if __name__ == "__main__":
    main()
