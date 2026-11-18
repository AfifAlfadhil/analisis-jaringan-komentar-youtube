"""
sentiment.py
============
Tahap 8 — Analisis sentimen 3 kelas (positif, netral, negatif).

1. Baseline leksikon: jumlah skor kata positif/negatif dengan penanganan negasi
   ("tidak bagus" -> negatif). Kamus bawaan kecil dan berfokus pada komentar politik.
   Bila file InSet (Koto & Rahmaningtyas, 2017; github.com/fajri91/InSet) disimpan di
   data/lexicon/positive.tsv dan negative.tsv, kamus itu dipakai sebagai gantinya.
2. Model Transformer (IndoBERT hasil fine-tune SmSA) via Hugging Face `transformers`.
3. Evaluasi terhadap label manual: accuracy, precision/recall/F1 (macro & per kelas),
   confusion matrix, dan Cohen's kappa antar-pelabel.

Teks yang dipakai: `text_light` (huruf, stopword, dan emoji-sebagai-kata dipertahankan),
karena model Transformer membutuhkan konteks kalimat utuh.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

from src.lexicons import SLANG_MAP

LABELS = ["positif", "netral", "negatif"]

# --------------------------------------------------------------------------- #
# Normalisasi label
# --------------------------------------------------------------------------- #
_LABEL_ALIASES = {
    "positif": "positif", "positive": "positif", "pos": "positif", "p": "positif", "+": "positif", "1": "positif",
    "netral": "netral", "neutral": "netral", "net": "netral", "n": "netral", "0": "netral",
    "negatif": "negatif", "negative": "negatif", "neg": "negatif", "-": "negatif", "-1": "negatif",
}


def normalize_label(x, extra_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Ubah berbagai penulisan label (positive, POS, LABEL_0 + peta, 1/0/-1) menjadi positif/netral/negatif."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    if extra_map and s in extra_map:
        return extra_map[s]
    return _LABEL_ALIASES.get(s.lower())


# --------------------------------------------------------------------------- #
# 1. Baseline leksikon
# --------------------------------------------------------------------------- #
POSITIVE_WORDS = """
bagus baik mantap mantab mantul hebat setuju cocok layak pantas cerdas pintar pinter jujur amanah tegas berani
sukses maju semangat dukung mendukung didukung bangga salut keren terbaik adil bersih peduli merakyat sejahtera
semoga aamiin amin amiin alhamdulillah top jempol suka senang bahagia menang harapan berharap luar-biasa
sehat berkah juara oke sip setia kompeten berprestasi berintegritas integritas tulus ikhlas bijak bijaksana santun
ramah nyata kerja-nyata pro-rakyat rendah-hati sopan beres tepat solusi hormat respek kagum idola favorit
""".split()

NEGATIVE_WORDS = """
buruk jelek bodoh goblok goblog tolol dungu bego oon korupsi koruptor korup curang kecurangan bohong pembohong
bohongi munafik hancur rusak merusak gagal kalah takut malu memalukan payah parah benci muak kecewa mengecewakan
sedih marah busuk licik zalim dzalim palsu plonga plongo omdo omon pencitraan dinasti oligarki nepotisme kkn
rakus serakah sombong jahat dosa neraka odgj gila sampah ogah nyungsep abal tipu menipu penipu haram bodo
lebay pecundang pengkhianat khianat mengkhianati ancur kacau bobrok rugi miskin susah sengsara tersiksa
mimpi halu halusinasi bacot ngibul ngawur lemah lelet bego anjing bangsat sialan brengsek kampret terpuruk
""".split()

NEGATORS = {"tidak", "bukan", "jangan", "belum", "tanpa", "anti", "kurang"}
_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")
_PHRASES = {"luar biasa": "luar-biasa", "kerja nyata": "kerja-nyata", "pro rakyat": "pro-rakyat",
            "rendah hati": "rendah-hati", "jempol ke atas": "jempol", "hati merah": "suka"}


def load_lexicon(lex_dir: Path) -> Dict[str, float]:
    """Kamus kata -> bobot. Pakai InSet bila ada, selain itu kamus bawaan (+1/-1)."""
    pos, neg = lex_dir / "positive.tsv", lex_dir / "negative.tsv"
    if pos.exists() and neg.exists():
        lex: Dict[str, float] = {}
        for path in (pos, neg):
            d = pd.read_csv(path, sep="\t", header=None, names=["word", "weight"], comment="#",
                            on_bad_lines="skip")
            d = d[pd.to_numeric(d["weight"], errors="coerce").notna()]
            for w, v in zip(d["word"].astype(str).str.lower(), d["weight"].astype(float)):
                lex[w.replace(" ", "-")] = lex.get(w.replace(" ", "-"), 0.0) + v
        return lex
    lex = {w: 1.0 for w in POSITIVE_WORDS}
    lex.update({w: -1.0 for w in NEGATIVE_WORDS})
    return lex


def lexicon_score(text: str, lex: Dict[str, float], window: int = 2) -> float:
    t = str(text).lower()
    for ph, rep in _PHRASES.items():
        t = t.replace(ph, rep)
    toks = []
    for tok in _TOKEN_RE.findall(t):
        tok = re.sub(r"(.)\1{2,}", r"\1", tok)
        toks.extend(SLANG_MAP.get(tok, tok).split())
    score, neg_left = 0.0, 0
    for tok in toks:
        if tok in NEGATORS:
            neg_left = window
            continue
        v = lex.get(tok, 0.0)
        if v:
            score += -v if neg_left > 0 else v
        neg_left = max(0, neg_left - 1)
    return score


def lexicon_predict(texts: Sequence[str], lex: Dict[str, float]) -> pd.DataFrame:
    scores = np.array([lexicon_score(t, lex) for t in texts])
    labels = np.where(scores > 0, "positif", np.where(scores < 0, "negatif", "netral"))
    return pd.DataFrame({"score_lexicon": scores, "label_lexicon": labels})


# --------------------------------------------------------------------------- #
# 2. Model Transformer
# --------------------------------------------------------------------------- #
def transformers_available() -> Optional[str]:
    missing = []
    for mod in ("transformers", "torch"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return None if not missing else ", ".join(missing)


def transformer_predict(texts: Sequence[str], model_name: str, label_map: Dict[str, str],
                        batch_size: int = 32, max_length: int = 128) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    id2label = {int(i): str(l) for i, l in model.config.id2label.items()}
    idx_to_ours = {}
    for i, raw in id2label.items():
        ours = normalize_label(raw, label_map)
        if ours is None:
            raise ValueError(f"Label model '{raw}' tidak dikenali. Tambahkan ke config.SENTIMENT_LABEL_MAP.")
        idx_to_ours[i] = ours
    if sorted(idx_to_ours.values()) != sorted(LABELS):
        raise ValueError(f"Model harus 3 kelas positif/netral/negatif, didapat: {id2label}")
    print(f"   label model -> {idx_to_ours} | device: {device}")

    probs = []
    texts = [str(t) if str(t).strip() else "." for t in texts]
    n = len(texts)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            probs.append(p)
            done = min(start + batch_size, n)
            if (start // batch_size) % 20 == 0 or done == n:
                print(f"   {done}/{n}", flush=True)
    P = np.vstack(probs)
    out = pd.DataFrame({f"prob_{idx_to_ours[i]}": P[:, i] for i in range(P.shape[1])})
    out = out[[f"prob_{l}" for l in LABELS]]
    out["label_model"] = out.idxmax(axis=1).str.replace("prob_", "", regex=False)
    out["confidence"] = out[[f"prob_{l}" for l in LABELS]].max(axis=1)
    return out


# --------------------------------------------------------------------------- #
# 3. Evaluasi
# --------------------------------------------------------------------------- #
def evaluate(y_true: Sequence[str], y_pred: Sequence[str], name: str) -> List[dict]:
    rows = [{"model": name, "metric": "n", "class": "all", "value": len(y_true)},
            {"model": name, "metric": "accuracy", "class": "all", "value": accuracy_score(y_true, y_pred)}]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    for i, lab in enumerate(LABELS):
        rows += [{"model": name, "metric": m, "class": lab, "value": v}
                 for m, v in (("precision", p[i]), ("recall", r[i]), ("f1", f[i]), ("support", s[i]))]
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    rows += [{"model": name, "metric": f"{m}_macro", "class": "all", "value": v}
             for m, v in (("precision", pm), ("recall", rm), ("f1", fm))]
    return rows


def confusion(y_true: Sequence[str], y_pred: Sequence[str], name: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    df.insert(0, "model", name)
    return df.reset_index().rename(columns={"index": "actual"})


def agreement(a: Sequence[str], b: Sequence[str]) -> dict:
    a, b = list(a), list(b)
    return {"n_double_labeled": len(a),
            "percent_agreement": float(np.mean([x == y for x, y in zip(a, b)])) if a else np.nan,
            "cohen_kappa": cohen_kappa_score(a, b, labels=LABELS) if len(a) > 1 else np.nan}
