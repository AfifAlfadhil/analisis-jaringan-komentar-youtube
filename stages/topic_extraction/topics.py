"""
topics.py
=========
Pemetaan isu.

1. Eksplorasi frekuensi kata/bigram (seluruh data).
2. Kata kunci khas per komunitas dengan class-based TF-IDF (c-TF-IDF): semua komentar
   satu komunitas digabung menjadi satu "dokumen kelas", lalu
       W(t,c) = tf(t,c) * log(1 + A / f(t))
   tf(t,c) = frekuensi relatif kata t di kelas c, f(t) = frekuensi t di semua kelas,
   A = rata-rata jumlah kata per kelas (Grootendorst, 2022). Ini metode yang sama dengan
   representasi topik BERTopic, sehingga hasilnya konsisten.
3. BERTopic pada level komentar (bila pustakanya terpasang): embedding dari `text_light`
   (kalimat utuh), kata kunci topik dari `text_full` (sudah dinormalisasi & tanpa stopword).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# --------------------------------------------------------------------------- #
# 1. Frekuensi kata
# --------------------------------------------------------------------------- #
def term_frequency(docs: Sequence[str], ngram: int, top: int = 50, min_df: int = 3) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=(ngram, ngram), min_df=min_df, token_pattern=r"(?u)\b\w\w+\b")
    X = vec.fit_transform(docs)
    freq = np.asarray(X.sum(axis=0)).ravel()
    dfreq = np.asarray((X > 0).sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    out = pd.DataFrame({"term": terms, "frequency": freq, "n_comments": dfreq, "ngram": ngram})
    return out.sort_values("frequency", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. c-TF-IDF per komunitas
# --------------------------------------------------------------------------- #
def ctfidf_by_class(docs: Sequence[str], classes: Sequence[str], top: int = 10,
                    ngram_range=(1, 2), min_df: int = 3) -> pd.DataFrame:
    d = pd.DataFrame({"doc": list(docs), "cls": list(classes)})
    joined = d.groupby("cls")["doc"].apply(" ".join)
    n_docs = d.groupby("cls").size()

    vec = CountVectorizer(ngram_range=ngram_range, min_df=1, token_pattern=r"(?u)\b\w\w+\b")
    vec.fit(d["doc"])
    # min_df dihitung di level komentar agar kata langka (salah ketik) tidak ikut
    doc_counts = np.asarray((vec.transform(d["doc"]) > 0).sum(axis=0)).ravel()
    X = vec.transform(joined.values).toarray().astype(float)
    keep = doc_counts >= min_df
    X, terms, doc_counts = X[:, keep], vec.get_feature_names_out()[keep], doc_counts[keep]

    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    A = X.sum() / X.shape[0]
    idf = np.log(1 + A / np.maximum(X.sum(axis=0), 1))
    W = tf * idf

    rows = []
    for i, cls in enumerate(joined.index):
        order = np.argsort(-W[i])[:top]
        for rank, j in enumerate(order, 1):
            if X[i, j] == 0:
                break
            rows.append({"community_id": cls, "rank": rank, "term": terms[j], "ctfidf": W[i, j],
                         "count_in_community": int(X[i, j]), "n_comments_overall": int(doc_counts[j]),
                         "n_comments_community": int(n_docs[cls])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. BERTopic
# --------------------------------------------------------------------------- #
def bertopic_available() -> Optional[str]:
    """None bila semua pustaka ada; selain itu pesan pustaka yang hilang."""
    missing = []
    for mod in ("bertopic", "sentence_transformers", "umap", "hdbscan"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return None if not missing else ", ".join(missing)


def cached_embeddings(texts: Sequence[str], ids: Sequence[str], model_name: str, cache_dir: Path):
    """Embedding disimpan ke .npy agar menjalankan ulang tidak perlu encode lagi."""
    from sentence_transformers import SentenceTransformer

    key = hashlib.md5(("|".join(ids) + model_name).encode()).hexdigest()[:12]
    path = cache_dir / f"embeddings_{key}.npy"
    model = SentenceTransformer(model_name)
    if path.exists():
        print(f"   memakai embedding tersimpan: {path.name}")
        return model, np.load(path)
    emb = model.encode(list(texts), batch_size=64, show_progress_bar=True)
    np.save(path, emb)
    return model, emb


def fit_bertopic(docs: List[str], emb: np.ndarray, emb_model, min_topic_size: int, nr_topics,
                 seed: int, reduce_outliers: bool):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    hdb = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean", cluster_selection_method="eom",
                  prediction_data=True)
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w\w+\b")
    tm = BERTopic(embedding_model=emb_model, umap_model=umap_model, hdbscan_model=hdb,
                  vectorizer_model=vec, nr_topics=nr_topics, calculate_probabilities=False,
                  verbose=True)
    topics, probs = tm.fit_transform(docs, embeddings=emb)
    topics_raw = list(topics)
    if reduce_outliers and -1 in topics:
        topics = tm.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)
        tm.update_topics(docs, topics=topics, vectorizer_model=vec)
    return tm, list(topics), topics_raw, probs


def topic_coherence(tm, docs: List[str], topic_ids: List[int], top_n: int = 10) -> Dict[int, float]:
    """Koherensi c_v per topik (butuh gensim). Bigram dipecah menjadi unigram."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        return {}
    tokens = [d.split() for d in docs]
    dic = Dictionary(tokens)
    words, kept = [], []
    for t in topic_ids:
        ws = []
        for w, _ in tm.get_topic(t)[:top_n]:
            for part in w.split():
                if part in dic.token2id and part not in ws:
                    ws.append(part)
        if len(ws) >= 2:
            words.append(ws)
            kept.append(t)
    if not words:
        return {}
    cm = CoherenceModel(topics=words, texts=tokens, dictionary=dic, coherence="c_v", processes=1)
    return dict(zip(kept, cm.get_coherence_per_topic()))
