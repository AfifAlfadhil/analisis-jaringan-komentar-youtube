"""Fungsi frekuensi kata dan c-TF-IDF untuk pemetaan isu."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def term_frequency(
    docs: Sequence[str],
    ngram: int,
    top: int = 50,
    min_df: int = 3,
) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        ngram_range=(ngram, ngram),
        min_df=min_df,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform(docs)
    frequency = np.asarray(matrix.sum(axis=0)).ravel()
    comment_count = np.asarray((matrix > 0).sum(axis=0)).ravel()
    return (
        pd.DataFrame({
            "term": vectorizer.get_feature_names_out(),
            "frequency": frequency,
            "n_comments": comment_count,
            "ngram": ngram,
        })
        .sort_values("frequency", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def ctfidf_by_class(
    docs: Sequence[str],
    classes: Sequence[str],
    top: int = 10,
    min_df: int = 3,
) -> pd.DataFrame:
    data = pd.DataFrame({"doc": list(docs), "community_id": list(classes)})
    joined = data.groupby("community_id")["doc"].apply(" ".join)
    comment_counts = data.groupby("community_id").size()

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    vectorizer.fit(data["doc"])
    document_matrix = vectorizer.transform(data["doc"])
    class_matrix = vectorizer.transform(joined.values).toarray().astype(float)
    document_counts = np.asarray((document_matrix > 0).sum(axis=0)).ravel()
    keep = document_counts >= min_df
    class_matrix = class_matrix[:, keep]
    terms = vectorizer.get_feature_names_out()[keep]
    document_counts = document_counts[keep]

    tf = class_matrix / np.maximum(class_matrix.sum(axis=1, keepdims=True), 1)
    average_words = class_matrix.sum() / max(len(class_matrix), 1)
    idf = np.log(1 + average_words / np.maximum(class_matrix.sum(axis=0), 1))
    scores = tf * idf

    rows = []
    for row_index, community_id in enumerate(joined.index):
        for rank, term_index in enumerate(np.argsort(-scores[row_index])[:top], 1):
            if class_matrix[row_index, term_index] == 0:
                break
            rows.append({
                "community_id": community_id,
                "rank": rank,
                "term": terms[term_index],
                "ctfidf": scores[row_index, term_index],
                "count_in_community": int(class_matrix[row_index, term_index]),
                "n_comments_overall": int(document_counts[term_index]),
                "n_comments_community": int(comment_counts[community_id]),
            })
    return pd.DataFrame(rows)


def bertopic_available() -> str | None:
    missing = []
    for module in ("bertopic", "sentence_transformers", "umap", "hdbscan"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return ", ".join(missing) if missing else None


def cached_embeddings(
    texts: Sequence[str],
    ids: Sequence[str],
    model_name: str,
    cache_dir: Path,
):
    from sentence_transformers import SentenceTransformer

    key = hashlib.md5(("|".join(ids) + model_name).encode()).hexdigest()[:12]
    path = cache_dir / f"embeddings_{key}.npy"
    model = SentenceTransformer(model_name)
    if path.exists():
        return model, np.load(path)
    embeddings = model.encode(list(texts), batch_size=64, show_progress_bar=True)
    np.save(path, embeddings)
    return model, embeddings


def fit_bertopic(
    docs: List[str],
    embeddings: np.ndarray,
    embedding_model,
    min_topic_size: int,
    nr_topics: int,
    seed: int,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            token_pattern=r"(?u)\b\w\w+\b",
        ),
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probabilities = model.fit_transform(docs, embeddings=embeddings)
    return model, list(topics), probabilities


def topic_coherence(model, docs: List[str], topic_ids: List[int]) -> Dict[int, float]:
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        return {}
    tokens = [doc.split() for doc in docs]
    dictionary = Dictionary(tokens)
    words = []
    kept = []
    for topic_id in topic_ids:
        topic_words = [
            word for word, _ in model.get_topic(topic_id)[:10]
            if word in dictionary.token2id
        ]
        if len(topic_words) >= 2:
            words.append(topic_words)
            kept.append(topic_id)
    if not words:
        return {}
    coherence = CoherenceModel(
        topics=words,
        texts=tokens,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return dict(zip(kept, coherence.get_coherence_per_topic()))
