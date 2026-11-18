"""Lexicon-based sentiment classification and evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABELS = ["positif", "netral", "negatif"]

POSITIVE = {
    "bagus", "baik", "dukung", "dukungan", "hebat", "mantap", "setuju",
    "sukses", "menang", "benar", "cinta", "harapan",
}
NEGATIVE = {
    "buruk", "bodoh", "benci", "gagal", "jahat", "korup", "marah",
    "salah", "tolak", "bohong", "hancur", "hina", "takut",
}


def normalize_label(value: object) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip().lower()
    aliases = {
        "pos": "positif", "positive": "positif", "1": "positif",
        "net": "netral", "neutral": "netral", "0": "netral",
        "neg": "negatif", "negative": "negatif", "-1": "negatif",
    }
    return aliases.get(value, value if value in LABELS else None)


def load_lexicon(path: Path) -> dict[str, int]:
    scores = {word: 1 for word in POSITIVE}
    scores.update({word: -1 for word in NEGATIVE})
    for filename, score in (("positive.tsv", 1), ("negative.tsv", -1)):
        file_path = path / filename
        if not file_path.exists():
            continue
        for line in file_path.read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word and not word.startswith("#"):
                scores[word] = score
    return scores


def lexicon_predict(texts: Iterable[str], lexicon: dict[str, int]) -> pd.DataFrame:
    rows = []
    for text in texts:
        words = re.findall(r"\w+", str(text).lower())
        score = sum(lexicon.get(word, 0) for word in words)
        label = "positif" if score > 0 else "negatif" if score < 0 else "netral"
        rows.append({"label_lexicon": label, "score_lexicon": score})
    return pd.DataFrame(rows)


def agreement(left: Iterable[str], right: Iterable[str]) -> dict:
    left = list(left)
    right = list(right)
    if not left:
        return {"n_double_labeled": 0, "percent_agreement": None, "cohen_kappa": None}
    return {
        "n_double_labeled": len(left),
        "percent_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
        "cohen_kappa": cohen_kappa_score(left, right),
    }


def evaluate(gold: Iterable[str], predicted: Iterable[str], model: str) -> list[dict]:
    gold = list(gold)
    predicted = list(predicted)
    metrics = {
        "accuracy": accuracy_score(gold, predicted),
        "precision_macro": precision_score(gold, predicted, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(gold, predicted, labels=LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0),
    }
    return [
        {"model": model, "class": "all", "metric": name, "value": value}
        for name, value in metrics.items()
    ]


def confusion(gold: Iterable[str], predicted: Iterable[str], model: str) -> pd.DataFrame:
    matrix = confusion_matrix(gold, predicted, labels=LABELS)
    return pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis(
        "gold"
    ).reset_index().melt(id_vars="gold", var_name="predicted", value_name="n").assign(
        model=model
    )
