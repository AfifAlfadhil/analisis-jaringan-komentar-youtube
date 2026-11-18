"""Lightweight text preparation for topic and sentiment analysis."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.network_analysis.network import prepare_accounts

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"[@#][\w-]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_comments(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    comments, _ = prepare_accounts(raw)
    comments["text_full"] = comments["text"].map(normalize_text)
    comments["text_light"] = comments["text_full"].str.replace(
        r"[^\w\s]", " ", regex=True
    ).str.replace(SPACE_RE, " ", regex=True).str.strip()
    token_count = comments["text_light"].str.split().str.len().fillna(0)
    comments["keep_sentiment"] = comments["text_light"].ne("") & token_count.ge(2)
    comments["keep_topic"] = comments["text_light"].ne("") & token_count.ge(3)
    comments["mentions"] = comments["text"].str.findall(r"[@#][\w-]+").str.join(";")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comments.to_csv(output_path, index=False, encoding="utf-8-sig")
    return comments
