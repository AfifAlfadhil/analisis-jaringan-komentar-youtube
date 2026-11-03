"""
preprocessing.py
================
Tahap 2 — Preprocessing komentar YouTube.

Input  : data/raw/comments.csv dari src/youtube_collector.py
         (satu baris = satu komentar tingkat atas atau satu reply).
Output : tabel bersih dengan
         - kolom jaringan (author_id anonim, parent_author_id, mentions)
         - dua versi teks: text_light (untuk sentimen) dan text_full (untuk topik)
         - flag kualitas (spam, copy-paste, pendek, kosong) dan flag keep_*

Prinsip: tidak ada baris yang dibuang karena teks pendek/spam. Baris hanya
diberi flag, supaya jaringan (Tahap 3) tetap utuh sedangkan analisis teks
(Tahap 7-8) tinggal memfilter lewat keep_topic / keep_sentiment.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import emoji
import pandas as pd

from src.lexicons import LAUGHTER_RE, SLANG_MAP, SPAM_PATTERNS, STOPWORDS

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.I)
_EMOJI_TAG_RE = re.compile(r"⟦([^⟧]+)⟧")
_EMOJI_REPEAT_RE = re.compile(r"(⟦[^⟧]+⟧)(?:\s*\1)+")
_MENTION_RE = re.compile(r"@[^\s@]+")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1{1,}")
_NON_LETTER_RE = re.compile(r"[^a-z\s\-]")
_SPACE_RE = re.compile(r"\s+")

MIN_WORDS_TOPIC = 3
COPYPASTE_MIN_WORDS = 5
COPYPASTE_MIN_AUTHORS = 3


# --------------------------------------------------------------------------- #
# 1. Normalisasi struktur data mentah
# --------------------------------------------------------------------------- #
def prepare_long(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standarkan comments.csv mentah: hapus duplikat comment_id (tumpang tindih
    paginasi API), tentukan tipe komentar/reply, kunci akun, dan penulis induk.
    Kunci akun = authorChannelId; bila kosong (akun terhapus) pakai nama tampilan.
    """
    df = raw.drop_duplicates("comment_id").copy()
    df["parent_id"] = df["parent_id"].where(df["parent_id"].notna() & (df["parent_id"] != ""), pd.NA)
    df["type"] = df["parent_id"].map(lambda p: "comment" if pd.isna(p) else "reply")

    name = df["author_name"].fillna("").astype(str).map(lambda s: _ZERO_WIDTH_RE.sub("", s).strip())
    df["author_name"] = name
    chan = df["author_id"].fillna("").astype(str).str.strip()
    df["author_key"] = chan.where(chan != "", "name:" + name.str.lower())
    df.loc[(chan == "") & (name == ""), "author_key"] = pd.NA

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    top_author = df[df["type"] == "comment"].set_index("comment_id")["author_key"]
    df["parent_author_key"] = df["parent_id"].map(top_author)
    df["text_raw"] = df["text"]
    return df.drop(columns=["author_id", "text"])


# --------------------------------------------------------------------------- #
# 2. Akun: anonimisasi
# --------------------------------------------------------------------------- #
def standardize_and_anonymize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Petakan author_key (channel ID) -> user_0001, user_0002, ... (urut kemunculan pertama).
    Return (df dengan author_id & parent_author_id anonim, author_map).
    author_map berisi channel_id dan handle asli: PRIVAT, jangan dipublikasikan.
    """
    df = df.sort_values(["published_at", "comment_id"]).reset_index(drop=True)

    seen: Dict[str, str] = {}
    for k in df["author_key"].dropna():
        if k not in seen:
            seen[k] = f"user_{len(seen) + 1:04d}"

    # nama tampilan terbaru per akun (dipakai untuk mencocokkan @mention)
    handles = df.dropna(subset=["author_key"]).groupby("author_key")["author_name"].last()
    amap = pd.DataFrame({
        "author_id": list(seen.values()),
        "channel_id": list(seen.keys()),
        "handle": [handles.get(k, "") for k in seen],
    })

    df["author_id"] = df["author_key"].map(seen)
    df["parent_author_id"] = df["parent_author_key"].map(seen)
    return df, amap


# --------------------------------------------------------------------------- #
# 3. Teks
# --------------------------------------------------------------------------- #
def strip_mentions(text: str, known: Dict[str, str]) -> Tuple[str, list]:
    """
    Hapus mention `@handle` dari teks dan kembalikan daftar author_id yang
    di-mention. Di YouTube, textDisplay sering menempelkan handle dengan kata
    berikutnya (mis. "@riom259siap"), jadi dicocokkan dengan prefiks terpanjang
    dari handle yang dikenal di dataset.
    `known`: handle huruf kecil (dengan '@') -> author_id.
    """
    found: list = []

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        for end in range(len(tok), 2, -1):
            cand = tok[:end].lower()
            if cand in known:
                found.append(known[cand])
                return " " + tok[end:]
        return " "  # mention tak dikenal: buang tokennya saja

    return _MENTION_RE.sub(repl, text), found


def _demojize_to_words(text: str) -> str:
    # Emoji yang sama berurutan (😂😂😂) cukup dihitung satu kali.
    out = emoji.demojize(text, language="id", delimiters=("⟦", "⟧"))
    out = _EMOJI_REPEAT_RE.sub(r"\1", out)
    return _EMOJI_TAG_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", out)


def make_light(text: str) -> str:
    """Versi bersih ringan untuk model sentimen: huruf besar/kecil dan stopword dipertahankan."""
    t = _demojize_to_words(text)
    t = _ELONGATION_RE.sub(r"\1", t)
    t = _PUNCT_REPEAT_RE.sub(r"\1", t)
    return _SPACE_RE.sub(" ", t).strip()


def make_full(text: str) -> str:
    """Versi bersih penuh untuk topic modelling: lowercase, tanpa emoji/angka/tanda baca, slang dinormalisasi, tanpa stopword."""
    t = emoji.replace_emoji(text, replace=" ").lower()
    t = _NON_LETTER_RE.sub(" ", t)
    t = _ELONGATION_RE.sub(r"\1", t)

    tokens = []
    for tok in t.split():
        tok = tok.strip("-")
        if not tok:
            continue
        if LAUGHTER_RE.match(tok):
            continue
        tok = SLANG_MAP.get(tok, tok)
        for sub in tok.split():  # beberapa entri slang berisi lebih dari satu kata
            if len(sub) > 1 and sub not in STOPWORDS:
                tokens.append(sub)
    return " ".join(tokens)


def _normalize_for_dedup(text: str) -> str:
    t = emoji.replace_emoji(text, replace=" ").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return _SPACE_RE.sub(" ", t).strip()


# --------------------------------------------------------------------------- #
# 4. Pipeline utama
# --------------------------------------------------------------------------- #
def preprocess(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Jalankan seluruh preprocessing.
    Return: (comments_clean, author_map, summary)
    """
    log = []

    def note(k, v):
        log.append({"metric": k, "value": v})

    note("baris_mentah", len(raw))
    df = prepare_long(raw)
    note("duplikat_comment_id_dibuang", len(raw) - len(df))
    note("komentar_tingkat_atas", int((df["type"] == "comment").sum()))
    note("reply", int((df["type"] == "reply").sum()))

    df, amap = standardize_and_anonymize(df)
    note("akun_tanpa_nama_dibuang", int(df["author_id"].isna().sum()))
    df = df[df["author_id"].notna()].copy()
    df = df.reset_index(drop=True)

    # Cleaning teks dasar + ekstraksi mention
    # Semua nama tampilan yang pernah dipakai tiap akun (untuk mencocokkan @mention)
    names = df[["author_name", "author_id"]].drop_duplicates()
    known = {n.lower(): a for n, a in zip(names["author_name"], names["author_id"]) if n}
    stripped, mentions = [], []
    for raw in df["text_raw"].fillna("").astype(str):
        t = _ZERO_WIDTH_RE.sub("", raw)
        t = _URL_RE.sub(" ", t)
        t, found = strip_mentions(t, known)
        stripped.append(t)
        mentions.append(found)
    df["has_url"] = df["text_raw"].fillna("").astype(str).str.contains(_URL_RE)
    df["mentions"] = [";".join(dict.fromkeys(m)) for m in mentions]  # unik, urutan tetap
    df["n_mentions"] = [len(set(m)) for m in mentions]

    # Duplikat persis: akun + video + teks sama (pertahankan yang paling awal)
    df["_norm"] = [_normalize_for_dedup(t) for t in stripped]
    dup_mask = df.duplicated(["video_id", "author_id", "_norm"], keep="first") & (df["_norm"] != "")
    note("duplikat_akun_video_teks_dibuang", int(dup_mask.sum()))
    df = df[~dup_mask].copy()
    stripped = [s for s, d in zip(stripped, dup_mask) if not d]

    df["text_light"] = [make_light(s) for s in stripped]
    df["text_full"] = [make_full(s) for s in stripped]
    df["word_count"] = df["text_light"].str.split().str.len().fillna(0).astype(int)

    # Flag kualitas
    df["is_empty"] = df["_norm"].eq("") & df["text_light"].eq("")
    df["is_short"] = df["word_count"] < MIN_WORDS_TOPIC
    raw_str = df["text_raw"].fillna("").astype(str)
    df["is_spam"] = raw_str.map(lambda s: any(p.search(s) for p in SPAM_PATTERNS))

    long_enough = df["_norm"].str.split().str.len().fillna(0) >= COPYPASTE_MIN_WORDS
    n_authors = df.loc[long_enough].groupby("_norm")["author_id"].nunique()
    copy_texts = set(n_authors[n_authors >= COPYPASTE_MIN_AUTHORS].index)
    df["is_copypaste"] = df["_norm"].isin(copy_texts)

    df["keep_sentiment"] = ~df["is_empty"] & ~df["is_spam"]
    df["keep_topic"] = (
        df["keep_sentiment"]
        & ~df["is_short"]
        & ~df["is_copypaste"]
        & (df["text_full"].str.split().str.len().fillna(0) >= 2)
    )

    note("komentar_akhir", len(df))
    note("akun_unik_penulis", int(df["author_id"].nunique()))
    note("flag_is_empty", int(df["is_empty"].sum()))
    note("flag_is_short", int(df["is_short"].sum()))
    note("flag_is_spam", int(df["is_spam"].sum()))
    note("flag_is_copypaste", int(df["is_copypaste"].sum()))
    note("flag_has_url", int(df["has_url"].sum()))
    note("keep_sentiment", int(df["keep_sentiment"].sum()))
    note("keep_topic", int(df["keep_topic"].sum()))
    note("komentar_dengan_mention_dikenal", int((df["n_mentions"] > 0).sum()))

    cols = [
        "comment_id", "video_id", "parent_id", "type",
        "author_id", "parent_author_id", "mentions", "n_mentions",
        "published_at", "like_count", "reply_count",
        "text_light", "text_full", "word_count",
        "has_url", "is_empty", "is_short", "is_spam", "is_copypaste",
        "keep_sentiment", "keep_topic",
    ]
    df = df[cols].sort_values(["video_id", "published_at", "comment_id"]).reset_index(drop=True)
    return df, amap, pd.DataFrame(log)
