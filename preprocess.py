"""
preprocess.py
=============
Tahap 2 — Preprocessing. Jalankan dari folder proyek:

    python preprocess.py

Input : data/raw/comments.csv (data mentah, tidak diubah)
Output: data/processed/comments_clean.csv
        data/processed/author_map.csv        (PRIVAT: pemetaan user_xxxx -> channel ID & handle asli,
                                              jangan dipublikasikan / dimasukkan ke laporan)
        outputs/tables/preprocess_summary.csv
"""

import pandas as pd

import config
from src.preprocessing import preprocess


def main():
    print(f"Membaca data mentah: {config.RAW_COMMENTS_PATH}")
    raw = pd.read_csv(config.RAW_COMMENTS_PATH, dtype=str, keep_default_na=False)
    raw["like_count"] = pd.to_numeric(raw["like_count"], errors="coerce")
    raw["reply_count"] = pd.to_numeric(raw["reply_count"], errors="coerce").fillna(0).astype(int)

    df, amap, summary = preprocess(raw)

    out_clean = config.PROCESSED_DIR / "comments_clean.csv"
    out_map = config.PROCESSED_DIR / "author_map.csv"
    out_sum = config.TABLES_DIR / "preprocess_summary.csv"

    df.to_csv(out_clean, index=False, encoding="utf-8-sig")
    amap.to_csv(out_map, index=False, encoding="utf-8-sig")
    summary.to_csv(out_sum, index=False, encoding="utf-8-sig")

    print("\nRingkasan preprocessing:")
    print(summary.to_string(index=False))
    print(f"\nTersimpan: {out_clean}")
    print(f"Tersimpan: {out_map}  (PRIVAT)")
    print(f"Tersimpan: {out_sum}")


if __name__ == "__main__":
    main()
