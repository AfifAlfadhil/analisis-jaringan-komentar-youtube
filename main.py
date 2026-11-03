"""
main.py
=======
Tahap 1 — Pengumpulan data YouTube (Pilpres 2029).

Contoh:
    python main.py                       # 5 kata kunci proposal, parameter default di config.py
    python main.py --max-videos 10 --days 90 --min-comments 100
    python main.py --skip-search         # lanjutkan ambil komentar dari data/raw/videos.csv
    python main.py -k "Pilpres 2029" --max-videos 2 --max-threads 20   # uji kecil

Output: data/raw/videos.csv dan data/raw/comments.csv
Pengambilan komentar bersifat inkremental: video yang sudah ada di comments.csv
dilewati, sehingga aman dijalankan ulang setelah kuota harian habis.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from src.youtube_collector import (
    QuotaExceeded,
    YouTubeCollector,
    append_comments,
    done_video_ids,
    merge_and_save_videos,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pengumpulan data YouTube untuk analisis Pilpres 2029")
    p.add_argument("-k", "--keyword", action="append",
                   help="Kata kunci pencarian (boleh diulang). Default: 5 kata kunci proposal")
    p.add_argument("--max-videos", type=int, default=config.DEFAULT_MAX_VIDEOS_PER_KEYWORD,
                   help="Maks video lolos seleksi per kata kunci")
    p.add_argument("--days", type=int, default=config.DEFAULT_DAYS, help="Rentang publikasi (hari terakhir)")
    p.add_argument("--published-after", help="ISO 8601, mis. 2026-07-01T00:00:00Z (menimpa --days)")
    p.add_argument("--published-before", help="ISO 8601")
    p.add_argument("--min-comments", type=int, default=config.DEFAULT_MIN_COMMENTS,
                   help="Jumlah komentar minimum per video")
    p.add_argument("--max-threads", type=int, default=config.DEFAULT_MAX_THREADS_PER_VIDEO,
                   help="Maks thread komentar tingkat atas per video (semua reply tetap diambil)")
    p.add_argument("--order", choices=["relevance", "time"], default=config.DEFAULT_COMMENT_ORDER)
    p.add_argument("--skip-search", action="store_true",
                   help="Lewati pencarian video; pakai data/raw/videos.csv yang sudah ada")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    collector = YouTubeCollector(config.validate_api_key())

    try:
        if not args.skip_search:
            keywords = args.keyword or config.SEARCH_KEYWORDS
            videos = collector.select_videos(
                keywords,
                max_videos_per_keyword=args.max_videos,
                days=args.days,
                min_comments=args.min_comments,
                published_after=args.published_after,
                published_before=args.published_before,
            )
            df_videos = merge_and_save_videos(videos)
            print(f"videos.csv: {len(df_videos)} video -> {config.RAW_VIDEOS_PATH}")
        else:
            if not config.RAW_VIDEOS_PATH.exists():
                raise SystemExit("data/raw/videos.csv belum ada; jalankan tanpa --skip-search dulu.")
            df_videos = pd.read_csv(config.RAW_VIDEOS_PATH, dtype={"video_id": str})

        done = done_video_ids()
        todo = df_videos[~df_videos["video_id"].isin(done)]
        print(f"\nKomentar: {len(done)} video sudah diambil, {len(todo)} video tersisa.")

        for n, v in enumerate(todo.itertuples(), 1):
            title = v.title if len(v.title) <= 55 else v.title[:52] + "..."
            print(f"[{n}/{len(todo)}] {v.video_id} | {title}", flush=True)
            rows = collector.fetch_comments(v.video_id, max_threads=args.max_threads, order=args.order)
            append_comments(rows)
            n_reply = sum(1 for r in rows if r["parent_id"])
            print(f"   {len(rows) - n_reply} komentar + {n_reply} reply | kuota terpakai: {collector.quota_used}", flush=True)

    except QuotaExceeded:
        print("\nKuota harian API habis. Data yang sudah diambil tersimpan; "
              "jalankan lagi dengan --skip-search besok untuk melanjutkan.")
    finally:
        print(f"\nTotal kuota terpakai pada sesi ini: {collector.quota_used} unit")


if __name__ == "__main__":
    main()
