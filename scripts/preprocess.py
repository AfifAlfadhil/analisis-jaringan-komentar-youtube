"""Create the text-enriched comment file used by NLP stages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.preprocessing.cleaning import clean_comments


def main() -> None:
    output = config.PREPROCESSING_PROCESSED_DIR / "comments_clean.csv"
    comments = clean_comments(config.RAW_COMMENTS_PATH, output)
    print(f"Komentar diproses: {len(comments)}")
    print(f"Tersimpan: {output}")


if __name__ == "__main__":
    main()
