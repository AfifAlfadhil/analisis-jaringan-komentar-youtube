"""Pemetaan kata kunci isu per komunitas dengan c-TF-IDF."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd

import config
from src.common import plots
from src.topic_extraction import topics as tp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bertopic", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv")
    df = df[df["keep_topic"]].copy()
    df["community_id"] = df["community_id"].map(config.normalize_community_label)
    df["text_full"] = df["text_full"].fillna("")
    df = df[df["text_full"].str.strip().ne("")].reset_index(drop=True)

    frequencies = pd.concat([
        tp.term_frequency(df["text_full"], 1),
        tp.term_frequency(df["text_full"], 2),
    ])
    keywords = tp.ctfidf_by_class(
        df["text_full"],
        df["community_id"],
        top=config.TOPIC_TOP_WORDS,
    )
    order = {
        community: index
        for index, community in enumerate(plots.community_order(keywords["community_id"]))
    }
    keywords = keywords.sort_values(
        ["community_id", "rank"],
        key=lambda values: values.map(order) if values.name == "community_id" else values,
    )
    wide = (
        keywords.groupby("community_id", sort=False)
        .agg(
            n_comments=("n_comments_community", "first"),
            keywords=("term", ", ".join),
        )
        .reset_index()
    )

    frequencies.to_csv(
        config.TOPIC_OUTPUT_DIR / "term_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    keywords.to_csv(
        config.TOPIC_OUTPUT_DIR / "community_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wide.to_csv(
        config.TOPIC_OUTPUT_DIR / "community_keywords_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Komentar untuk analisis isu: {len(df)}")
    print("Kata terbanyak:", ", ".join(
        frequencies[frequencies["ngram"] == 1]["term"].head(15)
    ))
    print(wide.to_string(index=False))
    if args.no_bertopic:
        return

    missing = tp.bertopic_available()
    if missing:
        raise RuntimeError(
            f"BERTopic belum dapat dijalankan. Pustaka belum tersedia: {missing}. "
            "Pasang requirements-nlp.txt atau gunakan --no-bertopic."
        )

    embedding_model, embeddings = tp.cached_embeddings(
        df["text_light"].tolist(),
        df["comment_id"].astype(str).tolist(),
        config.TOPIC_EMBEDDING_MODEL,
        config.TOPIC_PROCESSED_DIR,
    )
    model, topic_ids, _ = tp.fit_bertopic(
        df["text_full"].tolist(),
        embeddings,
        embedding_model,
        config.TOPIC_MIN_TOPIC_SIZE,
        config.TOPIC_NR_TOPICS,
        config.TOPIC_SEED,
    )
    info = model.get_topic_info()
    valid_topics = [int(topic) for topic in info["Topic"] if topic != -1]
    coherence = tp.topic_coherence(model, df["text_full"].tolist(), valid_topics)
    labels = {}
    topic_rows = []
    for topic in info["Topic"]:
        topic = int(topic)
        words = [] if topic == -1 else [word for word, _ in model.get_topic(topic)]
        label = "outlier" if topic == -1 else " / ".join(words[:3])
        labels[topic] = label
        topic_rows.append({
            "topic": topic,
            "label": label,
            "keywords": ", ".join(words[:config.TOPIC_TOP_WORDS]),
            "n_comments": int(sum(item == topic for item in topic_ids)),
            "coherence_c_v": coherence.get(topic),
        })
    pd.DataFrame(topic_rows).to_csv(config.TOPIC_OUTPUT_DIR / "topics.csv", index=False)

    df["topic"] = topic_ids
    df["topic_label"] = df["topic"].map(labels)
    df[["comment_id", "video_id", "author_id", "community_id", "topic", "topic_label"]].to_csv(
        config.TOPIC_PROCESSED_DIR / "comment_topics.csv",
        index=False,
    )
    by_community = (
        df.groupby(["community_id", "topic", "topic_label"])
        .size()
        .rename("n_comments")
        .reset_index()
    )
    by_community["share_in_community"] = (
        by_community["n_comments"]
        / by_community.groupby("community_id")["n_comments"].transform("sum")
    )
    by_community.to_csv(config.TOPIC_OUTPUT_DIR / "topic_by_community.csv", index=False)
    themes = (
        pd.DataFrame(topic_rows)
        .rename(columns={"label": "theme"})
        [["theme", "topic", "n_comments", "keywords", "coherence_c_v"]]
    )
    themes["share_comments"] = themes["n_comments"] / len(df)
    themes.to_csv(config.TOPIC_OUTPUT_DIR / "themes.csv", index=False)
    theme_by_community = by_community.rename(columns={"topic_label": "theme"})
    theme_by_community.to_csv(
        config.TOPIC_OUTPUT_DIR / "theme_by_community.csv",
        index=False,
    )
    print(f"BERTopic selesai: {len(valid_topics)} topik")


if __name__ == "__main__":
    main()
