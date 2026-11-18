"""
topic_modeling.py
=================
Tahap 7 — Pemetaan isu (TF-IDF + BERTopic). Jalankan setelah join_communities.py:

    python topic_modeling.py                    # TF-IDF + BERTopic
    python topic_modeling.py --no-bertopic      # hanya TF-IDF (cepat, tanpa model)
    python topic_modeling.py --reduce-outliers  # komentar outlier (-1) dipindah ke topik terdekat
    python topic_modeling.py --no-coherence     # lewati koherensi c_v (bila memori tidak cukup)

Input : data/processed/community_detection/comments_with_community.csv (hanya keep_topic == True)
        outputs/topic_extraction/tables/topic_labels.csv (opsional) nama topik manual
Output: outputs/topic_extraction/tables/           tabel frekuensi, kata kunci, topik dan tema
        --- bila BERTopic terpasang ---
        outputs/topic_extraction/tables/topics.csv
        outputs/topic_extraction/tables/topic_examples.csv
        outputs/topic_extraction/tables/topic_labels.csv
        outputs/topic_extraction/tables/topic_by_community.csv
        outputs/topic_extraction/tables/themes.csv
        outputs/topic_extraction/tables/theme_by_community.csv
        data/processed/topic_extraction/comment_topics.csv
        outputs/topic_extraction/figures/topic_by_community_heatmap.png

Alur penamaan topik:
1. Jalankan sekali -> baca topic_examples.csv -> isi kolom label_manual di topic_labels.csv.
2. Jalankan lagi (embedding tersimpan, jadi cepat) -> nama manual dipakai di semua output.
   Topik tetap sama selama data, model, seed, dan parameter topik di config.py tidak berubah.
   Beri NAMA YANG SAMA pada topik yang isinya mirip -> otomatis digabung menjadi satu tema.
   Topik yang tidak bermakna (mis. hanya "no no no") boleh diberi nama "Lainnya".
"""

import argparse
import os
import shutil

# Batasi thread BLAS: di Windows, banyak thread/proses OpenBLAS bisa memicu
# "OpenBLAS error: Memory allocation still failed". Harus di-set sebelum numpy diimpor.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import pandas as pd

import config
from src import plots
from stages.topic_extraction import topics as tp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-bertopic", action="store_true", help="Lewati BERTopic, hanya TF-IDF")
    p.add_argument("--reduce-outliers", action="store_true",
                   help="Pindahkan komentar outlier (-1) ke topik terdekat berdasarkan embedding")
    p.add_argument("--min-topic-size", type=int, default=config.TOPIC_MIN_TOPIC_SIZE)
    p.add_argument("--nr-topics", default=None,
                   help='Jumlah topik maksimum setelah penggabungan (angka atau "auto"); default config.TOPIC_NR_TOPICS')
    p.add_argument("--no-coherence", action="store_true",
                   help="Lewati perhitungan koherensi c_v (gensim) bila terlalu berat/memori habis")
    return p.parse_args()


def load():
    df = pd.read_csv(config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv")
    df = df[df["keep_topic"]].copy()
    df["community_id"] = df["community_id"].map(config.normalize_community_label)
    df["text_full"] = df["text_full"].fillna("")
    df["text_light"] = df["text_light"].fillna("")
    return df[df["text_full"].str.strip() != ""].reset_index(drop=True)


def tfidf_part(df: pd.DataFrame) -> None:
    tf = pd.concat([tp.term_frequency(df["text_full"], 1), tp.term_frequency(df["text_full"], 2)])
    tf.to_csv(config.TOPIC_TABLES_DIR / "term_frequency.csv", index=False, encoding="utf-8-sig")

    kw = tp.ctfidf_by_class(df["text_full"], df["community_id"], top=config.TOPIC_TOP_WORDS)
    order = {c: i for i, c in enumerate(plots.community_order(kw["community_id"]))}
    kw = kw.sort_values(["community_id", "rank"], key=lambda s: s.map(order) if s.name == "community_id" else s)
    kw.to_csv(config.TOPIC_TABLES_DIR / "community_keywords.csv", index=False, encoding="utf-8-sig")

    wide = (kw.groupby("community_id", sort=False)
              .agg(n_comments=("n_comments_community", "first"), keywords=("term", ", ".join))
              .reset_index())
    wide.to_csv(config.TOPIC_TABLES_DIR / "community_keywords_wide.csv", index=False, encoding="utf-8-sig")

    print("\nKata terbanyak (unigram):", ", ".join(tf[tf["ngram"] == 1]["term"].head(15)))
    print("Bigram terbanyak:", ", ".join(tf[tf["ngram"] == 2]["term"].head(10)))
    print("\nKata khas per komunitas (c-TF-IDF):")
    with pd.option_context("display.max_colwidth", 110):
        print(wide.to_string(index=False))


def bertopic_part(df: pd.DataFrame, args) -> None:
    missing = tp.bertopic_available()
    if missing:
        print(f"\n[BERTopic dilewati] Pustaka belum terpasang: {missing}\n"
              "   pip install bertopic sentence-transformers  (lalu jalankan ulang)")
        return

    print(f"\n[BERTopic] {len(df)} komentar | embedding: {config.TOPIC_EMBEDDING_MODEL}")
    emb_model, emb = tp.cached_embeddings(df["text_light"].tolist(), df["comment_id"].astype(str).tolist(),
                                          config.TOPIC_EMBEDDING_MODEL, config.TOPIC_PROCESSED_DIR)
    docs = df["text_full"].tolist()
    nr_topics = config.TOPIC_NR_TOPICS if args.nr_topics is None else (
        "auto" if args.nr_topics == "auto" else int(args.nr_topics))
    print(f"   min_topic_size={args.min_topic_size}, nr_topics={nr_topics}")
    tm, topics, topics_raw, probs = tp.fit_bertopic(
        docs, emb, emb_model, args.min_topic_size, nr_topics,
        config.TOPIC_SEED, args.reduce_outliers)
    df["topic"] = topics
    df["topic_before_outlier_reduction"] = topics_raw

    info = tm.get_topic_info()
    ids = [t for t in info["Topic"] if t != -1]
    if args.no_coherence:
        coh = {}
    else:
        print("   menghitung koherensi c_v (lewati dengan --no-coherence bila memori tidak cukup)...", flush=True)
        coh = tp.topic_coherence(tm, docs, ids)

    # nama manual (bila sudah diisi sebelumnya)
    lab_path = config.TOPIC_TABLES_DIR / "topic_labels.csv"
    legacy_lab_path = config.TABLES_DIR / "topic_labels.csv"
    if legacy_lab_path.exists() and not lab_path.exists():
        shutil.copy2(legacy_lab_path, lab_path)
        print(f"Mempertahankan label topik lama di {lab_path}")
    manual = {}
    if lab_path.exists():
        old = pd.read_csv(lab_path)
        manual = {int(t): str(l) for t, l in zip(old["topic"], old["label_manual"])
                  if pd.notna(l) and str(l).strip()}

    rows = []
    for t in info["Topic"]:
        words = [w for w, _ in tm.get_topic(t)][:config.TOPIC_TOP_WORDS] if t != -1 else []
        auto = "outlier (tidak masuk topik)" if t == -1 else "_".join(words[:3])
        rows.append({"topic": int(t), "label": manual.get(int(t), auto), "label_auto": auto,
                     "n_comments": int((df["topic"] == t).sum()),
                     "share_comments": float((df["topic"] == t).mean()),
                     "keywords": ", ".join(words), "coherence_c_v": coh.get(t)})
    topics_tbl = pd.DataFrame(rows)
    topics_tbl.to_csv(config.TOPIC_TABLES_DIR / "topics.csv", index=False, encoding="utf-8-sig")

    labels = topics_tbl[["topic", "label_auto", "keywords"]].copy()
    labels["label_manual"] = [manual.get(int(t), "") for t in labels["topic"]]
    labels.to_csv(lab_path, index=False, encoding="utf-8-sig")

    ex = []
    for t in ids:
        d = df[df["topic"] == t]
        for r in d.sample(min(8, len(d)), random_state=config.TOPIC_SEED).itertuples():
            ex.append({"topic": t, "label": manual.get(t, ""), "comment_id": r.comment_id,
                       "community_id": r.community_id, "text_light": r.text_light})
    pd.DataFrame(ex).to_csv(config.TOPIC_TABLES_DIR / "topic_examples.csv", index=False, encoding="utf-8-sig")

    label_of = topics_tbl.set_index("topic")["label"]
    df["topic_label"] = df["topic"].map(label_of)
    df[["comment_id", "video_id", "author_id", "community_id", "topic", "topic_label",
        "topic_before_outlier_reduction"]].to_csv(
        config.TOPIC_PROCESSED_DIR / "comment_topics.csv", index=False, encoding="utf-8-sig")

    by = (df.groupby(["community_id", "topic", "topic_label"]).size().rename("n_comments").reset_index())
    by["share_in_community"] = by["n_comments"] / by.groupby("community_id")["n_comments"].transform("sum")
    overall = df["topic"].value_counts(normalize=True)
    by["lift"] = by["share_in_community"] / by["topic"].map(overall)
    order = {c: i for i, c in enumerate(plots.community_order(by["community_id"]))}
    by["_o"] = by["community_id"].map(order)
    by = by.sort_values(["_o", "n_comments"], ascending=[True, False]).drop(columns="_o")
    by.to_csv(config.TOPIC_TABLES_DIR / "topic_by_community.csv", index=False, encoding="utf-8-sig")

    # Tema = topik yang diberi nama sama (penggabungan manual). Tanpa nama manual, 1 tema = 1 topik.
    themes = (topics_tbl.groupby("label", sort=False)
                .agg(topics=("topic", lambda s: ";".join(map(str, s))), n_comments=("n_comments", "sum"))
                .reset_index().rename(columns={"label": "theme"}))
    themes["share_comments"] = themes["n_comments"] / len(df)
    themes = themes.sort_values("n_comments", ascending=False)
    themes.to_csv(config.TOPIC_TABLES_DIR / "themes.csv", index=False, encoding="utf-8-sig")

    tb = df.groupby(["community_id", "topic_label"]).size().rename("n_comments").reset_index()
    tb = tb.rename(columns={"topic_label": "theme"})
    tb["share_in_community"] = tb["n_comments"] / tb.groupby("community_id")["n_comments"].transform("sum")
    tb["lift"] = tb["share_in_community"] / tb["theme"].map(df["topic_label"].value_counts(normalize=True))
    tb["_o"] = tb["community_id"].map(order)
    tb = tb.sort_values(["_o", "n_comments"], ascending=[True, False]).drop(columns="_o")
    tb.to_csv(config.TOPIC_TABLES_DIR / "theme_by_community.csv", index=False, encoding="utf-8-sig")

    outlier_label = label_of.get(-1, None)
    top_themes = themes[themes["theme"] != outlier_label].head(15)["theme"].tolist()
    mat = (tb[tb["theme"].isin(top_themes)]
             .pivot(index="community_id", columns="theme", values="share_in_community")
             .reindex(index=plots.community_order(df["community_id"]), columns=top_themes)
             .fillna(0) * 100)
    mat.columns = [str(c)[:34] for c in mat.columns]
    plots.heatmap(mat, config.TOPIC_FIGURES_DIR / "topic_by_community_heatmap.png",
                  "Distribusi tema/topik per komunitas (% komentar, termasuk outlier di penyebut)",
                  "% komentar komunitas pada tema", row_sizes=df.groupby("community_id").size())

    n_out = int((df["topic"] == -1).sum())
    print(f"\nTopik: {len(ids)} | outlier: {n_out} komentar ({n_out / len(df):.1%})")
    if coh:
        print(f"Koherensi c_v rata-rata: {sum(coh.values()) / len(coh):.3f}")
    else:
        print("Koherensi c_v tidak dihitung" + (" (--no-coherence)." if args.no_coherence
                                                 else " (pip install gensim untuk mengaktifkan)."))
    with pd.option_context("display.max_colwidth", 80):
        print(topics_tbl.head(20)[["topic", "n_comments", "label", "coherence_c_v"]].to_string(index=False))
    print("\nIsi label_manual di outputs/topic_extraction/tables/topic_labels.csv lalu jalankan ulang untuk memakai nama topik.")


def main():
    args = parse_args()
    df = load()
    print(f"Komentar untuk analisis isu (keep_topic): {len(df)}")
    tfidf_part(df)
    if not args.no_bertopic:
        bertopic_part(df, args)


if __name__ == "__main__":
    main()
