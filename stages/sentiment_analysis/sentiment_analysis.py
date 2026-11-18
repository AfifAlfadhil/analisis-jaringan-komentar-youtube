"""
sentiment_analysis.py
=====================
Tahap 8 — Analisis sentimen (positif / netral / negatif). Jalankan setelah join_communities.py.

Langkah:
    python sentiment_analysis.py --make-sample   # 1) buat sampel untuk dilabeli manual
    (isi kolom label_1, dan label_2 bila ada pelabel kedua, di file sampel)
    python sentiment_analysis.py                 # 2) prediksi semua komentar + evaluasi

Opsi:
    --model NAMA     ganti model Hugging Face (default config.SENTIMENT_MODEL)
    --no-model       hanya baseline leksikon (tanpa transformers/torch)
    --force          timpa file sampel yang sudah ada (HATI-HATI: label manual hilang)

Input : data/processed/community_detection/comments_with_community.csv (keep_sentiment == True)
        data/processed/sentiment_analysis/sentiment_label_sample.csv  (label manual, opsional)
        data/lexicon/positive.tsv, negative.tsv     (InSet, opsional)
Output: data/processed/sentiment_analysis/sentiment_predictions.csv label & probabilitas tiap komentar
                                                    (kolom `sentiment` = label final untuk Tahap 9)
        outputs/sentiment_analysis/tables/          hasil prediksi, evaluasi, dan distribusi

Pedoman pelabelan manual (tulis juga di BAB III):
- positif : mendukung, memuji, berharap baik, setuju terhadap tokoh/isu yang dibahas
- negatif : menolak, mengkritik, menghina, menyindir (sarkasme dilabeli sesuai MAKSUDNYA)
- netral  : pertanyaan, informasi, prediksi tanpa sikap, atau tidak jelas
Label boleh ditulis positif/netral/negatif, pos/net/neg, atau 1/0/-1.
Kolom `label` (opsional) = label final hasil diskusi bila label_1 dan label_2 berbeda.
"""

import argparse
import shutil

import numpy as np
import pandas as pd

import config
from stages.sentiment_analysis import sentiment as st

SAMPLE_PATH = config.SENTIMENT_PROCESSED_DIR / "sentiment_label_sample.csv"
LEGACY_SAMPLE_PATH = config.PROCESSED_DIR / "sentiment_label_sample.csv"
LEXICON_DIR = config.BASE_DIR / "data" / "lexicon"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--make-sample", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--model", default=config.SENTIMENT_MODEL)
    p.add_argument("--no-model", action="store_true")
    return p.parse_args()


def load() -> pd.DataFrame:
    df = pd.read_csv(config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv")
    df = df[df["keep_sentiment"]].copy()
    for column in ("community_id", "community_group"):
        if column in df:
            df[column] = df[column].map(config.normalize_community_label)
    df["text_light"] = df["text_light"].fillna("")
    return df.reset_index(drop=True)


def make_sample(df: pd.DataFrame, n: int, force: bool) -> None:
    """
    Sampel acak berstrata per komunitas: tiap komunitas utama dapat minimal 10 komentar
    (atau semua bila lebih sedikit); sisa kuota dibagi proporsional ke small & tanpa_interaksi.
    """
    if SAMPLE_PATH.exists() and not force:
        raise SystemExit(f"{SAMPLE_PATH.name} sudah ada. Pakai --force untuk menimpa (label manual akan hilang).")
    rng = np.random.default_rng(config.SENTIMENT_SEED)
    main = [c for c in df["community_id"].unique() if c not in (config.LABEL_SMALL, config.LABEL_NO_INTERACTION)]
    per_main = 10
    picks = []
    for c in sorted(main):
        d = df[df["community_id"] == c]
        picks.append(d.sample(min(per_main, len(d)), random_state=int(rng.integers(1e9))))
    rest_n = max(0, n - sum(len(p) for p in picks))
    rest = df[~df.index.isin(pd.concat(picks).index)]
    picks.append(rest.sample(min(rest_n, len(rest)), random_state=config.SENTIMENT_SEED))
    s = pd.concat(picks).sample(frac=1, random_state=config.SENTIMENT_SEED)
    s = s[["comment_id", "community_id", "video_id", "type", "text_light"]].copy()
    s["label_1"] = ""
    s["label_2"] = ""
    s["label"] = ""
    s["catatan"] = ""
    s.to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print(f"Sampel {len(s)} komentar -> {SAMPLE_PATH}")
    print(s["community_id"].value_counts().sort_index().to_string())
    print("\nIsi label_1 (dan label_2 bila ada pelabel kedua), lalu jalankan: python sentiment_analysis.py")


def gold_labels(sample: pd.DataFrame) -> pd.DataFrame:
    for c in ("label_1", "label_2", "label"):
        if c not in sample:
            sample[c] = None
        sample[c] = sample[c].map(st.normalize_label)
    sample["gold"] = sample["label"].fillna(sample["label_1"])
    return sample


def main():
    args = parse_args()
    if LEGACY_SAMPLE_PATH.exists() and not SAMPLE_PATH.exists():
        shutil.copy2(LEGACY_SAMPLE_PATH, SAMPLE_PATH)
        print(f"Mempertahankan sampel label lama di {SAMPLE_PATH}")
    df = load()
    if args.make_sample:
        make_sample(df, config.SENTIMENT_SAMPLE_SIZE, args.force)
        return

    print(f"Komentar untuk sentimen: {len(df)}")

    # 1) baseline leksikon
    lex = st.load_lexicon(LEXICON_DIR)
    src = "InSet" if (LEXICON_DIR / "positive.tsv").exists() else "kamus bawaan"
    print(f"[1] Baseline leksikon ({src}, {len(lex)} kata)")
    pred = pd.concat([df[["comment_id", "video_id", "author_id", "community_id", "community_group"]],
                      st.lexicon_predict(df["text_light"], lex)], axis=1)
    methods = {"lexicon": "label_lexicon"}

    # 2) model transformer
    if not args.no_model:
        missing = st.transformers_available()
        if missing:
            print(f"[2] Model dilewati: pustaka belum terpasang ({missing}).\n"
                  "    pip install transformers torch  lalu jalankan ulang.")
        else:
            print(f"[2] Model: {args.model}")
            mp = st.transformer_predict(df["text_light"].tolist(), args.model, config.SENTIMENT_LABEL_MAP,
                                        config.SENTIMENT_BATCH_SIZE, config.SENTIMENT_MAX_LENGTH)
            pred = pd.concat([pred, mp], axis=1)
            pred["model_name"] = args.model
            methods["indobert"] = "label_model"

    final_col = methods.get("indobert", "label_lexicon")
    pred["sentiment"] = pred[final_col]
    pred["sentiment_source"] = "indobert" if final_col == "label_model" else "lexicon"
    pred.to_csv(config.SENTIMENT_PROCESSED_DIR / "sentiment_predictions.csv", index=False, encoding="utf-8-sig")

    dist = pd.concat([pred[c].value_counts().reindex(st.LABELS).fillna(0).rename(m) for m, c in methods.items()],
                     axis=1).T
    share = dist.div(dist.sum(axis=1), axis=0).add_prefix("share_")
    dist = pd.concat([dist.astype(int).add_prefix("n_"), share], axis=1).rename_axis("method").reset_index()
    dist.to_csv(config.SENTIMENT_TABLES_DIR / "sentiment_distribution.csv", index=False, encoding="utf-8-sig")
    print("\nSebaran label:")
    print(dist.to_string(index=False))

    # 3) evaluasi terhadap label manual
    if not SAMPLE_PATH.exists():
        print(f"\n[3] Evaluasi dilewati: buat sampel dulu (python sentiment_analysis.py --make-sample).")
        return
    sample = gold_labels(pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False))
    lab = sample[sample["gold"].notna()]
    if lab.empty:
        print(f"\n[3] Evaluasi dilewati: kolom label_1 di {SAMPLE_PATH.name} belum diisi.")
        return

    both = sample[sample["label_1"].notna() & sample["label_2"].notna()]
    agr = st.agreement(both["label_1"], both["label_2"])
    pd.DataFrame([agr]).to_csv(config.SENTIMENT_TABLES_DIR / "sentiment_agreement.csv", index=False)

    ev = lab[["comment_id", "gold"]].merge(pred.astype({"comment_id": str}), on="comment_id", how="inner")
    rows, cms = [], []
    for m, c in methods.items():
        rows += st.evaluate(ev["gold"], ev[c], m)
        cms.append(st.confusion(ev["gold"], ev[c], m))
    evt = pd.DataFrame(rows)
    evt.to_csv(config.SENTIMENT_TABLES_DIR / "sentiment_eval.csv", index=False, encoding="utf-8-sig")
    pd.concat(cms).to_csv(config.SENTIMENT_TABLES_DIR / "sentiment_confusion.csv", index=False, encoding="utf-8-sig")

    print(f"\n[3] Evaluasi pada {len(ev)} komentar berlabel manual")
    if agr["n_double_labeled"]:
        print(f"    Kesepakatan pelabel: {agr['percent_agreement']:.1%}, Cohen's kappa = {agr['cohen_kappa']:.3f} "
              f"(n = {agr['n_double_labeled']})")
    summ = evt[evt["class"] == "all"].pivot(index="model", columns="metric", values="value")
    print(summ[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].round(3).to_string())


if __name__ == "__main__":
    main()
