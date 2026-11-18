"""
compare_communities.py
======================
Tahap 9 — Perbandingan isu dan sentimen antar komunitas. Jalankan setelah
sentiment_analysis.py (dan topic_modeling.py / extract_entities.py bila ingin
tabel topik x sentimen dan tokoh x sentimen):

    python compare_communities.py

Input : data/processed/sentiment_predictions.csv   (kolom `sentiment`)
        data/processed/comment_topics.csv          (opsional, dari topic_modeling.py)
        data/processed/entity_mentions.csv         (opsional, dari extract_entities.py)
Output: outputs/tables/sentiment_by_community.csv  proporsi sentimen per komunitas, versi komentar & versi akun
        outputs/tables/sentiment_tests.csv         chi-square, Cramér's V, uji permutasi di dalam video
        outputs/tables/sentiment_residuals.csv     adjusted standardized residual (sel mana yang menyimpang)
        outputs/tables/topic_sentiment.csv         sentimen per topik (dan per komunitas x topik)
        outputs/tables/entity_sentiment_by_community.csv  sentimen komentar yang menyebut tiap tokoh
        outputs/figures/sentiment_by_community.png

Catatan metodologis
- Versi akun: proporsi tiap label dihitung per akun dulu, lalu dirata-rata antar-akun,
  sehingga akun yang sangat aktif tidak mendominasi.
- Chi-square dihitung tiga kali: (a) antar komunitas utama, (b) komunitas utama + small +
  tanpa_interaksi, (c) tiga kelompok (komunitas / small / tanpa_interaksi).
  Cramér's V: < 0,1 sangat lemah, 0,1–0,3 lemah, 0,3–0,5 sedang, > 0,5 kuat (pedoman umum).
- Komunitas sangat terkait dengan video. Uji permutasi mengacak label komunitas DI DALAM
  setiap video; bila p-permutasi besar, perbedaan sentimen antar komunitas cukup dijelaskan
  oleh perbedaan video, bukan oleh komunitasnya.
- Hasil bersifat deskriptif-asosiatif, bukan hubungan sebab-akibat.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import config
from src import plots

LABELS = config.SENTIMENT_LABELS
SMALL, NONE = config.LABEL_SMALL, config.LABEL_NO_INTERACTION


def cramers_v(table: np.ndarray) -> float:
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else np.nan


def chi_test(df: pd.DataFrame, group_col: str, name: str) -> dict:
    tab = pd.crosstab(df[group_col], df["sentiment"]).reindex(columns=LABELS, fill_value=0)
    tab = tab[tab.sum(axis=1) > 0]
    chi2, p, dof, exp = chi2_contingency(tab.values, correction=False)
    return {"test": name, "n": int(tab.values.sum()), "n_groups": tab.shape[0], "chi2": chi2, "dof": dof,
            "p_value": p, "cramers_v": cramers_v(tab.values),
            "share_expected_lt5": float((exp < 5).mean())}


def adjusted_residuals(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    tab = pd.crosstab(df[group_col], df["sentiment"]).reindex(columns=LABELS, fill_value=0)
    O = tab.values.astype(float)
    n = O.sum()
    r, c = O.sum(axis=1, keepdims=True), O.sum(axis=0, keepdims=True)
    E = r @ c / n
    res = (O - E) / np.sqrt(E * (1 - r / n) * (1 - c / n))
    out = pd.DataFrame(res, index=tab.index, columns=[f"resid_{l}" for l in LABELS])
    return out.reindex(plots.community_order(out.index)).reset_index()


def permutation_within_video(df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Acak community_id di dalam tiap video; bandingkan chi-square asli dengan sebaran acak."""
    d = df[["video_id", "community_id", "sentiment"]].reset_index(drop=True)
    sent = pd.Categorical(d["sentiment"], categories=LABELS).codes

    def stat(comm: np.ndarray) -> float:
        codes, uniq = pd.factorize(comm)
        tab = np.zeros((len(uniq), len(LABELS)))
        np.add.at(tab, (codes, sent), 1)
        tab = tab[tab.sum(axis=1) > 0]
        return chi2_contingency(tab, correction=False)[0]

    rng = np.random.default_rng(seed)
    obs = stat(d["community_id"].values)
    groups = [idx for _, idx in d.groupby("video_id").indices.items() if len(idx) > 1]
    comm = d["community_id"].values.copy()
    ge = 0
    perm_stats = []
    for _ in range(n_perm):
        c = comm.copy()
        for idx in groups:
            c[idx] = rng.permutation(c[idx])
        s = stat(c)
        perm_stats.append(s)
        ge += s >= obs
    return {"test": "permutasi di dalam video (komunitas utama)", "n": len(d), "chi2": obs,
            "chi2_perm_mean": float(np.mean(perm_stats)), "p_value": (ge + 1) / (n_perm + 1),
            "n_permutations": n_perm,
            "n_videos_multi_community": int(sum(d.loc[idx, "community_id"].nunique() > 1 for idx in groups))}


def by_community(df: pd.DataFrame) -> pd.DataFrame:
    order = plots.community_order(df["community_id"].unique())
    cnt = pd.crosstab(df["community_id"], df["sentiment"]).reindex(index=order, columns=LABELS, fill_value=0)
    share_c = cnt.div(cnt.sum(axis=1), axis=0)
    per_author = (pd.crosstab([df["community_id"], df["author_id"]], df["sentiment"], normalize="index")
                    .reindex(columns=LABELS, fill_value=0))
    share_a = per_author.groupby(level=0).mean().reindex(order)
    out = pd.DataFrame({"community_id": order,
                        "n_comments": cnt.sum(axis=1).values,
                        "n_authors": df.groupby("community_id")["author_id"].nunique().reindex(order).values,
                        "n_videos": df.groupby("community_id")["video_id"].nunique().reindex(order).values})
    for l in LABELS:
        out[f"n_{l}"] = cnt[l].values
    for l in LABELS:
        out[f"share_{l}_comment"] = share_c[l].values
    for l in LABELS:
        out[f"share_{l}_author"] = share_a[l].values
    out["net_sentiment_comment"] = out["share_positif_comment"] - out["share_negatif_comment"]
    return out


def topic_sentiment(df: pd.DataFrame) -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "comment_topics.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path, usecols=["comment_id", "topic", "topic_label"])
    d = df.merge(t, on="comment_id", how="inner")
    rows = []
    for scope, keys in (("semua", ["topic_label"]), ("per_komunitas", ["community_id", "topic_label"])):
        tab = pd.crosstab([d[k] for k in keys], d["sentiment"]).reindex(columns=LABELS, fill_value=0)
        tot = tab.sum(axis=1)
        for idx, row in tab.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            rec = dict(zip(keys, idx))
            rec.update({"scope": scope, "n_comments": int(tot[idx if len(idx) > 1 else idx[0]])})
            for l in LABELS:
                rec[f"share_{l}"] = row[l] / rec["n_comments"]
            rec["net_sentiment"] = rec["share_positif"] - rec["share_negatif"]
            rows.append(rec)
    out = pd.DataFrame(rows)
    cols = ["scope", "community_id", "topic_label", "n_comments"] + [f"share_{l}" for l in LABELS] + ["net_sentiment"]
    return out.reindex(columns=cols)


def entity_sentiment(df: pd.DataFrame, min_n: int = 5) -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "entity_mentions.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    m = m[m["group"] == "tokoh"][["comment_id", "entity"]].drop_duplicates()
    d = m.merge(df[["comment_id", "community_id", "sentiment"]], on="comment_id", how="inner")
    rows = []
    for scope, keys in (("semua", ["entity"]), ("per_komunitas", ["community_id", "entity"])):
        g = d.groupby(keys)
        tab = pd.crosstab([d[k] for k in keys], d["sentiment"]).reindex(columns=LABELS, fill_value=0)
        n = tab.sum(axis=1)
        tab = tab[n >= min_n]
        for idx, row in tab.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            rec = dict(zip(keys, idx))
            tot = row.sum()
            rec.update({"scope": scope, "n_comments": int(tot)})
            for l in LABELS:
                rec[f"share_{l}"] = row[l] / tot
            rec["net_sentiment"] = rec["share_positif"] - rec["share_negatif"]
            rows.append(rec)
    out = pd.DataFrame(rows).reindex(columns=["scope", "community_id", "entity", "n_comments"]
                                     + [f"share_{l}" for l in LABELS] + ["net_sentiment"])
    return out.sort_values(["scope", "community_id", "n_comments"], ascending=[True, True, False], na_position="first")


def main():
    pred = pd.read_csv(config.PROCESSED_DIR / "sentiment_predictions.csv")
    src = pred["sentiment_source"].iloc[0]
    print(f"Sumber label sentimen: {src}" + ("  (PERHATIAN: masih baseline leksikon)" if src == "lexicon" else ""))
    df = pred[pred["sentiment"].isin(LABELS)].copy()

    bc = by_community(df)
    bc.to_csv(config.TABLES_DIR / "sentiment_by_community.csv", index=False, encoding="utf-8-sig")

    main_comm = df[~df["community_id"].isin([SMALL, NONE])]
    tests = [
        chi_test(main_comm, "community_id", "(a) antar komunitas utama"),
        chi_test(df, "community_id", "(b) komunitas utama + small + tanpa_interaksi"),
        chi_test(df, "community_group", "(c) komunitas vs small vs tanpa_interaksi"),
        permutation_within_video(main_comm, config.COMPARE_N_PERMUTATIONS, config.COMPARE_SEED),
    ]
    tests = pd.DataFrame(tests)
    tests["sentiment_source"] = src
    tests.to_csv(config.TABLES_DIR / "sentiment_tests.csv", index=False, encoding="utf-8-sig")
    adjusted_residuals(df, "community_id").to_csv(config.TABLES_DIR / "sentiment_residuals.csv",
                                                  index=False, encoding="utf-8-sig")

    sizes = bc.set_index("community_id")["n_comments"]
    share = bc.set_index("community_id")[[f"share_{l}_comment" for l in LABELS]]
    share.columns = LABELS
    plots.stacked_sentiment(share, config.FIGURES_DIR / "sentiment_by_community.png",
                            f"Proporsi sentimen komentar per komunitas ({src})", row_sizes=sizes)

    ts = topic_sentiment(df)
    if ts is not None:
        ts.to_csv(config.TABLES_DIR / "topic_sentiment.csv", index=False, encoding="utf-8-sig")
    es = entity_sentiment(df)
    if es is not None:
        es.to_csv(config.TABLES_DIR / "entity_sentiment_by_community.csv", index=False, encoding="utf-8-sig")

    pd.options.display.float_format = "{:.3f}".format
    print("\nProporsi sentimen per komunitas (versi komentar):")
    print(bc[["community_id", "n_comments", "share_positif_comment", "share_netral_comment",
              "share_negatif_comment", "share_negatif_author"]].to_string(index=False))
    print("\nUji statistik:")
    print(tests[["test", "n", "chi2", "p_value", "cramers_v", "share_expected_lt5"]].to_string(index=False))
    print("\nTabel topik x sentimen:", "dibuat" if ts is not None else "dilewati (jalankan topic_modeling.py dulu)")
    print("Tabel tokoh x sentimen:", "dibuat" if es is not None else "dilewati (jalankan extract_entities.py dulu)")
    if es is not None:
        top = es[es["scope"] == "semua"].head(8)
        print(top[["entity", "n_comments", "share_positif", "share_negatif", "net_sentiment"]].to_string(index=False))


if __name__ == "__main__":
    main()
