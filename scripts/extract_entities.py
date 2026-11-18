"""
extract_entities.py
===================
Tahap 7b — Nama tokoh/kandidat yang disebut di setiap komunitas.
Jalankan setelah detect_communities.py:

    python scripts/extract_entities.py

Input : data/processed/community_detection/comments_with_community.csv
        src/topic_extraction/entities.py (kamus tokoh dan partai)
Output: data/processed/topic_extraction/entity_mentions.csv
        outputs/topic_extraction/entity_*.csv
        outputs/topic_extraction/figures/entity_by_community_heatmap.png

Definisi
- Komentar ikut komunitas penulisnya. Penulis yang tidak masuk graf -> "tanpa_interaksi".
- share_comments   : % komentar di komunitas yang menyebut tokoh tsb.
- share_authors    : % akun di komunitas yang pernah menyebut tokoh tsb (mengurangi
                     dominasi akun yang sangat aktif).
- lift             : share_comments di komunitas / share_comments di seluruh data.
                     > 1 berarti tokoh lebih sering dibahas di komunitas itu dibanding rata-rata.
- Sebutan != dukungan. Julukan seperti "wowo"/"mulyono" bernada sindiran; gabungkan
  dengan sentimen sebelum menyimpulkan dukungan/penolakan.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.topic_extraction.entities import PATTERNS, find_entities

NO_INTERACTION = "tanpa_interaksi"
TOP_N = 5
MIN_COUNT_DISTINCT = 5      # minimal komentar agar tokoh dianggap "khas" komunitas
MIN_LIFT_DISTINCT = 1.5
HEATMAP_TOP_ENTITIES = 15


def load() -> pd.DataFrame:
    df = pd.read_csv(config.COMMUNITY_PROCESSED_DIR / "comments_with_community.csv")
    df = df[df["keep_sentiment"]].copy()
    df["community_id"] = df["community_id"].map(config.normalize_community_label)
    df["text_light"] = df["text_light"].fillna("")
    return df


def community_order(ids) -> list:
    ids = set(ids)
    main = sorted(i for i in ids if i not in (config.LABEL_SMALL, NO_INTERACTION))
    return main + [i for i in (config.LABEL_SMALL, NO_INTERACTION) if i in ids]


def mentions_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for group in PATTERNS:
            for ent in find_entities(r.text_light, group):
                rows.append((r.comment_id, r.video_id, r.author_id, r.community_id, group, ent))
    return pd.DataFrame(rows, columns=["comment_id", "video_id", "author_id", "community_id", "group", "entity"])


def by_community(df: pd.DataFrame, m: pd.DataFrame, group: str) -> pd.DataFrame:
    m = m[m["group"] == group]
    base = df.groupby("community_id").agg(n_comments_total=("comment_id", "size"),
                                          n_authors_total=("author_id", "nunique"))
    agg = m.groupby(["community_id", "entity"]).agg(n_comments=("comment_id", "nunique"),
                                                    n_authors=("author_id", "nunique")).reset_index()
    agg = agg.merge(base, left_on="community_id", right_index=True)
    agg["share_comments"] = agg["n_comments"] / agg["n_comments_total"]
    agg["share_authors"] = agg["n_authors"] / agg["n_authors_total"]
    overall = m.groupby("entity")["comment_id"].nunique() / len(df)
    agg["share_comments_overall"] = agg["entity"].map(overall)
    agg["lift"] = agg["share_comments"] / agg["share_comments_overall"]
    agg["rank_in_community"] = agg.groupby("community_id")["n_comments"].rank(method="first", ascending=False).astype(int)
    order = {c: i for i, c in enumerate(community_order(df["community_id"]))}
    agg = agg.sort_values(["community_id", "rank_in_community"], key=lambda s: s.map(order) if s.name == "community_id" else s)
    return agg.reset_index(drop=True)


def overall_table(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    t = m[m["group"] == "tokoh"]
    out = t.groupby("entity").agg(
        n_comments=("comment_id", "nunique"),
        n_authors=("author_id", "nunique"),
        n_videos=("video_id", "nunique"),
        n_communities=("community_id", lambda s: s[~s.isin([config.LABEL_SMALL, NO_INTERACTION])].nunique()),
    )
    out["share_comments"] = out["n_comments"] / len(df)
    return out.sort_values("n_comments", ascending=False).reset_index()


def top_summary(df: pd.DataFrame, bc: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    tok = m[m["group"] == "tokoh"]
    with_any = tok.groupby("community_id")["comment_id"].nunique()
    rows = []
    for cid in community_order(df["community_id"]):
        d = df[df["community_id"] == cid]
        sub = bc[bc["community_id"] == cid].sort_values("n_comments", ascending=False)
        top = [f"{r.entity} ({r.share_comments:.0%})" for r in sub.head(TOP_N).itertuples()]
        distinct = sub[(sub["n_comments"] >= MIN_COUNT_DISTINCT) & (sub["lift"] >= MIN_LIFT_DISTINCT)]
        distinct = distinct.sort_values("lift", ascending=False)
        rows.append({
            "community_id": cid,
            "n_comments": len(d),
            "n_authors": d["author_id"].nunique(),
            "n_videos": d["video_id"].nunique(),
            "share_comments_with_tokoh": with_any.get(cid, 0) / len(d),
            **{f"top{i + 1}": (top[i] if i < len(top) else "") for i in range(TOP_N)},
            "tokoh_khas (lift)": "; ".join(f"{r.entity} (x{r.lift:.1f})" for r in distinct.head(3).itertuples()),
        })
    return pd.DataFrame(rows)


def plot_heatmap(bc: pd.DataFrame, overall: pd.DataFrame, df: pd.DataFrame, path) -> None:
    ents = overall["entity"].head(HEATMAP_TOP_ENTITIES).tolist()
    comms = community_order(df["community_id"])
    mat = (bc.pivot(index="community_id", columns="entity", values="share_comments")
             .reindex(index=comms, columns=ents).fillna(0.0) * 100)
    sizes = df.groupby("community_id").size()

    blues = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", blues)

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(ents), 0.9 + 0.36 * len(comms)))
    vmax = max(10.0, float(np.nanpercentile(mat.values, 98)))
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ents)), [e.split(" (")[0] if "(" in e and len(e) > 22 else e for e in ents],
                  rotation=40, ha="right", fontsize=8.5, color="#3d3d3a")
    ax.set_yticks(range(len(comms)), [f"{c}  (n={sizes.get(c, 0)})" for c in comms], fontsize=8.5, color="#3d3d3a")
    for i in range(len(comms)):
        for j in range(len(ents)):
            v = mat.values[i, j]
            if v >= 10:  # label hanya sel yang menonjol
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > vmax * 0.55 else "#1f1f1d")
    ax.set_xticks(np.arange(-0.5, len(ents), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(comms), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% komentar komunitas yang menyebut tokoh", fontsize=8, color="#3d3d3a")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)
    ax.set_title("Tokoh yang disebut per komunitas (% komentar)", fontsize=11, loc="left", color="#1f1f1d")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    df = load()
    m = mentions_long(df)
    bc = by_community(df, m, "tokoh")
    party = by_community(df, m, "partai")
    overall = overall_table(df, m)
    summary = top_summary(df, bc, m)

    m.to_csv(config.TOPIC_PROCESSED_DIR / "entity_mentions.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(config.TOPIC_OUTPUT_DIR / "entity_overall.csv", index=False, encoding="utf-8-sig")
    bc.to_csv(config.TOPIC_OUTPUT_DIR / "entity_by_community.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(config.TOPIC_OUTPUT_DIR / "entity_top_by_community.csv", index=False, encoding="utf-8-sig")
    party.to_csv(config.TOPIC_OUTPUT_DIR / "party_by_community.csv", index=False, encoding="utf-8-sig")
    plot_heatmap(bc, overall, df, config.TOPIC_FIGURES_DIR / "entity_by_community_heatmap.png")

    n_any = m.loc[m["group"] == "tokoh", "comment_id"].nunique()
    print(f"Komentar dianalisis: {len(df)} | menyebut >= 1 tokoh: {n_any} ({n_any / len(df):.1%})")
    print("\nTokoh paling sering disebut:")
    print(overall.head(12).to_string(index=False))
    print("\nTop tokoh per komunitas:")
    print(summary[["community_id", "n_comments", "top1", "top2", "top3", "tokoh_khas (lift)"]].to_string(index=False))


if __name__ == "__main__":
    main()
