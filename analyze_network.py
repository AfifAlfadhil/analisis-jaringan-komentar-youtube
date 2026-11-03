"""
analyze_network.py
==================
Tahap 4 — Analisis jaringan (SNA). Jalankan setelah build_graph.py:

    python analyze_network.py

Input : outputs/graph/network.graphml
        data/processed/comments_clean.csv (untuk edge per video)
        data/raw/videos.csv + data/processed/author_map.csv (hanya untuk menandai
        akun pemilik channel video; identitas asli tidak ditulis ke output)
Output: outputs/tables/network_stats.csv
        outputs/tables/centrality.csv
        outputs/tables/top_actors.csv
        outputs/tables/degree_distribution.csv
        outputs/tables/bridging_accounts.csv  (akun yang berinteraksi di >1 video)
        outputs/figures/degree_distribution.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
from src.sna import (
    bridging_accounts,
    bridging_stats,
    centrality_table,
    degree_distribution,
    edges_by_video,
    network_stats,
    top_actors,
)


def owner_ids() -> set:
    """author_id (anonim) milik channel yang mengunggah video di dataset."""
    videos = pd.read_csv(config.RAW_VIDEOS_PATH, dtype=str)
    amap = pd.read_csv(config.PROCESSED_DIR / "author_map.csv", dtype=str)
    return set(amap.loc[amap["channel_id"].isin(set(videos["channel_id"])), "author_id"])


def plot_degree_distribution(dist: pd.DataFrame, alpha: float, xmin: float, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(dist["degree"], dist["share"], s=22, color="#2b6cb0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("Proporsi node (log)")
    title = "Distribusi degree"
    if not np.isnan(alpha):
        title += f" (power-law fit: alpha={alpha:.2f}, xmin={int(xmin)})"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    g = nx.read_graphml(config.GRAPH_DIR / "network.graphml")
    print(f"Graf dimuat: {g.number_of_nodes()} node, {g.number_of_edges()} edge")

    stats = network_stats(g)
    cent = centrality_table(g)
    owners = owner_ids()
    cent["is_channel_owner"] = cent["author_id"].isin(owners)
    top = top_actors(cent, n=10)
    dist = degree_distribution(g)

    comments = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv")
    comments["mentions"] = comments["mentions"].fillna("")
    ev = edges_by_video(comments)
    bridge = bridging_accounts(ev, cent)
    stats = pd.concat([stats, bridging_stats(ev, g, bridge)], ignore_index=True)

    bridge.to_csv(config.TABLES_DIR / "bridging_accounts.csv", index=False)
    stats.to_csv(config.TABLES_DIR / "network_stats.csv", index=False)
    cent.to_csv(config.TABLES_DIR / "centrality.csv", index=False)
    top.to_csv(config.TABLES_DIR / "top_actors.csv", index=False)
    dist.to_csv(config.TABLES_DIR / "degree_distribution.csv", index=False)

    pl = stats[stats["metric"].str.startswith("powerlaw_")].set_index("metric")["value"]
    plot_degree_distribution(dist, pl["powerlaw_alpha"], pl["powerlaw_xmin"],
                             config.FIGURES_DIR / "degree_distribution.png")

    pd.options.display.float_format = "{:.4f}".format
    print("\nStatistik jaringan:")
    print(stats.pivot(index="metric", columns="scope", values="value").fillna("").to_string())
    print(f"\nAkun pemilik channel di graf: {int(cent['is_channel_owner'].sum())}")
    print("Pemilik channel di top-10 per metrik:")
    print(top.groupby("metric")["is_channel_owner"].sum().to_string())
    print("\nAkun penjembatan antar-video:")
    print(stats[stats["scope"] == "bridging"][["metric", "value"]].to_string(index=False))
    print(bridge.head(5)[["author_id", "n_videos_interaction", "degree", "betweenness",
                          "is_channel_owner"]].to_string(index=False))
    print("\nTop-5 degree:")
    print(cent.head(5)[["author_id", "degree", "in_degree", "out_degree", "betweenness", "pagerank",
                        "n_comments", "n_videos", "is_channel_owner"]].to_string(index=False))


if __name__ == "__main__":
    main()
