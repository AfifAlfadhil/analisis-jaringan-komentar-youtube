"""
build_graph.py
==============
Tahap 3 — Membangun graf interaksi. Jalankan setelah preprocess.py:

    python build_graph.py

Input : data/processed/comments_clean.csv
Output: data/processed/edges.csv            (source, target, weight, type)
        data/processed/edges_selfloop.csv   (akun membalas/menyebut dirinya sendiri)
        outputs/graph/network.graphml       (untuk NetworkX / Gephi)
        outputs/graph/network.gexf          (untuk Gephi)
        outputs/tables/graph_summary.csv
"""

import networkx as nx
import pandas as pd

import config
from src.network import build_edges, build_graph, graph_summary


def main():
    df = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv")
    df["mentions"] = df["mentions"].fillna("")

    edges, selfloops = build_edges(df)
    g = build_graph(edges, df)
    summary = graph_summary(g)

    edges.to_csv(config.PROCESSED_DIR / "edges.csv", index=False)
    selfloops.to_csv(config.PROCESSED_DIR / "edges_selfloop.csv", index=False)
    nx.write_graphml(g, config.GRAPH_DIR / "network.graphml")
    nx.write_gexf(g, config.GRAPH_DIR / "network.gexf")
    summary.to_csv(config.TABLES_DIR / "graph_summary.csv", index=False)

    print("Edge per tipe:")
    print(edges.groupby("type")["weight"].agg(pairs="size", interactions="sum").to_string())
    print(f"Self-loop dipisahkan: {len(selfloops)} pasangan ({int(selfloops['weight'].sum()) if len(selfloops) else 0} interaksi)")
    print("\nRingkasan graf:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
