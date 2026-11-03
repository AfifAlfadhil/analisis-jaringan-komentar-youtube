"""
network.py
==========
Tahap 3 — Membangun graf interaksi antar-akun dari comments_clean.csv.

Definisi operasional edge (directed):
- reply   : penulis reply -> penulis komentar induk (thread)
- mention : penulis komentar -> akun lain yang di-mention (@handle) di teks,
            hanya untuk akun yang dikenal di dataset dan BUKAN penulis induk
            pada reply yang sama (agar tidak dihitung dua kali).
Bobot = jumlah interaksi per pasangan (source, target, type).
Self-loop dipisahkan dan tidak masuk graf.
Komentar spam/kosong (keep_sentiment == False) tidak membentuk edge.
"""

from __future__ import annotations

from typing import Tuple

import networkx as nx
import pandas as pd


def build_edges(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (edges, selfloops). Kolom keduanya: source, target, weight, type."""
    d = df[df["keep_sentiment"]]
    rows = []

    rep = d[(d["type"] == "reply") & d["parent_author_id"].notna()]
    for r in rep.itertuples():
        rows.append((r.author_id, r.parent_author_id, "reply"))

    with_mention = d[d["mentions"] != ""]
    for r in with_mention.itertuples():
        parent = r.parent_author_id if r.type == "reply" else None
        for m in str(r.mentions).split(";"):
            if m and m != parent:
                rows.append((r.author_id, m, "mention"))

    cols = ["source", "target", "weight", "type"]
    if not rows:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    raw = pd.DataFrame(rows, columns=["source", "target", "type"])
    is_self = raw["source"] == raw["target"]

    def aggregate(x: pd.DataFrame) -> pd.DataFrame:
        out = x.groupby(["source", "target", "type"]).size().rename("weight").reset_index()
        return out[cols].sort_values(
            ["type", "weight", "source", "target"], ascending=[True, False, True, True]
        ).reset_index(drop=True)

    return aggregate(raw[~is_self]), aggregate(raw[is_self])


def build_graph(edges: pd.DataFrame, comments: pd.DataFrame) -> nx.DiGraph:
    """Graf berarah berbobot. Satu edge per pasangan; bobot per tipe disimpan sebagai atribut."""
    g = nx.DiGraph()
    for (s, t), grp in edges.groupby(["source", "target"]):
        w = grp.set_index("type")["weight"].to_dict()
        g.add_edge(
            s, t,
            weight=int(sum(w.values())),
            weight_reply=int(w.get("reply", 0)),
            weight_mention=int(w.get("mention", 0)),
        )

    per_author = comments.groupby("author_id").agg(
        n_comments=("comment_id", "size"),
        n_videos=("video_id", "nunique"),
    )
    for n in g.nodes:
        if n in per_author.index:
            g.nodes[n]["n_comments"] = int(per_author.at[n, "n_comments"])
            g.nodes[n]["n_videos"] = int(per_author.at[n, "n_videos"])
        else:  # akun hanya pernah muncul sebagai target (dibalas/di-mention)
            g.nodes[n]["n_comments"] = 0
            g.nodes[n]["n_videos"] = 0
    return g


def graph_summary(g: nx.DiGraph) -> pd.DataFrame:
    """Ringkasan singkat untuk pengecekan awal (statistik lengkap ada di Tahap 4)."""
    comps = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
    lcc = len(comps[0]) if comps else 0
    n = g.number_of_nodes()
    return pd.DataFrame([
        {"metric": "nodes", "value": n},
        {"metric": "edges", "value": g.number_of_edges()},
        {"metric": "density", "value": round(nx.density(g), 6)},
        {"metric": "weakly_connected_components", "value": len(comps)},
        {"metric": "lcc_nodes", "value": lcc},
        {"metric": "lcc_share_of_nodes", "value": round(lcc / n, 4) if n else 0},
        {"metric": "components_with_2_nodes", "value": sum(1 for c in comps if len(c) == 2)},
    ])
