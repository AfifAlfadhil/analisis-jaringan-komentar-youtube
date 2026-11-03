"""
sna.py
======
Tahap 4 — Analisis jaringan (SNA): statistik graf, centrality, dan distribusi degree.

Catatan metodologis:
- Statistik dihitung untuk graf penuh dan untuk LCC (largest weakly connected component).
- Jarak (path length, diameter), closeness, dan eigenvector hanya bermakna pada graf
  terhubung, sehingga dihitung pada LCC (graf tak berarah); akun di luar LCC bernilai NaN.
- Betweenness dan closeness memakai jarak berbasis jumlah hop (bobot edge adalah
  frekuensi interaksi, bukan jarak). Eigenvector memakai bobot sebagai kekuatan ikatan.
"""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import zeta

from src.network import build_edges

# Peran tiap metrik (untuk laporan)
METRIC_ROLES = {
    "in_degree": "Paling banyak dibalas/disebut oleh akun lain (pusat perhatian)",
    "out_degree": "Paling banyak membalas/menyebut akun lain (penyulut diskusi)",
    "degree": "Paling banyak terhubung (total interaksi unik)",
    "in_strength": "Total balasan/mention yang diterima (bobot)",
    "out_strength": "Total balasan/mention yang dikirim (bobot)",
    "betweenness": "Penghubung/jembatan antar-kelompok diskusi",
    "closeness": "Paling dekat dengan seluruh akun dalam LCC (jangkauan cepat)",
    "eigenvector": "Berpengaruh karena terhubung ke akun-akun penting (LCC)",
    "pagerank": "Pengaruh berdasarkan arah dan bobot interaksi masuk",
}


def lcc_nodes(g: nx.Graph) -> set:
    comps = nx.weakly_connected_components(g) if g.is_directed() else nx.connected_components(g)
    return max(comps, key=len)


def powerlaw_fit(degrees: np.ndarray, min_tail: int = 50) -> Dict[str, float]:
    """
    Fit power law diskret (MLE pendekatan Clauset-Shalizi-Newman) dengan memilih
    xmin yang meminimalkan statistik KS. Hasil ini indikasi ekor berat (heavy-tailed),
    bukan uji formal terhadap alternatif seperti log-normal.
    """
    d = np.asarray([x for x in degrees if x >= 1], dtype=float)
    best: Optional[Dict[str, float]] = None
    for xmin in range(1, 21):
        tail = d[d >= xmin]
        if len(tail) < min_tail:
            break
        alpha = 1.0 + len(tail) / np.sum(np.log(tail / (xmin - 0.5)))
        xs = np.sort(np.unique(tail))
        emp_cdf = np.array([(tail <= x).mean() for x in xs])
        fit_cdf = 1.0 - zeta(alpha, xs + 1) / zeta(alpha, xmin)
        ks = float(np.max(np.abs(emp_cdf - fit_cdf)))
        if best is None or ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xmin), "ks": ks, "n_tail": float(len(tail))}
    return best or {"alpha": np.nan, "xmin": np.nan, "ks": np.nan, "n_tail": 0.0}


def _scope_stats(g: nx.DiGraph, scope: str, with_paths: bool) -> list:
    u = g.to_undirected()
    n = g.number_of_nodes()
    deg = np.array([d for _, d in u.degree()])
    rows = {
        "nodes": n,
        "edges_directed": g.number_of_edges(),
        "edges_undirected": u.number_of_edges(),
        "density_directed": nx.density(g),
        "density_undirected": nx.density(u),
        "avg_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
        "max_degree": int(deg.max()),
        "share_degree_1": float((deg == 1).mean()),
        "avg_in_degree": g.number_of_edges() / n,
        "avg_clustering": nx.average_clustering(u),
        "transitivity": nx.transitivity(u),
        "reciprocity": nx.reciprocity(g),
        "degree_assortativity": nx.degree_assortativity_coefficient(u),
        "total_weight": int(sum(d["weight"] for _, _, d in g.edges(data=True))),
    }
    if with_paths:
        rows["avg_shortest_path"] = nx.average_shortest_path_length(u)
        rows["diameter"] = nx.diameter(u)
    return [{"scope": scope, "metric": k, "value": v} for k, v in rows.items()]


def network_stats(g: nx.DiGraph) -> pd.DataFrame:
    """Statistik graf penuh dan LCC, plus rincian komponen dan power-law degree."""
    out = _scope_stats(g, "full", with_paths=False)

    u = g.to_undirected()
    comps = sorted(nx.connected_components(u), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    n = g.number_of_nodes()
    lcc = comps[0]
    out += [
        {"scope": "full", "metric": "weakly_connected_components", "value": len(comps)},
        {"scope": "full", "metric": "strongly_connected_components", "value": nx.number_strongly_connected_components(g)},
        {"scope": "full", "metric": "lcc_nodes", "value": len(lcc)},
        {"scope": "full", "metric": "lcc_share_nodes", "value": len(lcc) / n},
        {"scope": "full", "metric": "nodes_outside_lcc", "value": n - len(lcc)},
        {"scope": "full", "metric": "components_size_ge_10", "value": sum(1 for s in sizes if s >= 10)},
        {"scope": "full", "metric": "nodes_in_components_size_ge_10", "value": sum(s for s in sizes if s >= 10)},
        {"scope": "full", "metric": "components_size_2", "value": sizes.count(2)},
    ]

    out += _scope_stats(g.subgraph(lcc).copy(), "lcc", with_paths=True)

    deg = np.array([d for _, d in u.degree()])
    pl = powerlaw_fit(deg)
    out += [{"scope": "full", "metric": f"powerlaw_{k}", "value": v} for k, v in pl.items()]
    return pd.DataFrame(out)


def degree_distribution(g: nx.DiGraph) -> pd.DataFrame:
    deg = pd.Series(dict(g.to_undirected().degree()))
    dist = deg.value_counts().sort_index().rename_axis("degree").reset_index(name="n_nodes")
    dist["share"] = dist["n_nodes"] / dist["n_nodes"].sum()
    return dist


def centrality_table(g: nx.DiGraph) -> pd.DataFrame:
    """Satu baris per akun di graf, berisi seluruh metrik centrality."""
    u = g.to_undirected()
    for a, b, d in u.edges(data=True):
        # bobot gabungan dua arah (bila A<->B saling balas)
        d["weight"] = (g[a][b]["weight"] if g.has_edge(a, b) else 0) + (g[b][a]["weight"] if g.has_edge(b, a) else 0)

    df = pd.DataFrame(index=pd.Index(list(g.nodes), name="author_id"))
    df["in_degree"] = pd.Series(dict(g.in_degree()))
    df["out_degree"] = pd.Series(dict(g.out_degree()))
    df["degree"] = pd.Series(dict(u.degree()))
    df["in_strength"] = pd.Series(dict(g.in_degree(weight="weight")))
    df["out_strength"] = pd.Series(dict(g.out_degree(weight="weight")))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(u))
    df["pagerank"] = pd.Series(nx.pagerank(g, alpha=0.85, weight="weight"))

    lcc = lcc_nodes(g)
    sub = u.subgraph(lcc)
    df["in_lcc"] = df.index.isin(lcc)
    df["closeness"] = pd.Series(nx.closeness_centrality(sub))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(sub, weight="weight"))

    df["n_comments"] = pd.Series({n: g.nodes[n].get("n_comments", 0) for n in g.nodes})
    df["n_videos"] = pd.Series({n: g.nodes[n].get("n_videos", 0) for n in g.nodes})
    return df.reset_index().sort_values("degree", ascending=False).reset_index(drop=True)


def edges_by_video(comments: pd.DataFrame) -> pd.DataFrame:
    """Edge yang dibangun terpisah per video (kolom: source, target, weight, type, video_id)."""
    parts = []
    for vid, d in comments.groupby("video_id"):
        e, _ = build_edges(d)
        e["video_id"] = vid
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


def _videos_per_account(ev: pd.DataFrame) -> pd.Series:
    long = ev.melt(id_vars="video_id", value_vars=["source", "target"], value_name="author_id")
    return long.groupby("author_id")["video_id"].agg(lambda s: sorted(set(s)))


def bridging_accounts(ev: pd.DataFrame, cent: pd.DataFrame) -> pd.DataFrame:
    """
    Akun penjembatan: akun yang berinteraksi (reply/mention) di lebih dari satu video.
    Edge tidak pernah melintasi video, jadi akun inilah satu-satunya yang menyambung
    jaringan antar-video.
    """
    vids = _videos_per_account(ev)
    br = vids[vids.map(len) > 1]
    out = pd.DataFrame({
        "author_id": br.index,
        "n_videos_interaction": br.map(len).values,
        "video_ids": br.map(";".join).values,
    })
    keep = ["author_id", "degree", "in_degree", "out_degree", "betweenness", "n_comments",
            "n_videos", "is_channel_owner"]
    out = out.merge(cent[keep], on="author_id", how="left")
    out = out.rename(columns={"n_videos": "n_videos_commented"})
    return out.sort_values(["n_videos_interaction", "betweenness"], ascending=False).reset_index(drop=True)


def bridging_stats(ev: pd.DataFrame, g: nx.DiGraph, br: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan peran akun penjembatan terhadap konektivitas graf."""
    u = g.to_undirected()
    n = u.number_of_nodes()
    vids = _videos_per_account(ev)
    lcc = lcc_nodes(g)
    lcc_videos = set().union(*[set(vids[x]) for x in lcc if x in vids.index])

    u2 = u.copy()
    u2.remove_nodes_from(set(br["author_id"]))
    after = max(nx.connected_components(u2), key=len) if u2.number_of_nodes() else set()

    rows = {
        "bridging_accounts": len(br),
        "bridging_share_nodes": len(br) / n,
        "bridging_in_lcc": int(br["author_id"].isin(lcc).sum()),
        "bridging_channel_owners": int(br["is_channel_owner"].sum()),
        "videos_with_interaction": ev["video_id"].nunique(),
        "videos_touched_by_lcc": len(lcc_videos),
        "lcc_nodes": len(lcc),
        "lcc_nodes_without_bridging": len(after),
    }
    return pd.DataFrame([{"scope": "bridging", "metric": k, "value": v} for k, v in rows.items()])


def top_actors(c: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N akun per metrik, dengan keterangan peran metrik."""
    rows = []
    for metric, role in METRIC_ROLES.items():
        top = c.nlargest(n, metric)
        for rank, r in enumerate(top.itertuples(index=False), 1):
            rec = r._asdict()
            rows.append({
                "metric": metric, "rank": rank, "author_id": rec["author_id"],
                "value": rec[metric], "is_channel_owner": rec.get("is_channel_owner", False),
                "role": role,
            })
    return pd.DataFrame(rows)
