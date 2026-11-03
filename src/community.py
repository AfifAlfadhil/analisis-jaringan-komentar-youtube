"""
community.py
============
Tahap 5 — Community detection pada graf interaksi.

- Graf berarah diubah menjadi tak berarah berbobot (bobot dua arah dijumlahkan).
- Algoritma: Louvain (baseline, NetworkX) dan Leiden (pembanding, python-igraph).
- Kestabilan: dijalankan dengan banyak seed, diukur dengan modularity dan ARI antar-run.
- Komunitas lebih kecil dari ambang ukuran digabung menjadi satu label "small".
- Pemeriksaan keterkaitan komunitas dengan video (karena edge hanya terbentuk dalam video).
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Sequence, Set

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SMALL_LABEL = "small"


# --------------------------------------------------------------------------- #
# Graf dan algoritma
# --------------------------------------------------------------------------- #
def to_undirected_weighted(g: nx.DiGraph) -> nx.Graph:
    """Graf tak berarah; bobot A->B dan B->A dijumlahkan."""
    u = nx.Graph()
    u.add_nodes_from(g.nodes)
    for a, b, d in g.edges(data=True):
        w = d.get("weight", 1)
        if u.has_edge(a, b):
            u[a][b]["weight"] += w
        else:
            u.add_edge(a, b, weight=w)
    return u


def run_louvain(u: nx.Graph, resolution: float, seed: int) -> List[Set]:
    comms = nx.community.louvain_communities(u, weight="weight", resolution=resolution, seed=seed)
    return [set(c) for c in comms]


def run_leiden(u: nx.Graph, resolution: float, seed: int) -> List[Set]:
    nodes = list(u.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in u.edges()], directed=False)
    g.es["weight"] = [d["weight"] for _, _, d in u.edges(data=True)]
    random.seed(seed)  # igraph memakai generator acak Python
    part = g.community_leiden(
        objective_function="modularity", weights="weight",
        resolution=resolution, n_iterations=-1,
    )
    groups: Dict[int, Set] = {}
    for node, m in zip(nodes, part.membership):
        groups.setdefault(m, set()).add(node)
    return list(groups.values())


ALGORITHMS = {"louvain": run_louvain, "leiden": run_leiden}


def modularity(u: nx.Graph, comms: Sequence[Set]) -> float:
    """Modularity standar (resolution = 1) agar bisa dibandingkan antar-algoritma dan antar-resolution."""
    return nx.community.modularity(u, comms, weight="weight")


def labels_of(comms: Sequence[Set], nodes: Sequence) -> np.ndarray:
    lab = {}
    for i, c in enumerate(comms):
        for n in c:
            lab[n] = i
    return np.array([lab[n] for n in nodes])


# --------------------------------------------------------------------------- #
# Sweep resolution + kestabilan
# --------------------------------------------------------------------------- #
def run_many(u: nx.Graph, algorithm: str, resolution: float, seeds: Sequence[int]) -> List[List[Set]]:
    fn = ALGORITHMS[algorithm]
    return [fn(u, resolution, s) for s in seeds]


def summarize_runs(u: nx.Graph, runs: List[List[Set]], min_size: int, lcc: Set | None = None) -> Dict[str, float]:
    """
    Ringkasan banyak run. Modularity dan ARI pada graf penuh cenderung terlalu tinggi karena
    graf terdiri dari ratusan komponen kecil yang otomatis menjadi komunitas; bila `lcc`
    diberikan, metrik yang sama dihitung khusus di LCC (lebih informatif).
    """
    nodes = list(u.nodes)
    mods = [modularity(u, r) for r in runs]
    labs = [labels_of(r, nodes) for r in runs]
    aris = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)] or [1.0]
    n_all = [len(r) for r in runs]
    n_big = [sum(1 for c in r if len(c) >= min_size) for r in runs]
    share_big = [sum(len(c) for c in r if len(c) >= min_size) / len(nodes) for r in runs]
    out = {
        "n_runs": len(runs),
        "modularity_mean": float(np.mean(mods)),
        "modularity_std": float(np.std(mods)),
        "n_communities_mean": float(np.mean(n_all)),
        "n_communities_min_size_mean": float(np.mean(n_big)),
        "share_nodes_in_min_size_mean": float(np.mean(share_big)),
        "ari_pairwise_mean": float(np.mean(aris)),
        "ari_pairwise_min": float(np.min(aris)),
    }
    if lcc:
        sub = u.subgraph(lcc)
        lcc_nodes_list = list(sub.nodes)
        lcc_runs = [[c & lcc for c in r if c & lcc] for r in runs]
        lmods = [modularity(sub, r) for r in lcc_runs]
        llabs = [labels_of(r, lcc_nodes_list) for r in lcc_runs]
        laris = [adjusted_rand_score(a, b) for a, b in combinations(llabs, 2)] or [1.0]
        out.update({
            "lcc_modularity_mean": float(np.mean(lmods)),
            "lcc_modularity_std": float(np.std(lmods)),
            "lcc_n_communities_mean": float(np.mean([len(r) for r in lcc_runs])),
            "lcc_n_communities_min_size_mean": float(np.mean([sum(1 for c in r if len(c) >= min_size) for r in lcc_runs])),
            "lcc_ari_pairwise_mean": float(np.mean(laris)),
            "lcc_ari_pairwise_min": float(np.min(laris)),
        })
    return out


def sweep(u: nx.Graph, resolutions: Sequence[float], seeds: Sequence[int], min_size: int,
          lcc: Set | None = None) -> pd.DataFrame:
    rows = []
    for algo in ALGORITHMS:
        for res in resolutions:
            runs = run_many(u, algo, res, seeds)
            rows.append({"algorithm": algo, "resolution": res, **summarize_runs(u, runs, min_size, lcc)})
    return pd.DataFrame(rows)


def medoid_partition(u: nx.Graph, runs: List[List[Set]]) -> List[Set]:
    """Partisi yang paling mirip dengan semua run lain (rata-rata ARI tertinggi)."""
    nodes = list(u.nodes)
    labs = [labels_of(r, nodes) for r in runs]
    if len(runs) == 1:
        return runs[0]
    scores = [
        np.mean([adjusted_rand_score(labs[i], labs[j]) for j in range(len(labs)) if j != i])
        for i in range(len(labs))
    ]
    return runs[int(np.argmax(scores))]


# --------------------------------------------------------------------------- #
# Penetapan label komunitas + statistik
# --------------------------------------------------------------------------- #
def assign_labels(comms: Sequence[Set], min_size: int) -> Dict[str, str]:
    """Komunitas >= min_size diberi C01, C02, ... (urut ukuran); sisanya 'small'."""
    ordered = sorted(comms, key=len, reverse=True)
    mapping: Dict[str, str] = {}
    k = 0
    for c in ordered:
        if len(c) >= min_size:
            k += 1
            label = f"C{k:02d}"
        else:
            label = SMALL_LABEL
        for n in c:
            mapping[n] = label
    return mapping


def primary_video(ev: pd.DataFrame) -> pd.Series:
    """Video tempat tiap akun paling banyak berinteraksi (berdasarkan bobot edge)."""
    long = ev.melt(id_vars=["video_id", "weight"], value_vars=["source", "target"], value_name="author_id")
    w = long.groupby(["author_id", "video_id"])["weight"].sum().reset_index()
    w = w.sort_values(["author_id", "weight", "video_id"], ascending=[True, False, True])
    return w.drop_duplicates("author_id").set_index("author_id")["video_id"]


def community_stats(u: nx.Graph, mapping: Dict[str, str], ev: pd.DataFrame) -> pd.DataFrame:
    """Ukuran, density internal, hub utama, dan video dominan tiap komunitas."""
    rows = []
    labels = sorted(set(mapping.values()), key=lambda x: (x == SMALL_LABEL, x))
    ev = ev.copy()
    ev["c_s"] = ev["source"].map(mapping)
    ev["c_t"] = ev["target"].map(mapping)
    for lab in labels:
        members = [n for n, c in mapping.items() if c == lab]
        sub = u.subgraph(members)
        n = len(members)
        m_int = sub.number_of_edges()
        m_total = sum(dict(u.degree(members)).values()) - m_int  # edge unik yang menyentuh komunitas
        deg = sorted(sub.degree(), key=lambda x: -x[1])
        inside = ev[(ev["c_s"] == lab) & (ev["c_t"] == lab)]
        vshare = inside.groupby("video_id")["weight"].sum()
        rows.append({
            "community_id": lab,
            "n_nodes": n,
            "n_edges_internal": m_int,
            "share_edges_internal": m_int / m_total if m_total else np.nan,
            "density_internal": nx.density(sub) if n > 1 else np.nan,
            "avg_degree_internal": 2 * m_int / n,
            "hub_author_id": deg[0][0] if deg else "",
            "hub_degree_internal": deg[0][1] if deg else 0,
            "top3_hubs": ";".join(a for a, _ in deg[:3]),
            "n_videos": int(vshare.size),
            "dominant_video_id": vshare.idxmax() if len(vshare) else "",
            "dominant_video_share": float(vshare.max() / vshare.sum()) if len(vshare) else np.nan,
        })
    return pd.DataFrame(rows)


def video_association(mapping: Dict[str, str], pv: pd.Series) -> Dict[str, float]:
    """Seberapa jauh komunitas hanya mengulang pembagian per video (NMI/ARI terhadap video utama akun)."""
    nodes = [n for n in mapping if n in pv.index]
    c = [mapping[n] for n in nodes]
    v = [pv[n] for n in nodes]
    big = [i for i, x in enumerate(c) if x != SMALL_LABEL]
    return {
        "nmi_all_nodes": float(normalized_mutual_info_score(c, v)),
        "ari_all_nodes": float(adjusted_rand_score(c, v)),
        "nmi_nodes_in_min_size": float(normalized_mutual_info_score([c[i] for i in big], [v[i] for i in big])) if big else np.nan,
        "ari_nodes_in_min_size": float(adjusted_rand_score([c[i] for i in big], [v[i] for i in big])) if big else np.nan,
    }
