"""Deteksi komunitas pada graf interaksi."""

from itertools import combinations
import random
from typing import Dict, List, Sequence, Set

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SMALL_LABEL = "kecil"


def to_undirected_weighted(g: nx.DiGraph) -> nx.Graph:
    u = nx.Graph()
    u.add_nodes_from(g.nodes)
    for source, target, data in g.edges(data=True):
        weight = data.get("weight", 1)
        if u.has_edge(source, target):
            u[source][target]["weight"] += weight
        else:
            u.add_edge(source, target, weight=weight)
    return u


def run_louvain(u: nx.Graph, resolution: float, seed: int) -> List[Set[str]]:
    communities = nx.community.louvain_communities(
        u, weight="weight", resolution=resolution, seed=seed
    )
    return [set(community) for community in communities]


def run_leiden(u: nx.Graph, resolution: float, seed: int) -> List[Set[str]]:
    random.seed(seed)
    nodes = list(u.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    graph = ig.Graph(
        n=len(nodes),
        edges=[(index[source], index[target]) for source, target in u.edges()],
        directed=False,
    )
    graph.es["weight"] = [data["weight"] for _, _, data in u.edges(data=True)]
    partition = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=-1,
    )
    groups: Dict[int, Set[str]] = {}
    for node, community in zip(nodes, partition.membership):
        groups.setdefault(community, set()).add(node)
    return list(groups.values())


ALGORITHMS = {"louvain": run_louvain, "leiden": run_leiden}


def modularity(u: nx.Graph, communities: Sequence[Set[str]]) -> float:
    return nx.community.modularity(u, communities, weight="weight")


def labels_of(communities: Sequence[Set[str]], nodes: Sequence[str]) -> np.ndarray:
    labels = {}
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    return np.array([labels[node] for node in nodes])


def run_many(
    u: nx.Graph,
    algorithm: str,
    resolution: float,
    seeds: Sequence[int],
) -> List[List[Set[str]]]:
    runner = ALGORITHMS[algorithm]
    return [runner(u, resolution, seed) for seed in seeds]


def summarize_runs(
    u: nx.Graph,
    runs: Sequence[Sequence[Set[str]]],
    min_size: int,
) -> Dict[str, float]:
    nodes = list(u.nodes)
    modularities = [modularity(u, run) for run in runs]
    label_runs = [labels_of(run, nodes) for run in runs]
    ari_values = [
        adjusted_rand_score(left, right)
        for left, right in combinations(label_runs, 2)
    ] or [1.0]

    community_counts = [len(run) for run in runs]
    min_size_counts = [
        sum(1 for community in run if len(community) >= min_size)
        for run in runs
    ]
    node_shares = [
        sum(len(community) for community in run if len(community) >= min_size) / len(nodes)
        for run in runs
    ]

    return {
        "n_runs": len(runs),
        "modularity_mean": float(np.mean(modularities)),
        "modularity_std": float(np.std(modularities)),
        "n_communities_mean": float(np.mean(community_counts)),
        "n_communities_min_size_mean": float(np.mean(min_size_counts)),
        "share_nodes_in_min_size_mean": float(np.mean(node_shares)),
        "ari_pairwise_mean": float(np.mean(ari_values)),
        "ari_pairwise_min": float(np.min(ari_values)),
    }


def sweep(
    u: nx.Graph,
    resolutions: Sequence[float],
    seeds: Sequence[int],
    min_size: int,
) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        for resolution in resolutions:
            runs = run_many(u, algorithm, resolution, seeds)
            rows.append({
                "algorithm": algorithm,
                "resolution": resolution,
                **summarize_runs(u, runs, min_size),
            })
    return pd.DataFrame(rows)


def medoid_partition(
    u: nx.Graph,
    runs: Sequence[Sequence[Set[str]]],
) -> List[Set[str]]:
    if len(runs) == 1:
        return list(runs[0])

    nodes = list(u.nodes)
    label_runs = [labels_of(run, nodes) for run in runs]
    scores = []
    for i in range(len(label_runs)):
        others = [
            adjusted_rand_score(label_runs[i], label_runs[j])
            for j in range(len(label_runs))
            if i != j
        ]
        scores.append(float(np.mean(others)))
    return list(runs[int(np.argmax(scores))])


def assign_labels(
    communities: Sequence[Set[str]],
    min_size: int,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    community_number = 0
    for community in sorted(communities, key=len, reverse=True):
        if len(community) >= min_size:
            community_number += 1
            label = f"C{community_number:02d}"
        else:
            label = SMALL_LABEL
        for node in community:
            mapping[node] = label
    return mapping


def edges_by_video(
    raw_comments: pd.DataFrame,
) -> pd.DataFrame:
    from src.network import prepare_accounts

    comments, _ = prepare_accounts(raw_comments)
    comment_author = comments.set_index("comment_id")["author_id"]
    comments["parent_author_id"] = comments["parent_id"].replace("", np.nan).map(comment_author)
    replies = comments[
        comments["parent_id"].notna()
        & comments["author_id"].notna()
        & comments["parent_author_id"].notna()
    ].copy()

    replies["source"] = replies["author_id"]
    replies["target"] = replies["parent_author_id"]
    replies = replies[replies["source"].ne(replies["target"])]
    return (
        replies.groupby(["video_id", "source", "target"])
        .size()
        .rename("weight")
        .reset_index()
    )


def primary_video(edges: pd.DataFrame) -> pd.Series:
    long = edges.melt(
        id_vars=["video_id", "weight"],
        value_vars=["source", "target"],
        value_name="author_id",
    )
    weights = long.groupby(["author_id", "video_id"])["weight"].sum().reset_index()
    weights = weights.sort_values(
        ["author_id", "weight", "video_id"],
        ascending=[True, False, True],
    )
    return weights.drop_duplicates("author_id").set_index("author_id")["video_id"]


def community_stats(
    u: nx.Graph,
    mapping: Dict[str, str],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    labels = sorted(set(mapping.values()), key=lambda value: (value == SMALL_LABEL, value))

    edge_data = edges.copy()
    edge_data["source_community"] = edge_data["source"].map(mapping)
    edge_data["target_community"] = edge_data["target"].map(mapping)

    for label in labels:
        members = [node for node, community in mapping.items() if community == label]
        subgraph = u.subgraph(members)
        degree = sorted(subgraph.degree(weight="weight"), key=lambda item: item[1], reverse=True)

        internal = edge_data[
            edge_data["source_community"].eq(label)
            & edge_data["target_community"].eq(label)
        ]
        video_weights = internal.groupby("video_id")["weight"].sum()
        dominant_video = video_weights.idxmax() if len(video_weights) else ""
        dominant_share = (
            float(video_weights.max() / video_weights.sum())
            if len(video_weights)
            else np.nan
        )

        rows.append({
            "community_id": label,
            "n_nodes": len(members),
            "n_edges_internal": subgraph.number_of_edges(),
            "density_internal": nx.density(subgraph) if len(members) > 1 else np.nan,
            "hub_author_id": degree[0][0] if degree else "",
            "hub_degree": degree[0][1] if degree else 0,
            "dominant_video_id": dominant_video,
            "dominant_video_share": dominant_share,
        })
    return pd.DataFrame(rows)


def video_association(
    mapping: Dict[str, str],
    primary_videos: pd.Series,
) -> Dict[str, float]:
    nodes = [node for node in mapping if node in primary_videos.index]
    communities = [mapping[node] for node in nodes]
    videos = [primary_videos[node] for node in nodes]
    major = [i for i, label in enumerate(communities) if label != SMALL_LABEL]

    return {
        "nmi_all_nodes": float(normalized_mutual_info_score(communities, videos)),
        "ari_all_nodes": float(adjusted_rand_score(communities, videos)),
        "nmi_nodes_min_size": float(normalized_mutual_info_score(
            [communities[i] for i in major],
            [videos[i] for i in major],
        )) if major else np.nan,
        "ari_nodes_min_size": float(adjusted_rand_score(
            [communities[i] for i in major],
            [videos[i] for i in major],
        )) if major else np.nan,
    }
