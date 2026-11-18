"""Deteksi komunitas pada graf reply YouTube."""

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config
from src import community as cm


def main() -> None:
    graph = nx.read_graphml(config.GRAPH_DIR / "network.graphml")
    undirected = cm.to_undirected_weighted(graph)
    nodes = list(undirected.nodes)
    seeds = list(range(config.COMMUNITY_N_SEEDS))

    sweep = cm.sweep(
        undirected,
        config.COMMUNITY_RESOLUTIONS,
        seeds,
        config.COMMUNITY_MIN_SIZE,
    )
    sweep.to_csv(config.COMMUNITY_TABLES_DIR / "community_sweep.csv", index=False)

    runs = cm.run_many(
        undirected,
        config.COMMUNITY_ALGORITHM,
        config.COMMUNITY_RESOLUTION,
        seeds,
    )
    communities = cm.medoid_partition(undirected, runs)
    mapping = cm.assign_labels(communities, config.COMMUNITY_MIN_SIZE)

    assignment = pd.DataFrame({
        "author_id": nodes,
        "community_id": [mapping[node] for node in nodes],
    }).sort_values(["community_id", "author_id"]).reset_index(drop=True)

    raw_comments = pd.read_csv(
        config.RAW_COMMENTS_PATH,
        dtype=str,
        keep_default_na=False,
    )
    edges = cm.edges_by_video(raw_comments)
    stats = cm.community_stats(undirected, mapping, edges)
    primary = cm.primary_video(edges)
    association = cm.video_association(mapping, primary)

    assignment.to_csv(config.COMMUNITY_TABLES_DIR / "community_assignment.csv", index=False)
    stats.to_csv(config.COMMUNITY_TABLES_DIR / "community_stats.csv", index=False)

    summary_rows = [
        {"metric": "algorithm_final", "value": config.COMMUNITY_ALGORITHM},
        {"metric": "resolution_final", "value": config.COMMUNITY_RESOLUTION},
        {"metric": "n_seeds", "value": len(seeds)},
        {"metric": "modularity_final", "value": cm.modularity(undirected, communities)},
        {"metric": "n_communities_final", "value": len(communities)},
        {"metric": "n_communities_min_size", "value": sum(
            1 for community in communities
            if len(community) >= config.COMMUNITY_MIN_SIZE
        )},
        {"metric": "nodes_in_min_size_communities", "value": sum(
            len(community) for community in communities
            if len(community) >= config.COMMUNITY_MIN_SIZE
        )},
    ]
    for metric, value in association.items():
        summary_rows.append({"metric": metric, "value": value})

    louvain_runs = cm.run_many(
        undirected,
        "louvain",
        config.COMMUNITY_RESOLUTION,
        seeds,
    )
    louvain = cm.medoid_partition(undirected, louvain_runs)
    louvain_labels = cm.labels_of(louvain, nodes)
    leiden_labels = cm.labels_of(communities, nodes)
    summary_rows.append({
        "metric": "ari_louvain_vs_leiden",
        "value": adjusted_rand_score(louvain_labels, leiden_labels),
    })

    videos = pd.read_csv(config.RAW_VIDEOS_PATH, dtype=str)
    account_map = pd.read_csv(config.PROCESSED_DIR / "author_map.csv", dtype=str)
    owner_ids = set(
        account_map.loc[
            account_map["channel_id"].isin(set(videos["channel_id"])),
            "author_id",
        ]
    )
    without_owners = undirected.copy()
    without_owners.remove_nodes_from(owner_ids)
    without_owner_runs = cm.run_many(
        without_owners,
        config.COMMUNITY_ALGORITHM,
        config.COMMUNITY_RESOLUTION,
        seeds,
    )
    without_owner_partition = cm.medoid_partition(without_owners, without_owner_runs)
    shared_nodes = [
        node for node in without_owners.nodes
        if node in mapping
    ]
    without_owner_labels = cm.labels_of(without_owner_partition, shared_nodes)
    original_labels = cm.labels_of(communities, shared_nodes)
    summary_rows.extend([
        {"metric": "without_owners_n_removed", "value": len(owner_ids)},
        {"metric": "without_owners_modularity", "value": cm.modularity(
            without_owners,
            without_owner_partition,
        )},
        {"metric": "without_owners_ari", "value": adjusted_rand_score(
            original_labels,
            without_owner_labels,
        )},
    ])
    pd.DataFrame(summary_rows).to_csv(
        config.COMMUNITY_TABLES_DIR / "community_summary.csv",
        index=False,
    )

    print(f"Graf: {undirected.number_of_nodes()} akun, {undirected.number_of_edges()} hubungan")
    print(f"Modularity: {cm.modularity(undirected, communities):.4f}")
    print(stats.to_string(index=False))
    print(f"Tersimpan: {config.COMMUNITY_TABLES_DIR / 'community_assignment.csv'}")


if __name__ == "__main__":
    main()
