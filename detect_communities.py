"""
detect_communities.py
=====================
Tahap 5 — Community detection. Jalankan setelah analyze_network.py:

    python detect_communities.py

Parameter (algoritma final, resolution, jumlah seed, ukuran minimum) ada di config.py.

Input : outputs/graph/network.graphml
        outputs/tables/centrality.csv        (penanda pemilik channel)
        data/processed/comments_clean.csv    (edge per video)
Output: outputs/tables/community_sweep.csv        perbandingan algoritma x resolution (modularity, kestabilan)
        outputs/tables/community_assignment.csv   akun -> id komunitas
        outputs/tables/community_stats.csv        ukuran, density internal, hub utama, video dominan
        outputs/tables/community_summary.csv      perbandingan algoritma, kaitan dengan video, sensitivitas pemilik channel
"""

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config
from src import community as cm
from src.sna import edges_by_video, lcc_nodes


def main():
    g = nx.read_graphml(config.GRAPH_DIR / "network.graphml")
    u = cm.to_undirected_weighted(g)
    nodes = list(u.nodes)
    seeds = list(range(config.COMMUNITY_N_SEEDS))
    min_size = config.COMMUNITY_MIN_SIZE
    print(f"Graf: {u.number_of_nodes()} node, {u.number_of_edges()} edge (tak berarah, berbobot)")

    # 1) Sweep algoritma x resolution
    print("\n[1/4] Sweep resolution dan uji kestabilan seed...")
    lcc = lcc_nodes(g)
    sw = cm.sweep(u, config.COMMUNITY_RESOLUTIONS, seeds, min_size, lcc)
    sw.to_csv(config.TABLES_DIR / "community_sweep.csv", index=False)
    pd.options.display.float_format = "{:.3f}".format
    show = ["algorithm", "resolution", "modularity_mean", "ari_pairwise_mean", "lcc_modularity_mean",
            "lcc_modularity_std", "lcc_n_communities_mean", "lcc_n_communities_min_size_mean",
            "lcc_ari_pairwise_mean", "lcc_ari_pairwise_min"]
    print(sw[show].to_string(index=False))

    # 2) Partisi final
    algo, res = config.COMMUNITY_ALGORITHM, config.COMMUNITY_RESOLUTION
    print(f"\n[2/4] Partisi final: {algo}, resolution={res}, {len(seeds)} seed (medoid)")
    runs = {a: cm.run_many(u, a, res, seeds) for a in cm.ALGORITHMS}
    final = cm.medoid_partition(u, runs[algo])
    mapping = cm.assign_labels(final, min_size)

    # 3) Statistik komunitas, kaitan dengan video
    print("\n[3/4] Statistik komunitas dan kaitan dengan video...")
    comments = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv")
    comments["mentions"] = comments["mentions"].fillna("")
    ev = edges_by_video(comments)
    stats = cm.community_stats(u, mapping, ev)
    pv = cm.primary_video(ev)
    assoc = cm.video_association(mapping, pv)

    cent = pd.read_csv(config.TABLES_DIR / "centrality.csv")
    owner = cent.set_index("author_id")["is_channel_owner"]
    raw_size = {n: len(c) for c in final for n in c}
    assign = pd.DataFrame({
        "author_id": nodes,
        "community_id": [mapping[n] for n in nodes],
        "raw_community_size": [raw_size[n] for n in nodes],
        "in_lcc": [n in lcc for n in nodes],
        "is_channel_owner": [bool(owner.get(n, False)) for n in nodes],
        "primary_video_id": [pv.get(n, "") for n in nodes],
    }).sort_values(["community_id", "author_id"]).reset_index(drop=True)

    assign.to_csv(config.TABLES_DIR / "community_assignment.csv", index=False)
    stats.to_csv(config.TABLES_DIR / "community_stats.csv", index=False)

    # 4) Ringkasan: Louvain vs Leiden, sensitivitas pemilik channel
    print("\n[4/4] Perbandingan algoritma dan sensitivitas pemilik channel...")
    summary = []

    def add(k, v):
        summary.append({"metric": k, "value": v})

    add("algorithm_final", algo)
    add("resolution_final", res)
    add("n_seeds", len(seeds))
    add("modularity_final", cm.modularity(u, final))
    lcc_final = [c & lcc for c in final if c & lcc]
    add("modularity_final_lcc_only", cm.modularity(u.subgraph(lcc), lcc_final))
    add("n_communities_final_lcc_only", len(lcc_final))
    n_big = int((stats["community_id"] != cm.SMALL_LABEL).sum())
    add("n_communities_min_size", n_big)
    add("nodes_in_min_size_communities", int((assign["community_id"] != cm.SMALL_LABEL).sum()))
    add("nodes_in_small", int((assign["community_id"] == cm.SMALL_LABEL).sum()))
    add("min_size", min_size)

    lou = cm.medoid_partition(u, runs["louvain"])
    lei = cm.medoid_partition(u, runs["leiden"])
    add("modularity_louvain", cm.modularity(u, lou))
    add("modularity_leiden", cm.modularity(u, lei))
    add("ari_louvain_vs_leiden", adjusted_rand_score(cm.labels_of(lou, nodes), cm.labels_of(lei, nodes)))

    for k, v in assoc.items():
        add(f"video_{k}", v)
    big_stats = stats[stats["community_id"] != cm.SMALL_LABEL]
    if len(big_stats):
        wavg = (big_stats["dominant_video_share"] * big_stats["n_nodes"]).sum() / big_stats["n_nodes"].sum()
        add("dominant_video_share_weighted_mean", wavg)

    owners = set(assign.loc[assign["is_channel_owner"], "author_id"])
    u2 = u.copy()
    u2.remove_nodes_from(owners)
    runs2 = cm.run_many(u2, algo, res, seeds)
    final2 = cm.medoid_partition(u2, runs2)
    shared = [n for n in u2.nodes]
    add("without_owners_n_removed", len(owners))
    add("without_owners_modularity", cm.modularity(u2, final2))
    add("without_owners_n_communities_min_size", sum(1 for c in final2 if len(c) >= min_size))
    # ARI pada akun yang sama; bandingkan dengan partisi final (tanpa memotong pemilik channel)
    lab_full = pd.Series({n: i for i, c in enumerate(final) for n in c})
    lab_wo = pd.Series({n: i for i, c in enumerate(final2) for n in c})
    add("without_owners_ari_vs_final", adjusted_rand_score(lab_full[shared], lab_wo[shared]))

    summary = pd.DataFrame(summary)
    summary.to_csv(config.TABLES_DIR / "community_summary.csv", index=False)

    print("\nRingkasan:")
    print(summary.to_string(index=False))
    print("\nKomunitas (>= min_size):")
    cols = ["community_id", "n_nodes", "n_edges_internal", "density_internal", "share_edges_internal",
            "hub_degree_internal", "n_videos", "dominant_video_share"]
    print(stats[cols].to_string(index=False))


if __name__ == "__main__":
    main()
