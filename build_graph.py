"""Membangun graf dari reply komentar mentah."""

import networkx as nx
import pandas as pd

import config
from src.network import build_edges, build_graph, prepare_accounts


def main():
    df = pd.read_csv(config.RAW_COMMENTS_PATH, dtype=str, keep_default_na=False)
    edges, selfloops, account_map = build_edges(df)
    g = build_graph(edges)
    prepared, _ = prepare_accounts(df)
    activity = prepared.groupby("author_id").agg(
        n_comments=("comment_id", "size"),
        n_videos=("video_id", "nunique"),
    )
    for node in g.nodes:
        if node in activity.index:
            g.nodes[node]["n_comments"] = int(activity.at[node, "n_comments"])
            g.nodes[node]["n_videos"] = int(activity.at[node, "n_videos"])
        else:
            g.nodes[node]["n_comments"] = 0
            g.nodes[node]["n_videos"] = 0

    edges.to_csv(config.PROCESSED_DIR / "edges.csv", index=False)
    selfloops.to_csv(config.PROCESSED_DIR / "edges_selfloop.csv", index=False)
    account_map.to_csv(config.PROCESSED_DIR / "author_map.csv", index=False)
    nx.write_graphml(g, config.GRAPH_DIR / "network.graphml")
    nx.write_gexf(g, config.GRAPH_DIR / "network.gexf")

    print(f"Graf: {g.number_of_nodes()} akun, {g.number_of_edges()} hubungan")
    print(f"Edge tersimpan: {config.PROCESSED_DIR / 'edges.csv'}")


if __name__ == "__main__":
    main()
