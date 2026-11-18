"""Membuat output centrality dari graf interaksi."""

import networkx as nx
import pandas as pd

import config
from src.sna import centrality_table, top_actors


def main() -> None:
    graph = nx.read_graphml(config.GRAPH_DIR / "network.graphml")
    videos = pd.read_csv(config.RAW_VIDEOS_PATH, dtype=str)
    account_map = pd.read_csv(
        config.PROCESSED_DIR / "author_map.csv",
        dtype=str,
    )

    owner_ids = set(
        account_map.loc[
            account_map["channel_id"].isin(set(videos["channel_id"])),
            "author_id",
        ]
    )
    centrality = centrality_table(graph, owner_ids)
    actors = top_actors(centrality)

    centrality.to_csv(
        config.TABLES_DIR / "centrality.csv",
        index=False,
    )
    actors.to_csv(
        config.TABLES_DIR / "top_actors.csv",
        index=False,
    )

    print(f"Centrality tersimpan untuk {len(centrality)} akun")


if __name__ == "__main__":
    main()
