"""Metrik centrality untuk graf interaksi akun."""

import networkx as nx
import pandas as pd


def lcc_nodes(g: nx.DiGraph) -> set:
    undirected = g.to_undirected()
    components = nx.connected_components(undirected)
    return max(components, key=len)


def centrality_table(
    g: nx.DiGraph,
    owner_ids: set,
) -> pd.DataFrame:
    undirected = g.to_undirected()
    for source, target, data in undirected.edges(data=True):
        incoming = g[source][target]["weight"] if g.has_edge(source, target) else 0
        outgoing = g[target][source]["weight"] if g.has_edge(target, source) else 0
        data["weight"] = incoming + outgoing

    rows = pd.DataFrame(index=pd.Index(list(g.nodes), name="author_id"))
    rows["in_degree"] = pd.Series(dict(g.in_degree()))
    rows["out_degree"] = pd.Series(dict(g.out_degree()))
    rows["degree"] = pd.Series(dict(undirected.degree()))
    rows["in_strength"] = pd.Series(dict(g.in_degree(weight="weight")))
    rows["out_strength"] = pd.Series(dict(g.out_degree(weight="weight")))
    rows["betweenness"] = pd.Series(nx.betweenness_centrality(undirected))
    rows["pagerank"] = pd.Series(nx.pagerank(g, alpha=0.85, weight="weight"))

    lcc = lcc_nodes(g)
    subgraph = undirected.subgraph(lcc)
    rows["in_lcc"] = rows.index.isin(lcc)
    rows["closeness"] = pd.Series(nx.closeness_centrality(subgraph))
    rows["eigenvector"] = pd.Series(
        nx.eigenvector_centrality_numpy(subgraph, weight="weight")
    )

    rows["n_comments"] = pd.Series({
        node: g.nodes[node].get("n_comments", 0)
        for node in g.nodes
    })
    rows["n_videos"] = pd.Series({
        node: g.nodes[node].get("n_videos", 0)
        for node in g.nodes
    })
    rows["is_channel_owner"] = rows.index.isin(owner_ids)

    return rows.reset_index().sort_values("degree", ascending=False).reset_index(drop=True)


def top_actors(centrality: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    metrics = [
        "in_degree",
        "out_degree",
        "degree",
        "in_strength",
        "out_strength",
        "betweenness",
        "pagerank",
        "closeness",
        "eigenvector",
    ]
    rows = []
    for metric in metrics:
        top = centrality.nlargest(limit, metric)
        for rank, row in enumerate(top.itertuples(index=False), 1):
            rows.append({
                "metric": metric,
                "rank": rank,
                "author_id": row.author_id,
                "value": getattr(row, metric),
                "is_channel_owner": row.is_channel_owner,
            })
    return pd.DataFrame(rows)
