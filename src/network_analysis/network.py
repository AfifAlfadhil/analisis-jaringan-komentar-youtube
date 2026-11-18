"""Pembentukan graf interaksi dari data komentar mentah."""

from __future__ import annotations

from typing import Tuple

import networkx as nx
import pandas as pd


def prepare_accounts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Anonymize channel IDs and map comment_id to the account that wrote it."""
    df = df.copy()
    df["author_id"] = df["author_id"].fillna("").astype(str).str.strip()
    df["author_name"] = df["author_name"].fillna("").astype(str).str.strip()

    df["account_key"] = df["author_id"]
    no_channel = df["account_key"].eq("")
    df.loc[no_channel, "account_key"] = "name:" + df.loc[no_channel, "author_name"].str.lower()
    df.loc[df["account_key"].eq("name:"), "account_key"] = pd.NA

    mapping = {}
    for key in df["account_key"].dropna():
        if key not in mapping:
            mapping[key] = f"user_{len(mapping) + 1:04d}"

    df["author_id"] = df["account_key"].map(mapping)
    handles = df.dropna(subset=["account_key"]).groupby("account_key")["author_name"].last()
    account_map = pd.DataFrame({
        "author_id": list(mapping.values()),
        "channel_id": list(mapping.keys()),
        "handle": [handles.get(key, "") for key in mapping],
    })
    return df, account_map


def build_edges(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return reply edges and self-loops."""
    df, account_map = prepare_accounts(df)
    comment_author = df.set_index("comment_id")["author_id"]

    df["parent_author_id"] = df["parent_id"].map(comment_author)
    replies = df[
        df["parent_id"].notna()
        & df["author_id"].notna()
        & df["parent_author_id"].notna()
    ]

    if replies.empty:
        return pd.DataFrame(columns=["source", "target", "weight"]), pd.DataFrame(columns=["source", "target", "weight"]), account_map

    rows = replies.rename(columns={"author_id": "source"})[["source", "parent_author_id"]]
    rows = rows.rename(columns={"parent_author_id": "target"})
    rows = rows[rows["source"].ne(rows["target"])]
    edges = rows.groupby(["source", "target"]).size().rename("weight").reset_index()
    edges = edges.sort_values(["weight", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)

    raw_loops = replies.rename(columns={"author_id": "source"})[["source", "parent_author_id"]]
    raw_loops = raw_loops.rename(columns={"parent_author_id": "target"})
    loops = raw_loops[raw_loops["source"].eq(raw_loops["target"])]
    loops = loops.groupby(["source", "target"]).size().rename("weight").reset_index() if not loops.empty else pd.DataFrame(columns=["source", "target", "weight"])
    return edges, loops, account_map


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in edges.itertuples(index=False):
        g.add_edge(row.source, row.target, weight=int(row.weight))
    return g
