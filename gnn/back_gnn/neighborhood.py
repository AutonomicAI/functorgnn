"""Neighborhood construction stage for BackGNN training.

The GNN consumes each graph through its canonical neighborhood frame:
canonical node ordering, correspondingly permuted edge attributes, and normalized
adjacency.  Materializing this before the epoch loop keeps training focused on
optimization rather than graph bookkeeping.
"""

from __future__ import annotations

from typing import Any

from .canonicalize import canonicalize_edge_attr, canonicalize_graph, normalize_adj


def build(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach canonical neighborhood tensors to each prepared graph."""
    built: list[dict[str, Any]] = []
    for graph in prepared:
        x_c, adj_c, perm = canonicalize_graph(graph["x"], graph["adj"])
        edge_attr = graph.get("edge_attr")
        edge_attr_c = (
            canonicalize_edge_attr(edge_attr, perm)
            if edge_attr is not None else None
        )

        item = dict(graph)
        item["x_c"] = x_c
        item["adj_c"] = adj_c
        item["adj_norm"] = normalize_adj(adj_c)
        item["edge_attr_c"] = edge_attr_c
        item["canonical_perm"] = perm
        built.append(item)
    return built
