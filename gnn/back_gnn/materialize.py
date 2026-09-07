"""Backloaded graph materialization for BackGNN.

This module owns deterministic graph preparation: projection feature application,
canonicalization, edge canonicalization, adjacency normalization, and stable
artifact hashing.  Inference consumes these prepared artifacts directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .canonicalize import canonicalize_edge_attr, canonicalize_graph, normalize_adj
from .projection import (
    Projection,
    apply_projection_features,
    get_projection,
    projection_response,
)


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "values": t.tolist(),
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _projection_feature_weights_hash(projection: Projection | None) -> str | None:
    if projection is None:
        return None
    return _stable_hash({
        "projection_id": projection.projection_id,
        "feature_weights": list(projection.feature_weights),
    })


@dataclass(frozen=True)
class PreparedGraph:
    """A graph artifact ready for model forward without preprocessing."""

    graph_id: str
    projection_id: str
    canonical_hash: str
    prepared_id: str
    x_c: torch.Tensor
    adj_norm: torch.Tensor
    edge_attr_c: torch.Tensor | None
    label: Any = None
    metadata: dict[str, Any] | None = None

    @property
    def x(self) -> torch.Tensor:
        """Backward-compatible alias used by the training loop."""
        return self.x_c

    @property
    def edge_attr(self) -> torch.Tensor | None:
        """Backward-compatible alias used by the training loop."""
        return self.edge_attr_c

    @property
    def y(self) -> Any:
        """Alias for callers that use y instead of label."""
        return self.label


def materialize_graph(
    graph: dict[str, Any],
    projection: str | Projection | None = None,
) -> PreparedGraph:
    """Build one deterministic prepared graph artifact from a raw graph."""
    proj = projection if isinstance(projection, Projection) else get_projection(projection)
    projection_id = proj.projection_id if proj is not None else "default"
    graph_id = str(graph.get("graph_id", graph.get("graph_index", "unknown")))

    x_projected = apply_projection_features(graph["x"], proj)
    x_c, adj_c, perm = canonicalize_graph(x_projected, graph["adj"])
    edge_attr = graph.get("edge_attr")
    edge_attr_c = canonicalize_edge_attr(edge_attr, perm) if edge_attr is not None else None
    adj_norm = normalize_adj(adj_c)

    canonical_payload: dict[str, Any] = {
        "graph_id": graph_id,
        "projection_id": projection_id,
        "x_c": _tensor_payload(x_c),
        "adj_norm": _tensor_payload(adj_norm),
    }
    if edge_attr_c is not None:
        canonical_payload["edge_attr_c"] = _tensor_payload(edge_attr_c)
    canonical_hash = _stable_hash(canonical_payload)
    prepared_id = deterministic_prepared_id(graph_id, projection_id, canonical_hash)

    metadata = {
        "task": graph.get("task", "cycle_detection"),
        "type": graph.get("type"),
        "n": graph.get("n", int(x_c.shape[0])),
        "graph_index": graph.get("graph_index"),
        "projection_description": getattr(proj, "description", None) if proj else None,
        "assay_context": getattr(proj, "assay_context", None) if proj else None,
        "projection_feature_weights_hash": _projection_feature_weights_hash(proj),
        "projection_response": projection_response(graph, proj),
    }

    return PreparedGraph(
        graph_id=graph_id,
        projection_id=projection_id,
        canonical_hash=canonical_hash,
        prepared_id=prepared_id,
        x_c=x_c,
        adj_norm=adj_norm,
        edge_attr_c=edge_attr_c,
        label=graph.get("label", graph.get("y")),
        metadata=metadata,
    )


def deterministic_prepared_id(graph_id: str, projection_id: str, canonical_hash: str) -> str:
    """Return the stable identifier used by prepared_store."""
    safe = f"{graph_id}:{projection_id}:{canonical_hash}"
    return hashlib.sha256(safe.encode("utf-8")).hexdigest()[:16]


def prepare(
    graphs: Iterable[dict[str, Any]],
    projection: str | Projection | None = None,
) -> list[PreparedGraph]:
    """Materialize all graphs for one projection."""
    return [materialize_graph(graph, projection) for graph in graphs]
