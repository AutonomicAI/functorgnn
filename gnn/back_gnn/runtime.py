"""
Prepared steady-state runtime for FunctorGNN inference.

This module deliberately contains no audit orchestration.  Commit loading,
projection application, canonicalization, adjacency normalization, and batch
assembly happen before the measured steady-state path.  ``forward`` performs
only the neural computation on already-prepared tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .canonicalize import canonicalize_edge_attr, canonicalize_graph, normalize_adj
from .model import FunctorGNN
from .projection import Projection, apply_projection_features, get_projection


@dataclass(frozen=True)
class PreparedFunctorGraph:
    """Canonicalized tensors for one graph, reusable across forward calls."""

    x: torch.Tensor
    adj_norm: torch.Tensor
    edge_attr: torch.Tensor | None


@dataclass(frozen=True)
class PreparedFunctorBatch:
    """Dense block-diagonal batch preserving independent graph semantics."""

    batch_x: torch.Tensor
    batch_adj: torch.Tensor
    batch_edge_attr: torch.Tensor | None
    scatter: torch.Tensor
    num_graphs: int


PreparedInput = PreparedFunctorGraph | PreparedFunctorBatch


class PreparedFunctorRuntime:
    """
    Commit-fixed, preprocessing-complete runtime for repeated inference.

    Session setup:
      * construct the model once
      * load one commit once
      * put the model in evaluation mode

    Input setup:
      * apply an optional projection
      * canonicalize node and edge tensors
      * normalize adjacency
      * optionally assemble a reusable block-diagonal batch

    Hot path:
      * ``forward(prepared_input)`` only
    """

    def __init__(
        self,
        model: torch.nn.Module,
        commit_store,
        commit_id: str,
    ):
        self.model = model
        self.commit_id = commit_id
        commit_store.load_into(self.model, commit_id)
        self.model.eval()

    @classmethod
    def from_commit(cls, commit_store, commit_id: str) -> "PreparedFunctorRuntime":
        """Construct and initialize a FunctorGNN from a stored model config."""
        model = FunctorGNN(**commit_store.load_model_config())
        return cls(model, commit_store, commit_id)

    @staticmethod
    def prepare_graph(
        graph: dict,
        projection: str | Projection | None = None,
    ) -> PreparedFunctorGraph:
        """Prepare one graph once using the same transformations as run_inference."""
        proj = projection if isinstance(projection, Projection) else get_projection(projection)
        x = apply_projection_features(graph["x"], proj)
        x_c, adj_c, perm = canonicalize_graph(x, graph["adj"])
        edge_attr = graph.get("edge_attr")
        edge_attr_c = (
            canonicalize_edge_attr(edge_attr, perm)
            if edge_attr is not None else None
        )
        return PreparedFunctorGraph(
            x=x_c,
            adj_norm=normalize_adj(adj_c),
            edge_attr=edge_attr_c,
        )

    @classmethod
    def prepare_graphs(
        cls,
        graphs: list[dict],
        projection: str | Projection | None = None,
    ) -> list[PreparedFunctorGraph]:
        """Prepare a reusable collection of independent graphs."""
        return [cls.prepare_graph(graph, projection) for graph in graphs]

    @staticmethod
    def prepare_batch(
        graphs: list[PreparedFunctorGraph],
    ) -> PreparedFunctorBatch:
        """
        Assemble prepared graphs into the benchmark's exact block-diagonal form.

        This retains the existing dense benchmark mathematics.  It changes only
        when the tensors are constructed: once during setup rather than in the
        measured loop.
        """
        if not graphs:
            raise ValueError("Cannot prepare an empty graph batch.")

        xs = [graph.x for graph in graphs]
        adjs = [graph.adj_norm for graph in graphs]
        edge_attrs = [graph.edge_attr for graph in graphs]
        sizes = [x.shape[0] for x in xs]
        num_graphs = len(sizes)
        total_nodes = sum(sizes)

        batch_x = torch.cat(xs, dim=0)
        batch_adj = batch_x.new_zeros((total_nodes, total_nodes))
        offset = 0
        for adj, size in zip(adjs, sizes):
            batch_adj[offset:offset + size, offset:offset + size] = adj
            offset += size

        batch_edge_attr = None
        if all(edge_attr is not None for edge_attr in edge_attrs):
            edge_dim = edge_attrs[0].shape[-1]
            batch_edge_attr = batch_x.new_zeros(
                (total_nodes, total_nodes, edge_dim)
            )
            offset = 0
            for edge_attr, size in zip(edge_attrs, sizes):
                batch_edge_attr[
                    offset:offset + size,
                    offset:offset + size,
                ] = edge_attr
                offset += size

        scatter = batch_x.new_zeros((num_graphs, total_nodes))
        offset = 0
        for index, size in enumerate(sizes):
            scatter[index, offset:offset + size] = 1.0 / size
            offset += size

        return PreparedFunctorBatch(
            batch_x=batch_x,
            batch_adj=batch_adj,
            batch_edge_attr=batch_edge_attr,
            scatter=scatter,
            num_graphs=num_graphs,
        )

    def forward(self, prepared: PreparedInput) -> torch.Tensor:
        """Execute only neural inference for an already-prepared input."""
        with torch.no_grad():
            if isinstance(prepared, PreparedFunctorGraph):
                return self.model(
                    prepared.x,
                    prepared.adj_norm,
                    prepared.edge_attr,
                )

            h = prepared.batch_x
            for layer in self.model.layers:
                h = layer(h, prepared.batch_adj, prepared.batch_edge_attr)
            graph_embeddings = prepared.scatter @ h
            return self.model.head(graph_embeddings)

