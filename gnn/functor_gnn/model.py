"""
FunctorGNN — minimal message-passing GNN for graph-level binary classification.

Architecture:
    K × MPLayer  →  mean-pool readout  →  Linear classifier

Each MPLayer implements:
    h_v^(k) = ReLU( W · Σ_{u ∈ N(v)∪{v}} h_u^(k-1) / |N(v)∪{v}| )

When optional chemistry edge features are provided, the same aggregation is
kept but neighbour messages are deterministically scaled by a fixed bond-type
strength (single=1.0, double=2.0, triple=3.0, aromatic=1.5). This adds a
minimal edge-aware path without attention, kernels, or additional parameters.

This file implements only the core graph network used in the demo. Any
functor-style or canonicalization guarantees are external to this module and
come from the surrounding pipeline (for example canonicalize.py and the
commit-indexed inference path), not from this model in isolation.

The aggregation is permutation-equivariant (Proposition 4 in GNNEquiv),
and mean-pool readout is permutation-invariant (Proposition 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPLayer(nn.Module):
    """Single message-passing layer using optionally edge-aware mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(
        self,
        h: torch.Tensor,
        adj_norm: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h        : (N, in_dim)  node embeddings
        adj_norm : (N, N)       row-normalised adjacency with self-loops
        edge_attr: optional (N, N, 4) chemistry bond one-hot tensor ordered as
                   [single, double, triple, aromatic]

        Returns  : (N, out_dim)
        """
        if edge_attr is not None:
            bond_strengths = torch.tensor(
                [1.0, 2.0, 3.0, 1.5],
                dtype=h.dtype,
                device=h.device,
            )
            edge_scale = 1.0 + edge_attr.to(dtype=h.dtype, device=h.device) @ (
                bond_strengths - 1.0
            )
            adj_norm = adj_norm * edge_scale

        agg = adj_norm @ h                  # (N, in_dim) — neighbourhood mean
        return F.relu(self.linear(agg))     # (N, out_dim)


class FunctorGNN(nn.Module):
    """
    Minimal graph network used within the Functor GNN demo.

    This class is compatible with a functor-style pipeline in which graphs are
    canonicalized before inference, but the canonicalization and equivalence
    guarantees are external to this module. It can also consume optional
    chemistry bond edge features from the restricted demo subset.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 32,
        num_classes: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        # Store config for serialisation / reconstruction
        self.config = dict(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
        )

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            MPLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x        : (N, in_dim)  node features expected to have already been
                                ordered/preprocessed by the upstream pipeline
        adj_norm : (N, N)       row-normalised adjacency with self-loops
        edge_attr: optional (N, N, 4) bond features, already ordered by the
                                upstream pipeline

        Returns logits : (num_classes,)
        """
        # This module assumes any canonicalization or equivalence-preserving
        # preprocessing has already been performed upstream.
        h = x
        for layer in self.layers:
            h = layer(h, adj_norm, edge_attr)  # (N, hidden_dim)
        graph_emb = h.mean(dim=0)           # mean-pool → (hidden_dim,)
        return self.head(graph_emb)         # (num_classes,)
