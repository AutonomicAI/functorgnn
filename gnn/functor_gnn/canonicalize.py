"""
Graph canonicalization: C(X)

Implements Definition 2 & 3 from FunctorModelBrief:
    X ~ Y  ⟺  S(X) ≅ S(Y)          (structural equivalence)
    C(X) ∈ [X]_~                    (canonical representative)

We reorder nodes into a deterministic canonical form so that the Functor
GNN satisfies F(X) = F(C(X)) — Theorem 1 from the brief.

Sort key (primary→secondary→tertiary):
    1. Degree           (descending)   — structural signature
    2. Feature L2-norm  (descending)   — breaks degree ties
    3. Original index   (ascending)    — final tiebreaker for full determinism

Because node features are derived from graph structure (see dataset.py),
structurally isomorphic graphs produce identical canonical feature matrices,
confirming C(X) is a valid canonical representative.
"""

import torch


def canonicalize_graph(
    x: torch.Tensor,
    adj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (x_canonical, adj_canonical, permutation).

    x   : (N, F)  node features
    adj : (N, N)  binary undirected adjacency matrix

    permutation[i] = j means new position i was originally node j.
    """
    degrees = adj.sum(dim=1)        # (N,)
    feat_norms = x.norm(dim=1)      # (N,)
    N = x.shape[0]

    keys = [
        (-degrees[i].item(), -feat_norms[i].item(), i)
        for i in range(N)
    ]
    keys.sort()
    perm = torch.tensor([k[2] for k in keys], dtype=torch.long)

    x_c = x[perm]
    adj_c = adj[perm][:, perm]
    return x_c, adj_c, perm


def canonicalize_edge_attr(
    edge_attr: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    Apply an existing node permutation to edge features.

    edge_attr : (N, N, E) directed/undirected edge feature tensor
    perm      : permutation returned by canonicalize_graph
    """
    return edge_attr[perm][:, perm]


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Add self-loops and row-normalise: D^{-1}(A + I).

    This is the normalisation used in Canonical Message Passing Semantics
    (Definition 2 / GNNEquiv): AGG is a mean over N(v) ∪ {v}.
    """
    N = adj.shape[0]
    adj_sl = adj + torch.eye(N)
    deg = adj_sl.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj_sl / deg
