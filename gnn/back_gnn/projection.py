"""
Toy projection contexts for projection-aware inference.

A Projection is a small, deterministic lens over graph features. It is not a
chemistry assay model: the weights below are hand-written probes that let the
CLI demonstrate how the same (graph, commit, seed) can be interpreted under
multiple assay/reference-frame contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class Projection:
    """Minimal projection descriptor used by the demo inference path."""

    projection_id: str
    task: str
    assay_context: str
    reference_frame: str
    feature_weights: tuple[float, ...]
    probe_weights: tuple[float, ...] | None = None

    def weights_for_dim(self, dim: int) -> torch.Tensor:
        """
        Return feature weights compatible with a graph feature dimension.

        The registry includes chemistry-oriented 13D weights. For the original
        synthetic 4D cycle/tree dataset, the first four values form a stable
        structural lens. If a future toy graph has another dimension, weights
        are deterministically padded/truncated to preserve light dependencies.
        """
        if dim <= 0:
            raise ValueError(f"Feature dimension must be positive, got {dim}.")
        values = list(self.feature_weights)
        if len(values) < dim:
            values.extend([1.0] * (dim - len(values)))
        return torch.tensor(values[:dim], dtype=torch.float32)


# Chemistry feature layout from chemistry.py:
# [C,H,N,O,S,F,Cl,Br, charge(-1),charge(0),charge(+1), aromatic, degree]
# Synthetic feature layout from dataset.py:
# [degree, degree^2, inverse_degree, is_leaf]
PROJECTIONS: dict[str, Projection] = {
    "assay_polarity": Projection(
        projection_id="assay_polarity",
        task="toy hetero/polarity sensitivity",
        assay_context="emphasize hetero atoms and formal charge channels",
        reference_frame="restricted chemistry atom-feature frame",
        feature_weights=(0.8, 0.5, 1.8, 2.0, 1.6, 1.7, 1.5, 1.5, 1.4, 1.0, 1.4, 0.8, 1.0),
        probe_weights=(0.2, 0.1, 0.9, 1.0, 0.8, 0.7, 0.7, 0.7, 0.5, 0.0, 0.5, 0.1, 0.2),
    ),
    "assay_hydrogen_blind": Projection(
        projection_id="assay_hydrogen_blind",
        task="toy hydrogen-collapsed chemistry fixture",
        assay_context="collapse explicit hydrogen-count channel while retaining hetero/polar channels",
        reference_frame="restricted chemistry atom-feature frame with H channel quotient",
        feature_weights=(1.0, 0.0, 1.8, 2.0, 1.6, 1.7, 1.5, 1.5, 1.4, 1.0, 1.4, 0.8, 1.0),
        probe_weights=(0.2, 0.0, 0.9, 1.0, 0.8, 0.7, 0.7, 0.7, 0.5, 0.0, 0.5, 0.1, 0.2),
    ),
    "assay_aromaticity": Projection(
        projection_id="assay_aromaticity",
        task="toy aromatic motif sensitivity",
        assay_context="emphasize aromatic flag and carbon scaffold",
        reference_frame="restricted chemistry atom-feature frame",
        feature_weights=(1.4, 0.4, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.8, 1.0, 0.8, 2.5, 1.1),
        probe_weights=(0.6, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2),
    ),
    "assay_size": Projection(
        projection_id="assay_size",
        task="toy size/connectivity sensitivity",
        assay_context="emphasize explicit degree / structural size features",
        reference_frame="graph connectivity frame",
        feature_weights=(1.2, 1.5, 0.7, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 2.2),
        probe_weights=(0.2, 0.3, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0),
    ),
    "assay_hydrophobicity": Projection(
        projection_id="assay_hydrophobicity",
        task="toy hydrophobic scaffold sensitivity",
        assay_context="emphasize carbon/hydrogen channels; de-emphasize hetero atoms",
        reference_frame="restricted chemistry atom-feature frame",
        feature_weights=(1.8, 1.4, 0.5, 0.5, 0.6, 0.4, 0.4, 0.4, 0.8, 1.0, 0.8, 1.0, 1.0),
        probe_weights=(0.61, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.98, 0.2),
    ),
    "assay_charge": Projection(
        projection_id="assay_charge",
        task="toy formal-charge sensitivity",
        assay_context="emphasize formal charge features",
        reference_frame="restricted chemistry atom-feature frame",
        feature_weights=(0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.8, 0.9),
        probe_weights=(0.2, 0.1, 0.9, 1.0, 0.8, 0.7, 0.7, 0.7, 0.5, 0.0, 0.5, 0.1, 0.2),
    ),
    "assay_topology": Projection(
        projection_id="assay_topology",
        task="toy topology motif sensitivity",
        assay_context="emphasize structural degree and leaf-like channels",
        reference_frame="synthetic graph structural-feature frame",
        feature_weights=(1.0, 1.7, 0.8, 1.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 0.7, 1.8),
        probe_weights=(0.2, 0.3, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0),
    ),
}


def list_projection_ids() -> list[str]:
    return sorted(PROJECTIONS)


def get_projection(projection_id: str | None) -> Projection | None:
    """Return a projection by ID; None preserves legacy inference behavior."""
    if projection_id in (None, "", "default", "identity"):
        return None
    try:
        return PROJECTIONS[str(projection_id)]
    except KeyError as exc:
        available = ", ".join(list_projection_ids())
        raise ValueError(f"Unknown projection {projection_id!r}. Available: {available}") from exc


def apply_projection_features(x: torch.Tensor, projection: Projection | None) -> torch.Tensor:
    """Apply a deterministic feature-weight lens; identity when projection is None."""
    if projection is None:
        return x
    weights = projection.weights_for_dim(int(x.shape[1])).to(dtype=x.dtype, device=x.device)
    return x * weights


def projection_response(graph: dict, projection: Projection | None) -> float:
    """
    Compute a tiny deterministic probe response for demos and equivalence tests.

    The response is a weighted mean over node features plus a small graph-size
    term. This is motif/probe-style compression only; it is not a validated
    molecular endpoint.
    """
    x = graph["x"].float()
    if projection is None:
        weights = torch.ones(x.shape[1], dtype=x.dtype)
    else:
        source = projection.probe_weights or projection.feature_weights
        values = list(source)
        if len(values) < x.shape[1]:
            values.extend([1.0] * (x.shape[1] - len(values)))
        weights = torch.tensor(values[: x.shape[1]], dtype=x.dtype)
    mean_features = x.mean(dim=0)
    denom = weights.abs().sum().clamp(min=1.0)
    feature_score = float((mean_features * weights).sum().item() / float(denom.item()))
    size_score = 0.01 * float(graph.get("n", x.shape[0]))
    return round(feature_score + size_score, 8)


def response_vector(graphs: Iterable[dict], projection: Projection) -> list[float]:
    return [projection_response(g, projection) for g in graphs]
