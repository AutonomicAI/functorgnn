"""
Commit-indexed, projection-aware inference engine.

Implements the demo's commit-indexed replay path:

    F̃(c, G, ω, p) = S(π(c), C(P_p(G)), ω)

Given a commit_id c, a graph G, a random seed ω, and an optional projection p,
inference is:
    1. Load weights via  π(c)      — CommitStore.load_into
    2. Fix randomness via seed ω   — torch.manual_seed
    3. Apply projection lens P_p(G) — deterministic feature weighting
    4. Canonicalize graph  C(G)    — canonicalize_graph
    5. Normalise adjacency
    6. Forward pass  S(π(c), C(P_p(G)), ω)

The output is fully determined by (c, G, ω, p).  Calling this function
twice with identical arguments will always return identical results,
regardless of any intervening model mutations — this is referential
transparency of commit-indexed inference.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

from .canonicalize import canonicalize_edge_attr, canonicalize_graph, normalize_adj
from .logger import log_inference_event
from .projection import (
    Projection,
    apply_projection_features,
    get_projection,
    projection_response,
)


def _projection_feature_weights_hash(projection: Projection | None) -> str | None:
    """Stable SHA-256 checksum for registered projection feature weights."""
    if projection is None:
        return None
    payload = {
        "projection_id": projection.projection_id,
        "feature_weights": list(projection.feature_weights),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_inference(
    model: torch.nn.Module,
    commit_store,
    commit_id: str,
    graph: dict,
    seed: int,
    graph_id: str = "unknown",
    run_id: str | None = None,
    projection: str | Projection | None = None,
) -> dict:
    """
    Run commit-indexed inference: F̃(c, G, ω, p).

    Parameters
    ----------
    model        : FunctorGNN instance (weights will be replaced by π(commit_id))
    commit_store : CommitStore providing the projection π
    commit_id    : identifies the model snapshot to use
    graph        : dict with keys "x" (N×F tensor) and "adj" (N×N tensor);
                   may also include "edge_attr" (N×N×E bond features)
    seed         : integer random seed (ω)
    graph_id     : human-readable graph identifier for logging
    run_id       : optional run identifier; generated if None
    projection   : optional Projection or projection_id. None preserves the
                   original inference path.

    Returns
    -------
    dict with classification keys predicted_class/probs/logits, or regression
    keys prediction/task for ESOL.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    # Step 1 — π: project commit → weights
    commit_store.load_into(model, commit_id)
    model.eval()

    # Step 2 — fix randomness (ω)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(projection, Projection):
        proj = projection
    else:
        proj = get_projection(projection)
    projection_id = proj.projection_id if proj is not None else "default"

    # Step 3 — projection-aware feature lens: P_p(G)
    x, adj = graph["x"], graph["adj"]
    x = apply_projection_features(x, proj)

    # Step 4 — canonicalize: C(G)
    edge_attr = graph.get("edge_attr")
    x_c, adj_c, perm = canonicalize_graph(x, adj)
    edge_attr_c = (
        canonicalize_edge_attr(edge_attr, perm)
        if edge_attr is not None else None
    )

    # Step 5 — normalise adjacency
    adj_norm = normalize_adj(adj_c)

    # Step 6 — forward pass: S(π(c), C(P_p(G)), ω)
    task = graph.get("task", "cycle_detection")
    is_regression = task == "esol"

    with torch.no_grad():
        logits = model(x_c, adj_norm, edge_attr_c)

    result = {
        "run_id":          run_id,
        "commit_id":       commit_id,
        "seed":            seed,
        "graph_id":        graph_id,
        "task":            task,
        "projection_id":   projection_id,
        "projection_description": (proj.assay_context if proj is not None else None),
        "assay_context":   (proj.assay_context if proj is not None else None),
        "projection_feature_weights_hash": _projection_feature_weights_hash(proj),
        "projection_response": projection_response(graph, proj),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }
    if is_regression:
        result["prediction"] = round(float(logits.view(-1)[0].item()), 8)
    else:
        probs  = torch.softmax(logits, dim=-1)
        result.update({
            "predicted_class": int(probs.argmax().item()),
            "probs":           [round(float(p), 8) for p in probs.tolist()],
            "logits":          [round(float(v), 8) for v in logits.tolist()],
        })

    log_inference_event(result)
    return result
