"""
Commit-indexed, prepared-artifact inference engine.

Backloaded invariant: deterministic graph work is completed by training /
materialization and persisted in PreparedStore.  Inference loads the prepared
artifact, loads commit weights, executes the model forward pass, and formats /
logs the result.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

from .logger import log_inference_event
from .prepared_store import PreparedStore
from .projection import Projection, get_projection


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
    prepared_dir: str = "prepared",
) -> dict:
    """
    Run commit-indexed inference using a prepared graph artifact.

    ``graph`` is accepted for CLI/API compatibility but is not canonicalized or
    normalized here.  The matching artifact must already exist in PreparedStore.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    commit_store.load_into(model, commit_id)
    model.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    proj = projection if isinstance(projection, Projection) else get_projection(projection)
    projection_id = proj.projection_id if proj is not None else "default"
    resolved_graph_id = str(graph_id or graph.get("graph_id", "unknown"))

    try:
        prepared = PreparedStore(prepared_dir).load(
            graph_id=resolved_graph_id,
            projection_id=projection_id,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No prepared graph artifact found. Run train or materialize first."
        ) from exc

    task = (prepared.metadata or {}).get("task", graph.get("task", "cycle_detection"))
    is_regression = task == "esol"

    with torch.no_grad():
        logits = model(prepared.x_c, prepared.adj_norm, prepared.edge_attr_c)

    metadata = prepared.metadata or {}
    result = {
        "run_id":          run_id,
        "commit_id":       commit_id,
        "seed":            seed,
        "graph_id":        resolved_graph_id,
        "task":            task,
        "projection_id":   prepared.projection_id,
        "projection_description": metadata.get(
            "projection_description",
            getattr(proj, "description", None) if proj else None,
        ),
        "assay_context": metadata.get(
            "assay_context",
            getattr(proj, "assay_context", None) if proj else None,
        ),
        "projection_feature_weights_hash": metadata.get(
            "projection_feature_weights_hash",
            _projection_feature_weights_hash(proj),
        ),
        "projection_response": metadata.get("projection_response"),
        "prepared_id": prepared.prepared_id,
        "canonical_hash": prepared.canonical_hash,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }
    if is_regression:
        result["prediction"] = round(float(logits.view(-1)[0].item()), 8)
    else:
        probs = torch.softmax(logits, dim=-1)
        result.update({
            "predicted_class": int(probs.argmax().item()),
            "probs":           [round(float(p), 8) for p in probs.tolist()],
            "logits":          [round(float(v), 8) for v in logits.tolist()],
        })

    log_inference_event(result)
    return result
