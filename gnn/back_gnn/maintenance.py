"""Maintenance hooks for BackGNN training.

The hook is deliberately lightweight: it centralizes checkpoint metadata updates
without changing CommitStore semantics.
"""

from __future__ import annotations

from typing import Any


def update(
    *,
    epoch: int,
    metrics: dict[str, Any],
    task: str,
    problem_type: str,
    prepared_count: int,
    prepared_ids: list[str] | None = None,
    canonical_hashes: list[str] | None = None,
    projection_id: str = "default",
) -> dict[str, Any]:
    """Return commit metadata for the current training checkpoint."""
    metadata = {
        "epoch": epoch,
        "task": task,
        "problem_type": problem_type,
        "prepared_graphs": prepared_count,
        "projection_id": projection_id,
    }
    if prepared_ids is not None:
        metadata["prepared_artifact_ids"] = prepared_ids
    if canonical_hashes is not None:
        metadata["canonical_hashes"] = canonical_hashes

    metadata.update(metrics)
    return metadata
