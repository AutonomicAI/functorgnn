"""Graph input preparation stage for BackGNN training.

This stage intentionally performs only structural validation and shallow copying.
The heavier tensor transformations are split into later stages so ``train.py``
reads as an explicit dataflow pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


_REQUIRED_KEYS = ("x", "adj", "label")


def prepare(graphs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy raw dataset graphs for the training pipeline.

    The returned dictionaries keep the original tensor objects.  Later pipeline
    stages may attach prepared fields without mutating the caller's graph list.
    """
    prepared: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        missing = [key for key in _REQUIRED_KEYS if key not in graph]
        if missing:
            raise KeyError(f"graph[{index}] missing required keys: {missing}")

        item = dict(graph)
        item.setdefault("graph_index", index)
        prepared.append(item)

    if not prepared:
        raise ValueError("Cannot train on an empty graph collection.")
    return prepared
