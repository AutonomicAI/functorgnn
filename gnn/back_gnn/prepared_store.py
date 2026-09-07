"""Filesystem store for backloaded prepared graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .materialize import PreparedGraph, deterministic_prepared_id


class PreparedStore:
    """Small local prepared-artifact store using torch PT plus JSON index."""

    _INDEX = "index.json"

    def __init__(self, store_dir: str | Path = "prepared"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.store_dir / self._INDEX
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"prepared": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)

    @staticmethod
    def deterministic_id(graph_id: str, projection_id: str, canonical_hash: str) -> str:
        return deterministic_prepared_id(graph_id, projection_id, canonical_hash)

    def _path(self, prepared_id: str) -> Path:
        return self.store_dir / f"{prepared_id}.pt"

    def save(self, prepared_graph: PreparedGraph) -> str:
        """Persist a prepared graph and return its deterministic prepared_id."""
        prepared_id = prepared_graph.prepared_id
        torch.save({
            "graph_id": prepared_graph.graph_id,
            "projection_id": prepared_graph.projection_id,
            "canonical_hash": prepared_graph.canonical_hash,
            "prepared_id": prepared_id,
            "x_c": prepared_graph.x_c.detach().cpu(),
            "adj_norm": prepared_graph.adj_norm.detach().cpu(),
            "edge_attr_c": None if prepared_graph.edge_attr_c is None else prepared_graph.edge_attr_c.detach().cpu(),
            "label": prepared_graph.label,
            "metadata": prepared_graph.metadata or {},
        }, self._path(prepared_id))
        self._index["prepared"][prepared_id] = {
            "prepared_id": prepared_id,
            "graph_id": prepared_graph.graph_id,
            "projection_id": prepared_graph.projection_id,
            "canonical_hash": prepared_graph.canonical_hash,
        }
        self._save_index()
        return prepared_id

    def exists(
        self,
        prepared_id: str | None = None,
        *,
        graph_id: str | None = None,
        projection_id: str = "default",
        canonical_hash: str | None = None,
    ) -> bool:
        try:
            pid = prepared_id or self._resolve_id(graph_id, projection_id, canonical_hash)
        except FileNotFoundError:
            return False
        return self._path(pid).exists()

    def load(
        self,
        prepared_id: str | None = None,
        *,
        graph_id: str | None = None,
        projection_id: str = "default",
        canonical_hash: str | None = None,
    ) -> PreparedGraph:
        """Load by prepared_id, or by graph/projection/canonical hash selector."""
        pid = prepared_id or self._resolve_id(graph_id, projection_id, canonical_hash)
        path = self._path(pid)
        if not path.exists():
            raise FileNotFoundError(
                f"Prepared graph artifact {pid!r} not found in {self.store_dir}. "
                "Run train/materialize first."
            )
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        return PreparedGraph(
            graph_id=data["graph_id"],
            projection_id=data["projection_id"],
            canonical_hash=data["canonical_hash"],
            prepared_id=data["prepared_id"],
            x_c=data["x_c"],
            adj_norm=data["adj_norm"],
            edge_attr_c=data.get("edge_attr_c"),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
        )

    def _resolve_id(
        self,
        graph_id: str | None,
        projection_id: str,
        canonical_hash: str | None,
    ) -> str:
        if graph_id is None:
            raise FileNotFoundError("graph_id is required when prepared_id is not supplied")
        matches = []
        for prepared_id, record in self._index.get("prepared", {}).items():
            if record.get("graph_id") != graph_id:
                continue
            if record.get("projection_id") != projection_id:
                continue
            if canonical_hash is not None and record.get("canonical_hash") != canonical_hash:
                continue
            matches.append((prepared_id, record))
        if not matches:
            raise FileNotFoundError(
                f"No prepared artifact for graph_id={graph_id!r}, "
                f"projection_id={projection_id!r}. Run train/materialize first."
            )
        return matches[-1][0]


def save(prepared_graph: PreparedGraph, store_dir: str | Path = "prepared") -> str:
    return PreparedStore(store_dir).save(prepared_graph)


def load(prepared_id: str | None = None, store_dir: str | Path = "prepared", **kwargs) -> PreparedGraph:
    return PreparedStore(store_dir).load(prepared_id, **kwargs)


def exists(prepared_id: str | None = None, store_dir: str | Path = "prepared", **kwargs) -> bool:
    return PreparedStore(store_dir).exists(prepared_id, **kwargs)
