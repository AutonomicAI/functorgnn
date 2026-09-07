"""Focused tests for the BackGNN backloaded inference refactor."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from back_gnn import materialize, prepare_graph
from back_gnn.canonicalize import canonicalize_edge_attr, canonicalize_graph, normalize_adj
from back_gnn.chemistry import toy_molecule_dataset
from back_gnn.commit_store import CommitStore
from back_gnn.inference import run_inference
from back_gnn.model import FunctorGNN
from back_gnn.prepared_store import PreparedStore
from back_gnn.train import train


def _ethanol() -> dict:
    for graph in toy_molecule_dataset():
        if graph["graph_id"] == "ethanol":
            return graph
    raise AssertionError("ethanol fixture not found")


def _old_frontloaded_logits(model: FunctorGNN, graph: dict) -> torch.Tensor:
    x_c, adj_c, perm = canonicalize_graph(graph["x"], graph["adj"])
    edge_attr_c = canonicalize_edge_attr(graph["edge_attr"], perm)
    return model(x_c, normalize_adj(adj_c), edge_attr_c)


def test_materialize_produces_prepared_tensors():
    prepared = materialize.prepare(prepare_graph.prepare([_ethanol()]))[0]

    assert prepared.graph_id == "ethanol"
    assert prepared.projection_id == "default"
    assert prepared.canonical_hash
    assert prepared.x_c is not None
    assert prepared.adj_norm is not None
    assert prepared.edge_attr_c is not None


def test_prepared_store_round_trips_prepared_graph(tmp_path):
    prepared = materialize.prepare(prepare_graph.prepare([_ethanol()]))[0]
    store = PreparedStore(tmp_path)

    prepared_id = store.save(prepared)
    loaded = store.load(prepared_id)

    assert store.exists(prepared_id)
    assert loaded.prepared_id == prepared.prepared_id
    assert loaded.graph_id == prepared.graph_id
    torch.testing.assert_close(loaded.x_c, prepared.x_c)
    torch.testing.assert_close(loaded.adj_norm, prepared.adj_norm)
    torch.testing.assert_close(loaded.edge_attr_c, prepared.edge_attr_c)


def test_inference_does_not_call_canonicalization_or_normalization_directly():
    import back_gnn.inference as inference

    source = inspect.getsource(inference)
    assert "canonicalize_graph" not in source
    assert "canonicalize_edge_attr" not in source
    assert "normalize_adj" not in source
    assert "apply_projection_features" not in source


def test_inference_matches_old_frontloaded_default_ethanol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _ethanol()
    model = FunctorGNN(in_dim=int(graph["x"].shape[1]))
    store = CommitStore(tmp_path / "commits")
    commit_id = store.commit(model, message="initial")
    PreparedStore(tmp_path / "prepared").save(materialize.prepare([graph])[0])

    old_model = FunctorGNN(in_dim=int(graph["x"].shape[1]))
    store.load_into(old_model, commit_id)
    with torch.no_grad():
        old_logits = _old_frontloaded_logits(old_model, graph)

    with mock.patch("back_gnn.inference.log_inference_event"):
        result = run_inference(model, store, commit_id, graph, seed=42, graph_id="ethanol")

    assert result["logits"] == [round(float(value), 8) for value in old_logits.tolist()]


def test_train_creates_prepared_artifacts_before_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _ethanol()
    model = FunctorGNN(in_dim=int(graph["x"].shape[1]))
    store = CommitStore(tmp_path / "commits")

    committed = train(model, [graph], store, epochs=1, commit_every=1, seed=0)

    assert committed
    metadata = store.get_commit(committed[-1][1])["metadata"]
    assert metadata["prepared_graphs"] == 1
    assert metadata["prepared_artifact_ids"]
    assert PreparedStore(tmp_path / "prepared").exists(metadata["prepared_artifact_ids"][0])

    with mock.patch("back_gnn.inference.log_inference_event"):
        result = run_inference(model, store, committed[-1][1], graph, seed=42, graph_id="ethanol")
    assert result["prepared_id"] == metadata["prepared_artifact_ids"][0]
