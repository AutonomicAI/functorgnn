"""Smoke tests for the BackGNN package-local CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from back_gnn import cli
from back_gnn.dataset import generate_dataset
from back_gnn.prepared_store import PreparedStore


def _json_from_output(output: str) -> dict:
    start = output.find("{")
    assert start >= 0, output
    return json.loads(output[start:])


def _seed_chemistry_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    commit_dir = tmp_path / "commits"
    prepared_dir = tmp_path / "prepared"
    generate_dataset(data_dir=str(data_dir), kind="chemistry")
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(cli, "COMMIT_DIR", str(commit_dir))
    monkeypatch.setattr(cli, "PREPARED_DIR", str(prepared_dir))


def test_materialize_command_creates_ethanol_default_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed_chemistry_data(tmp_path, monkeypatch)

    cli.main(["materialize", "--graph", "ethanol", "--projection", "default"])

    result = _json_from_output(capsys.readouterr().out)
    assert result["graph_id"] == "ethanol"
    assert result["projection_id"] == "default"
    assert PreparedStore(tmp_path / "prepared").exists(result["prepared_id"])


def test_train_command_creates_commits_and_prepared_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed_chemistry_data(tmp_path, monkeypatch)

    cli.main(["train", "--epochs", "1", "--commit-every", "1"])

    result = _json_from_output(capsys.readouterr().out)
    assert result["commits_created"] == 1
    assert result["head"]
    assert PreparedStore(tmp_path / "prepared").exists(graph_id="ethanol", projection_id="default")


def test_infer_command_works_after_train_and_includes_probs_logits(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed_chemistry_data(tmp_path, monkeypatch)
    cli.main(["train", "--epochs", "1", "--commit-every", "1"])
    capsys.readouterr()

    cli.main(["infer", "--graph", "ethanol", "--seed", "42", "--iterations", "3"])

    result = _json_from_output(capsys.readouterr().out)
    assert result["iterations"] == 3
    assert result["graph_id"] == "ethanol"
    assert result["seed"] == 42
    assert result["commit_id"]
    assert "last_result" in result
    assert "probs" in result
    assert "logits" in result
    assert "predicted_class" in result


def test_infer_clear_error_without_prepared_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed_chemistry_data(tmp_path, monkeypatch)
    cli.main(["train", "--epochs", "1", "--commit-every", "1"])
    capsys.readouterr()

    monkeypatch.setattr(cli, "PREPARED_DIR", str(tmp_path / "missing_prepared"))
    try:
        cli.main(["infer", "--graph", "ethanol", "--seed", "42", "--iterations", "1"])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("infer should fail without a prepared artifact")

    assert "No prepared graph artifact found. Run train or materialize first." in capsys.readouterr().err
