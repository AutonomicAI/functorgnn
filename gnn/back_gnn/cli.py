"""BackGNN command-line entry point for backloaded benchmarking.

Examples:
  python3 -m back_gnn.cli train --epochs 30 --commit-every 10
  python3 -m back_gnn.cli infer --graph ethanol --seed 42 --iterations 100000
  python3 -m back_gnn.cli materialize --graph ethanol --projection default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .commit_store import CommitStore
from .dataset import get_graph, load_dataset
from .inference import run_inference
from .materialize import prepare as materialize_graphs
from .model import FunctorGNN
from .prepare_graph import prepare as prepare_raw_graphs
from .prepared_store import PreparedStore
from .train import train as train_model


_PACKAGE_DIR = Path(__file__).resolve().parent
_GNN_DIR = _PACKAGE_DIR.parent
COMMIT_DIR = str(_PACKAGE_DIR / "commits")
DATA_DIR = str(_GNN_DIR / "data")
PREPARED_DIR = str(_PACKAGE_DIR / "prepared")
DEFAULT_SEED = 42


def _get_store() -> CommitStore:
    return CommitStore(COMMIT_DIR)


def _get_prepared_store() -> PreparedStore:
    return PreparedStore(PREPARED_DIR)


def _require_head(store: CommitStore) -> str:
    head = store.head()
    if head is None:
        raise RuntimeError("No commits found. Run train first.")
    return head


def _task_and_problem_type(graphs: list[dict]) -> tuple[str, str]:
    task = graphs[0].get("task", "cycle_detection")
    return task, "regression" if task == "esol" else "classification"


def _build_model_for_training(graphs: list[dict], args: argparse.Namespace) -> FunctorGNN:
    in_dim = int(graphs[0]["x"].shape[1])
    _, problem_type = _task_and_problem_type(graphs)
    num_classes = 1 if problem_type == "regression" else len({int(g["label"]) for g in graphs})
    return FunctorGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
    )


def _build_model_for_inference(store: CommitStore) -> FunctorGNN:
    return FunctorGNN(**store.load_model_config())


def _normalize_projection_id(projection: str | None) -> str:
    return "default" if projection in (None, "", "identity") else str(projection)


def cmd_materialize(args: argparse.Namespace) -> dict:
    graphs = load_dataset(DATA_DIR)
    graph = get_graph(graphs, args.graph)
    raw = prepare_raw_graphs([graph])
    prepared = materialize_graphs(raw, projection=None if args.projection == "default" else args.projection)[0]
    prepared_id = _get_prepared_store().save(prepared)
    result = {
        "command": "materialize",
        "graph_id": prepared.graph_id,
        "projection_id": prepared.projection_id,
        "prepared_id": prepared_id,
        "canonical_hash": prepared.canonical_hash,
    }
    print(json.dumps(result, indent=2))
    return result


def cmd_train(args: argparse.Namespace) -> dict:
    graphs = load_dataset(DATA_DIR)
    store = _get_store()
    model = _build_model_for_training(graphs, args)
    store.save_model_config(model.config)

    committed = train_model(
        model=model,
        graphs=graphs,
        commit_store=store,
        epochs=args.epochs,
        lr=args.lr,
        commit_every=args.commit_every,
        seed=args.seed,
        prepared_dir=PREPARED_DIR,
    )
    result = {
        "command": "train",
        "epochs": args.epochs,
        "commit_every": args.commit_every,
        "commits_created": len(committed),
        "head": store.head(),
        "commits": [{"epoch": epoch, "commit_id": cid} for epoch, cid in committed],
    }
    print(json.dumps(result, indent=2))
    return result


def _hot_result_from_logits(base: dict, logits: torch.Tensor) -> dict:
    result = dict(base)
    if result.get("task") == "esol":
        result["prediction"] = round(float(logits.view(-1)[0].item()), 8)
        return result
    probs = torch.softmax(logits, dim=-1)
    result.update({
        "predicted_class": int(probs.argmax().item()),
        "probs": [round(float(p), 8) for p in probs.tolist()],
        "logits": [round(float(v), 8) for v in logits.tolist()],
    })
    return result


def cmd_infer(args: argparse.Namespace) -> dict:
    graphs = load_dataset(DATA_DIR)
    graph = get_graph(graphs, args.graph)
    store = _get_store()
    commit_id = args.commit or _require_head(store)
    model = _build_model_for_inference(store)
    projection_id = _normalize_projection_id(args.projection)

    if not _get_prepared_store().exists(graph_id=args.graph, projection_id=projection_id):
        raise FileNotFoundError("No prepared graph artifact found. Run train or materialize first.")

    last_result = run_inference(
        model=model,
        commit_store=store,
        commit_id=commit_id,
        graph=graph,
        seed=args.seed,
        graph_id=args.graph,
        projection=None if projection_id == "default" else projection_id,
        prepared_dir=PREPARED_DIR,
    )

    # For benchmark iterations, keep inference backloaded and avoid repeated
    # logging/commit loads. The first call above verifies the public inference
    # path and produces a compatible logged result; the loop below exercises the
    # prepared-artifact forward path.
    if args.iterations > 1:
        prepared = _get_prepared_store().load(graph_id=args.graph, projection_id=projection_id)
        with torch.no_grad():
            logits = None
            for _ in range(args.iterations - 1):
                logits = model(prepared.x_c, prepared.adj_norm, prepared.edge_attr_c)
        if logits is not None:
            last_result = _hot_result_from_logits(last_result, logits)

    result = {
        "command": "infer",
        "iterations": args.iterations,
        "graph_id": args.graph,
        "seed": args.seed,
        "commit_id": commit_id,
        "last_result": last_result,
    }
    if "probs" in last_result:
        result["probs"] = last_result["probs"]
    if "logits" in last_result:
        result["logits"] = last_result["logits"]
    if "predicted_class" in last_result:
        result["predicted_class"] = last_result["predicted_class"]
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="back-gnn", description="Backloaded BackGNN CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("materialize", help="Materialize prepared graph artifacts without training")
    p.add_argument("--graph", required=True, help="Graph ID, e.g. ethanol")
    p.add_argument("--projection", default="default", help="Projection ID (default: default)")
    p.set_defaults(func=cmd_materialize)

    p = sub.add_parser("train", help="Train model and commit checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--commit-every", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer", help="Run prepared-artifact inference")
    p.add_argument("--graph", required=True, help="Graph ID, e.g. ethanol")
    p.add_argument("--commit", default=None, help="Commit ID (default: HEAD)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--projection", default="default", help="Projection ID (default: default)")
    p.set_defaults(func=cmd_infer)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as exc:
        message = str(exc)
        if "prepared" in message.lower():
            message = "No prepared graph artifact found. Run train or materialize first."
        print(f"[error] {message}", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
