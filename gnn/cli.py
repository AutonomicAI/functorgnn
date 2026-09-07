#!/usr/bin/env python3
"""
Functor GNN — Command-Line Interface

Commands:
    generate        Generate synthetic, toy chemistry, ESOL, or BACE dataset
    train           Train model and commit weight snapshots
    commits         List all commits in the store
    infer           Run commit-indexed inference on a graph
    rollback-demo   Demonstrate rollback to earlier commits
    equiv-demo      Demonstrate deterministic replay (same inputs → same output)
    projection-equiv-demo
                    Demonstrate non-identity projection equivalence
    log             Show structured inference log
"""

import argparse
import copy
import hashlib
import json
import sys


import torch

from functor_gnn import CommitStore, FunctorGNN
from functor_gnn import generate_dataset, get_graph, load_dataset
from functor_gnn import run_inference
from functor_gnn.canonicalize import canonicalize_edge_attr, canonicalize_graph
from functor_gnn.logger import read_log
from functor_gnn.projection import apply_projection_features, get_projection
from functor_gnn.train import train as _train

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

COMMIT_DIR   = "commits"
DATA_DIR     = "data"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_store() -> CommitStore:
    return CommitStore(COMMIT_DIR)


def _build_model(store: CommitStore) -> FunctorGNN:
    config = store.load_model_config()
    return FunctorGNN(**config)


def _require_head(store: CommitStore) -> str:
    head = store.head()
    if head is None:
        print("[error] No commits found. Run:  python cli.py train")
        sys.exit(1)
    return head


def _task_and_problem_type(graphs: list) -> tuple[str, str]:
    task = graphs[0].get("task", "cycle_detection")
    return task, "regression" if task == "esol" else "classification"


def _format_inference_result(result: dict) -> str:
    if result.get("task") == "esol":
        return f"prediction={result['prediction']}  task={result['task']}"
    return f"class={result['predicted_class']}  probs={result['probs']}"


def _result_signature(result: dict):
    return result["prediction"] if result.get("task") == "esol" else result["logits"]


def _result_close_enough(a: dict, b: dict) -> bool:
    """Bit/display-level equality for demo outputs."""
    return _result_signature(a) == _result_signature(b)


def _tensor_payload(t: torch.Tensor):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "values": t.detach().cpu().tolist(),
    }


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _raw_graph_hash(graph: dict) -> str:
    payload = {
        "x": _tensor_payload(graph["x"]),
        "adj": _tensor_payload(graph["adj"]),
    }
    if "edge_attr" in graph:
        payload["edge_attr"] = _tensor_payload(graph["edge_attr"])
    return _stable_hash(payload)


def _canonical_projected_graph_hash(graph: dict, projection_id: str) -> str:
    proj = get_projection(projection_id)
    if proj is None:
        raise ValueError("projection-equiv-demo requires a non-identity projection")
    x = apply_projection_features(graph["x"], proj)
    x_c, adj_c, perm = canonicalize_graph(x, graph["adj"])
    payload = {
        "projection_id": proj.projection_id,
        "x_c": _tensor_payload(x_c),
        "adj_c": _tensor_payload(adj_c),
    }
    if "edge_attr" in graph:
        payload["edge_attr_c"] = _tensor_payload(canonicalize_edge_attr(graph["edge_attr"], perm))
    return _stable_hash(payload)


def _permute_graph(graph: dict, perm: torch.Tensor, graph_id: str) -> dict:
    """Return a distinct raw graph representation with nodes reordered."""
    g = copy.deepcopy(graph)
    g["graph_id"] = graph_id
    g["x"] = graph["x"][perm].clone()
    g["adj"] = graph["adj"][perm][:, perm].clone()
    if "edge_attr" in graph:
        g["edge_attr"] = graph["edge_attr"][perm][:, perm].clone()
    g["raw_basis"] = f"node permutation {perm.tolist()} of {graph.get('graph_id', 'unknown')}"
    return g


def _collapsed_feature_variant(graph: dict, projection_id: str, graph_id: str) -> dict | None:
    """
    Return a raw-distinct graph that is identical after projection by changing
    only a feature channel whose projection weight is exactly zero.
    """
    proj = get_projection(projection_id)
    if proj is None:
        return None
    weights = proj.weights_for_dim(int(graph["x"].shape[1]))
    zero_cols = (weights == 0).nonzero(as_tuple=False).view(-1)
    if zero_cols.numel() == 0:
        return None
    g = copy.deepcopy(graph)
    g["graph_id"] = graph_id
    col = int(zero_cols[0].item())
    g["x"] = graph["x"].clone()
    g["x"][0, col] = g["x"][0, col] + 1.0
    g["raw_basis"] = (
        f"feature-column {col} perturbation collapsed by projection {projection_id}"
    )
    return g


def _commits_compatible_with_model(store: CommitStore, model: FunctorGNN) -> list[dict]:
    """
    Return commits whose saved tensor shapes match the current model config.

    The commit registry is append-only, so a local demo directory may contain
    older checkpoints from a different dataset/model input dimension. Rollback
    demos must choose from the compatible subsequence, otherwise loading the
    first historical commit can fail with a PyTorch size-mismatch error.
    """
    expected = model.state_dict()
    compatible: list[dict] = []
    for c in store.list_commits():
        try:
            state = store.project(c["commit_id"])
        except Exception:
            continue
        if state.keys() != expected.keys():
            continue
        if all(state[k].shape == expected[k].shape for k in expected):
            compatible.append(c)
    return compatible


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    generate_dataset(
        seed=args.seed,
        n_graphs=args.n_graphs,
        data_dir=DATA_DIR,
        kind=args.dataset,
    )


def cmd_train(args: argparse.Namespace) -> None:
    graphs = load_dataset(DATA_DIR)
    store  = _get_store()
    in_dim = int(graphs[0]["x"].shape[1])
    task, problem_type = _task_and_problem_type(graphs)
    num_classes = 1 if problem_type == "regression" else len({int(g["label"]) for g in graphs})

    model = FunctorGNN(
        in_dim     = in_dim,
        hidden_dim = args.hidden_dim,
        num_classes= num_classes,
        num_layers = args.num_layers,
    )
    store.save_model_config(model.config)

    print(
        f"[train] {len(graphs)} graphs | in_dim={in_dim} | "
        f"task={task} | problem_type={problem_type} | outputs={num_classes} | "
        f"epochs={args.epochs} | "
        f"lr={args.lr} | commit-every={args.commit_every} | seed={args.seed}"
    )
    committed = _train(
        model        = model,
        graphs       = graphs,
        commit_store = store,
        epochs       = args.epochs,
        lr           = args.lr,
        commit_every = args.commit_every,
        seed         = args.seed,
    )
    print(f"\n[train] Done. {len(committed)} commits created.")
    print(f"[train] HEAD = {store.head()}")


def cmd_commits(args: argparse.Namespace) -> None:
    store   = _get_store()
    commits = store.list_commits()
    if not commits:
        print("No commits yet. Run:  python cli.py train")
        return
    head = store.head()
    header = f"{'#':<4}  {'COMMIT ID':<18}  {'MESSAGE':<16}  {'TASK':<22}  {'TYPE':<14}  {'METRIC':>12}  {'LOSS':>7}  TIMESTAMP"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(commits):
        meta    = c.get("metadata", {})
        task = str(meta.get("task", "unknown"))
        problem_type = str(meta.get("problem_type", "unknown"))
        if "rmse" in meta:
            metric_str = f"rmse={meta['rmse']:.4f}"
        elif "acc" in meta and task not in {"contains_hetero_atom", "cycle_detection", "unknown"}:
            metric_str = f"acc={meta['acc']:.1%}"
        elif "acc" in meta:
            metric_str = "fixture/legacy"
        else:
            metric_str = "—"
        loss_str= f"{meta['loss']:.4f}" if "loss" in meta else "—"
        marker  = "  ← HEAD" if c["commit_id"] == head else ""
        print(
            f"{i:<4}  {c['commit_id']:<18}  {c['message']:<16}  "
            f"{task:<22}  {problem_type:<14}  {metric_str:>12}  "
            f"{loss_str:>7}  {c['timestamp']}{marker}"
        )


def cmd_infer(args: argparse.Namespace) -> None:
    store     = _get_store()
    model     = _build_model(store)
    graphs    = load_dataset(DATA_DIR)
    commit_id = args.commit or _require_head(store)
    graph     = get_graph(graphs, args.graph)

    result = run_inference(
        model        = model,
        commit_store = store,
        commit_id    = commit_id,
        graph        = graph,
        seed         = args.seed,
        graph_id     = args.graph,
    )
    print("\n[infer] Result:")
    print(json.dumps(result, indent=2))


def cmd_rollback_demo(args: argparse.Namespace) -> None:
    store   = _get_store()
    model   = _build_model(store)
    commits = _commits_compatible_with_model(store, model)

    if len(commits) < 2:
        print(
            "[error] Need at least 2 commits compatible with the current model "
            "config for a rollback demo.\n"
            "Run:  python cli.py train --epochs 30 --commit-every 10"
        )
        sys.exit(1)

    graphs     = load_dataset(DATA_DIR)
    graph      = get_graph(graphs, args.graph)
    first_c    = commits[0]
    last_c     = commits[-1]

    _sep = "=" * 64

    print(_sep)
    print("  ROLLBACK DEMO")
    print(_sep)
    print(f"  Graph  : {args.graph}  |  Seed : {args.seed}")
    print(f"  Commits: {len(commits)} compatible with current model config")
    print(f"           ({store.count()} total records in commit store)")
    print(f"  First  : {first_c['commit_id']}  ({first_c['message']})")
    print(f"  Last   : {last_c['commit_id']}   ({last_c['message']})")
    print()

    print("[1/4]  Inference under LAST commit (most-trained model)")
    r_last_a = run_inference(
        model, store, last_c["commit_id"], graph, args.seed, args.graph
    )
    print(f"       {_format_inference_result(r_last_a)}")

    print("\n[2/4]  Rollback → FIRST commit (early, undertrained model)")
    r_first_a = run_inference(
        model, store, first_c["commit_id"], graph, args.seed, args.graph
    )
    print(f"       {_format_inference_result(r_first_a)}")

    print("\n[3/4]  Replay FIRST commit (reproducibility check)")
    r_first_b = run_inference(
        model, store, first_c["commit_id"], graph, args.seed, args.graph
    )
    print(f"       {_format_inference_result(r_first_b)}")

    print("\n[4/4]  Re-run LAST commit (reproducibility check)")
    r_last_b = run_inference(
        model, store, last_c["commit_id"], graph, args.seed, args.graph
    )
    print(f"       {_format_inference_result(r_last_b)}")

    print()
    print(_sep)
    ok_first  = _result_signature(r_first_a) == _result_signature(r_first_b)
    ok_last   = _result_signature(r_last_a)  == _result_signature(r_last_b)
    different = _result_signature(r_first_a) != _result_signature(r_last_a)
    print(f"  ✓ First commit reproducible  (run1 == run2) : {ok_first}")
    print(f"  ✓ Last  commit reproducible  (run1 == run2) : {ok_last}")
    print(f"  ✓ First ≠ Last (distinct model state)       : {different}")
    print()
    print("  Rollback is lossless: compatible commits replay bit-exactly.")
    print("  Projection context: identity/default; no non-identity projection theorem is claimed.")
    print(_sep)


def cmd_equiv_demo(args: argparse.Namespace) -> None:
    store     = _get_store()
    commit_id = _require_head(store)
    model     = _build_model(store)
    graphs    = load_dataset(DATA_DIR)
    graph     = get_graph(graphs, args.graph)

    _sep = "=" * 64

    print(_sep)
    print("  INFERENCE EQUIVALENCE DEMO")
    print("  Demonstrates bit-exact deterministic replay under fixed (commit, graph, seed).")
    print("  Projection context: identity/default.")
    print(_sep)
    print(f"  Commit : {commit_id}")
    print(f"  Graph  : {args.graph}")
    print(f"  Seed   : {args.seed}")
    print()

    print("[Run 1]")
    r1 = run_inference(model, store, commit_id, graph, args.seed, args.graph)
    if r1.get("task") == "esol":
        print(f"  prediction      = {r1['prediction']}")
        print(f"  task            = {r1['task']}")
    else:
        print(f"  predicted_class = {r1['predicted_class']}")
        print(f"  probs           = {r1['probs']}")
        print(f"  logits          = {r1['logits']}")

    print("\n[Run 2]  (identical commit + graph + seed)")
    r2 = run_inference(model, store, commit_id, graph, args.seed, args.graph)
    if r2.get("task") == "esol":
        print(f"  prediction      = {r2['prediction']}")
        print(f"  task            = {r2['task']}")
    else:
        print(f"  predicted_class = {r2['predicted_class']}")
        print(f"  probs           = {r2['probs']}")
        print(f"  logits          = {r2['logits']}")

    print()
    print(_sep)
    if r1.get("task") == "esol":
        prediction_match = r1["prediction"] == r2["prediction"]
        print(f"  ✓ prediction identical : {prediction_match}")
    else:
        logits_match = r1["logits"] == r2["logits"]
        probs_match  = r1["probs"]  == r2["probs"]
        print(f"  ✓ logits identical : {logits_match}")
        print(f"  ✓ probs  identical : {probs_match}")
    print()
    print("  Same (commit, graph, seed) yielded identical output in this replay.")
    print("  Identity projection only; projection-equivalence content is not exercised here.")
    print(_sep)


def cmd_projection_equiv_demo(args: argparse.Namespace) -> None:
    store = _get_store()
    commit_id = _require_head(store)
    model = _build_model(store)
    graphs = load_dataset(DATA_DIR)
    base = get_graph(graphs, args.graph)
    projection_id = args.projection

    proj = get_projection(projection_id)
    if proj is None:
        print("[error] projection-equiv-demo requires a non-identity projection.")
        sys.exit(1)

    if int(base["x"].shape[0]) < 2:
        print("[error] Need a graph with at least two nodes for the positive permutation case.")
        sys.exit(1)

    # Positive case: prefer a raw feature perturbation intentionally collapsed
    # by the non-identity projection. If a caller supplies a projection with no
    # zero-weight feature, fall back to a node-ordering permutation.
    g1 = copy.deepcopy(base)
    g1["graph_id"] = f"{base.get('graph_id', args.graph)}__raw_A"
    g2 = _collapsed_feature_variant(
        base,
        projection_id,
        f"{base.get('graph_id', args.graph)}__raw_B_collapsed_feature",
    )
    if g2 is None:
        perm = torch.arange(base["x"].shape[0] - 1, -1, -1, dtype=torch.long)
        g2 = _permute_graph(base, perm, f"{base.get('graph_id', args.graph)}__raw_B_permuted")

    raw1 = _raw_graph_hash(g1)
    raw2 = _raw_graph_hash(g2)
    pi1 = _canonical_projected_graph_hash(g1, projection_id)
    pi2 = _canonical_projected_graph_hash(g2, projection_id)

    r1 = run_inference(model, store, commit_id, g1, args.seed, g1["graph_id"], projection=projection_id)
    r2 = run_inference(model, store, commit_id, g2, args.seed, g2["graph_id"], projection=projection_id)

    fw1 = r1.get("projection_feature_weights_hash")
    fw2 = r2.get("projection_feature_weights_hash")
    raw_distinct = raw1 != raw2
    canonical_equal = pi1 == pi2
    non_null_projection = fw1 is not None and fw1 == fw2
    outputs_equal = _result_close_enough(r1, r2)

    # Negative control: find a graph with a distinct projected canonical form
    # that also yields distinct logits/prediction under the same projection.
    neg = None
    neg_hash = None
    neg_result = None
    for candidate in graphs:
        candidate_hash = _canonical_projected_graph_hash(candidate, projection_id)
        if candidate_hash == pi1:
            continue
        candidate_result = run_inference(
            model,
            store,
            commit_id,
            candidate,
            args.seed,
            f"{candidate.get('graph_id', 'candidate')}__negative",
            projection=projection_id,
        )
        if not _result_close_enough(r1, candidate_result):
            neg = candidate
            neg_hash = candidate_hash
            neg_result = candidate_result
            break

    negative_ok = neg is not None

    _sep = "=" * 72
    print(_sep)
    print("  PROJECTION-EQUIVALENCE DEMO")
    print("  Demonstrates F̃(c,G,ω) = F(π(c),G,ω) for an exercised non-identity projection.")
    print(_sep)
    print(f"  Commit     : {commit_id}")
    print(f"  Seed       : {args.seed}")
    print(f"  Projection : {proj.projection_id}")
    print(f"  Feature-weight hash (non-null): {fw1}")
    print()
    print("[Positive control: distinct raw inputs collapse to one canonical projection]")
    print(f"  G1 raw id/hash       : {g1['graph_id']} / {raw1}")
    print(f"  G2 raw id/hash       : {g2['graph_id']} / {raw2}")
    print(f"  G1 != G2             : {raw_distinct}")
    print(f"  π(G1) canonical hash : {pi1}")
    print(f"  π(G2) canonical hash : {pi2}")
    print(f"  π(G1) == π(G2)       : {canonical_equal}")
    print(f"  projection hash set  : {non_null_projection}")
    print(f"  G1 output            : {_format_inference_result(r1)}")
    print(f"  G2 output            : {_format_inference_result(r2)}")
    print(f"  outputs identical    : {outputs_equal}")
    print()
    print("[Negative control: distinct projection gives distinct output]")
    if negative_ok:
        print(f"  G3 raw id/hash       : {neg.get('graph_id', 'unknown')} / {_raw_graph_hash(neg)}")
        print(f"  π(G3) canonical hash : {neg_hash}")
        print(f"  π(G1) != π(G3)       : {pi1 != neg_hash}")
        print(f"  G3 output            : {_format_inference_result(neg_result)}")
        print(f"  G1 output            : {_format_inference_result(r1)}")
        print(f"  outputs differ       : {not _result_close_enough(r1, neg_result)}")
    else:
        print("  No negative-control graph with distinct projection and distinct output was found.")
    print()

    all_ok = raw_distinct and canonical_equal and non_null_projection and outputs_equal and negative_ok
    print(_sep)
    print(f"  Acceptance checks passed: {all_ok}")
    if all_ok:
        print("  Projection equivalence demonstrated for the exercised non-identity projection.")
    else:
        print("  Projection equivalence NOT claimed; one or more checks failed.")
    print(_sep)
    if not all_ok:
        sys.exit(1)


def cmd_log(args: argparse.Namespace) -> None:
    records = read_log(tail=args.tail)
    if not records:
        print("No inference records yet. Run 'infer' or a demo first.")
        return
    header = f"{'RUN_ID':<10}  {'COMMIT_ID':<18}  {'SEED':>5}  {'GRAPH_ID':<14}  {'OUTPUT':>12}  TIMESTAMP"
    print(header)
    print("-" * len(header))
    for r in records:
        output = f"pred={r['prediction']:.4f}" if "prediction" in r else f"class={r['predicted_class']}"
        print(
            f"{r['run_id']:<10}  {r['commit_id']:<18}  {r['seed']:>5}  "
            f"{r['graph_id']:<14}  {output:>12}  {r['timestamp']}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "functor-gnn",
        description = "Functor GNN — commit-indexed graph inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py generate
  python cli.py generate --dataset chemistry
  python cli.py generate --dataset esol
  python cli.py generate --dataset bace
  python cli.py train --epochs 30 --commit-every 10
  python cli.py commits
  python cli.py infer --graph cycle_00 --seed 42
  python cli.py infer --graph ethanol --seed 42
  python cli.py equiv-demo --graph cycle_00 --seed 42
  python cli.py projection-equiv-demo --graph ethanol --seed 42
  python cli.py rollback-demo --graph tree_03 --seed 7
  python cli.py log --tail 10
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate synthetic, toy chemistry, ESOL, or BACE dataset")
    p.add_argument("--dataset", choices=["synthetic", "chemistry", "esol", "bace"], default="synthetic",
                   help="Dataset kind to generate (default: synthetic)")
    p.add_argument("--n-graphs", type=int, default=20, metavar="N",
                   help="Total synthetic graphs to generate; ignored for chemistry/ESOL/BACE (default: 20)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Dataset generation seed (default: 42)")

    # train
    p = sub.add_parser("train", help="Train model and commit checkpoints")
    p.add_argument("--epochs",       type=int,   default=30,   help="Training epochs (default: 30)")
    p.add_argument("--lr",           type=float, default=0.01, help="Learning rate (default: 0.01)")
    p.add_argument("--commit-every", type=int,   default=10,   help="Commit every N epochs (default: 10)")
    p.add_argument("--hidden-dim",   type=int,   default=32,   help="Hidden layer width (default: 32)")
    p.add_argument("--num-layers",   type=int,   default=2,    help="Number of MP layers (default: 2)")
    p.add_argument("--seed",         type=int,   default=DEFAULT_SEED, help="Training seed (default: 42)")

    # commits
    sub.add_parser("commits", help="List all commits")

    # infer
    p = sub.add_parser("infer", help="Run commit-indexed inference on a graph")
    p.add_argument("--graph",  required=True, help="Graph ID (e.g. cycle_00, tree_03, ethanol)")
    p.add_argument("--commit", default=None,  help="Commit ID to use (default: HEAD)")
    p.add_argument("--seed",   type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")

    # rollback-demo
    p = sub.add_parser("rollback-demo",
                        help="Demonstrate rollback: replay earlier commits reproducibly")
    p.add_argument("--graph", default="cycle_00", help="Graph to use (default: cycle_00)")
    p.add_argument("--seed",  type=int, default=DEFAULT_SEED)

    # equiv-demo
    p = sub.add_parser("equiv-demo",
                        help="Demonstrate inference equivalence under fixed (commit, graph, seed)")
    p.add_argument("--graph", default="cycle_00", help="Graph to use (default: cycle_00)")
    p.add_argument("--seed",  type=int, default=DEFAULT_SEED)

    # projection-equiv-demo
    p = sub.add_parser(
        "projection-equiv-demo",
        help="Demonstrate non-identity projection equivalence with positive and negative controls",
    )
    p.add_argument("--graph", default="ethanol", help="Base graph for positive control (default: ethanol)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--projection",
        default="assay_hydrogen_blind",
        help="Non-identity projection id to exercise (default: assay_polarity)",
    )

    # log
    p = sub.add_parser("log", help="Show structured inference log")
    p.add_argument("--tail", type=int, default=20, help="Last N records (default: 20)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DISPATCH = {
    "generate":      cmd_generate,
    "train":         cmd_train,
    "commits":       cmd_commits,
    "infer":         cmd_infer,
    "rollback-demo": cmd_rollback_demo,
    "equiv-demo":    cmd_equiv_demo,
    "projection-equiv-demo": cmd_projection_equiv_demo,
    "log":           cmd_log,
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
