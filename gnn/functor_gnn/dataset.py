"""
Dataset generation for the Functor GNN demo.

Default synthetic task: predict whether a graph contains a cycle.
    label 0 → acyclic (randomly wired tree)
    label 1 → cyclic  (simple cycle graph)

Node features are derived purely from graph structure (degree-based),
making them deterministic given the adjacency matrix. This aligns with
the Functor Model's canonicalization principle: structurally equivalent
graphs produce identical feature matrices.

Feature vector per node (dim=4):
    [degree,  degree²,  1/degree,  is_leaf(degree==1)]

Optional restricted chemistry task: classify toy molecule graphs by whether
they contain a hetero atom. Chemistry encoding lives in chemistry.py and is a
proof-of-architecture subset, not a production chemistry model.

Optional benchmark chemistry tasks: load ESOL or BACE MoleculeNet datasets via
the adapter in moleculenet.py. These adapters require DeepChem and RDKit.
"""

import json
from pathlib import Path
import numpy as np
import torch

from .chemistry import toy_molecule_dataset
from .moleculenet import load_moleculenet_dataset


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _structural_features(adj: torch.Tensor) -> torch.Tensor:
    """
    Derive 4-dimensional structural node features from the adjacency matrix.

    f_v = [deg_v,  deg_v²,  1/deg_v,  1(deg_v == 1)]
    """
    deg = adj.sum(dim=1)                        # (N,)
    return torch.stack([
        deg,
        deg ** 2,
        1.0 / deg.clamp(min=1.0),              # inverse degree
        (deg == 1).float(),                    # leaf indicator
    ], dim=1).float()                          # (N, 4)


# ---------------------------------------------------------------------------
# Graph constructors
# ---------------------------------------------------------------------------

def _make_cycle(n: int) -> dict:
    """Cycle graph C_n — every node has degree 2, no leaves. Label = 1."""
    adj = torch.zeros(n, n)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1.0
    return {
        "adj": adj,
        "x": _structural_features(adj),
        "label": 1,
        "type": "cycle",
        "n": n,
        "task": "cycle_detection",
    }


def _make_tree(n: int, rng: np.random.Generator) -> dict:
    """
    Random labelled tree via Prüfer-style growth — at least one leaf.
    Label = 0.
    """
    adj = torch.zeros(n, n)
    for i in range(1, n):
        parent = int(rng.integers(0, i))
        adj[i, parent] = adj[parent, i] = 1.0
    return {
        "adj": adj,
        "x": _structural_features(adj),
        "label": 0,
        "type": "tree",
        "n": n,
        "task": "cycle_detection",
    }


# ---------------------------------------------------------------------------
# Dataset generation and loading
# ---------------------------------------------------------------------------

def generate_dataset(
    seed: int = 42,
    n_graphs: int = 20,
    data_dir: str = "data",
    kind: str = "synthetic",
) -> list:
    """
    Generate and save a dataset.

    kind="synthetic" preserves the original cycle-vs-tree demo. Half the
    graphs are cycles (label=1), half are trees (label=0), and node counts are
    drawn uniformly from [4, 8].

    kind="chemistry" writes a deterministic toy molecule dataset using the
    restricted chemistry subset implemented in chemistry.py. The molecule
    labels are for a demo task only: contains a hetero atom (1) vs hydrocarbon
    subset (0).

    kind="esol" or kind="bace" delegates MoleculeNet benchmark loading and
    SMILES parsing to moleculenet.py. These paths require DeepChem and RDKit.

    Saves:
        data/graphs.pt   — list of graph dicts (torch.save)
        data/meta.json   — lightweight JSON metadata for inspection
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if kind not in {"synthetic", "chemistry", "esol", "bace"}:
        raise ValueError("kind must be one of 'synthetic', 'chemistry', 'esol', or 'bace'.")

    graphs: list[dict] = []
    if kind in {"esol", "bace"}:
        graphs = load_moleculenet_dataset(kind, data_dir=data_dir)
    elif kind == "chemistry":
        graphs = toy_molecule_dataset()
    else:
        half = n_graphs // 2

        for i in range(half):
            n = int(rng.integers(4, 9))
            g = _make_cycle(n)
            g["graph_id"] = f"cycle_{i:02d}"
            graphs.append(g)

        for i in range(half):
            n = int(rng.integers(4, 9))
            g = _make_tree(n, rng)
            g["graph_id"] = f"tree_{i:02d}"
            graphs.append(g)

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(graphs, out / "graphs.pt")

    meta = [
        {
            "graph_id": g["graph_id"],
            "label":    g["label"],
            "type":     g["type"],
            "n":        g["n"],
            "task":     g.get("task", "cycle_detection"),
            "feature_dim": int(g["x"].shape[1]),
            "edge_feature_dim": int(g.get("edge_attr", torch.empty(0, 0, 0)).shape[-1])
                                if "edge_attr" in g else 0,
        }
        for g in graphs
    ]
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[dataset] {len(graphs)} {kind} graphs saved → {out}/")
    return graphs


def load_dataset(data_dir: str = "data") -> list:
    """Load dataset from disk. Raises FileNotFoundError if not generated yet."""
    path = Path(data_dir) / "graphs.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run:  python cli.py generate "
            "[--dataset synthetic|chemistry|esol|bace]"
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_graph(graphs: list, graph_id: str) -> dict:
    """Retrieve a specific graph by ID. Raises ValueError if not found."""
    for g in graphs:
        if g["graph_id"] == graph_id:
            return g
    available = [g["graph_id"] for g in graphs]
    raise ValueError(
        f"Graph {graph_id!r} not found. Available: {available}"
    )
