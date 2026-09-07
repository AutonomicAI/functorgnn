"""
Restricted chemistry graph encoding for the Functor GNN demo.

This module is intentionally small and deterministic. It supports only the
following proof-of-architecture subset:

Atoms:
    C, H, N, O, S, F, Cl, Br

Bonds:
    single, double, triple, aromatic

It is not a production cheminformatics toolkit. In particular, it does not
parse SMILES, infer valence, add implicit hydrogens, or validate chemistry
beyond the explicit domain constraints below.
"""

from __future__ import annotations

from typing import Iterable

import torch


SUPPORTED_ATOMS: tuple[str, ...] = ("C", "H", "N", "O", "S", "F", "Cl", "Br")
SUPPORTED_BONDS: tuple[str, ...] = ("single", "double", "triple", "aromatic")
SUPPORTED_FORMAL_CHARGES: tuple[int, ...] = (-1, 0, 1)

ATOM_TO_INDEX = {symbol: i for i, symbol in enumerate(SUPPORTED_ATOMS)}
BOND_TO_INDEX = {bond: i for i, bond in enumerate(SUPPORTED_BONDS)}
FORMAL_CHARGE_TO_INDEX = {charge: i for i, charge in enumerate(SUPPORTED_FORMAL_CHARGES)}

# Deterministic scalar strengths used when the model receives edge features.
# They keep the edge-aware message passing simple: no attention, kernels, or
# extra learned parameters are introduced.
BOND_STRENGTHS = torch.tensor([1.0, 2.0, 3.0, 1.5], dtype=torch.float32)

ATOM_FEATURE_DIM = len(SUPPORTED_ATOMS) + len(SUPPORTED_FORMAL_CHARGES) + 2
BOND_FEATURE_DIM = len(SUPPORTED_BONDS)


def _require_supported_atom(symbol: str) -> None:
    if symbol not in ATOM_TO_INDEX:
        raise ValueError(
            f"Unsupported atom type {symbol!r}. Supported atoms: {list(SUPPORTED_ATOMS)}"
        )


def _require_supported_bond(bond_type: str) -> None:
    if bond_type not in BOND_TO_INDEX:
        raise ValueError(
            f"Unsupported bond type {bond_type!r}. Supported bonds: {list(SUPPORTED_BONDS)}"
        )


def _require_supported_charge(formal_charge: int) -> None:
    if formal_charge not in FORMAL_CHARGE_TO_INDEX:
        raise ValueError(
            "Unsupported formal charge "
            f"{formal_charge!r}. Supported charges: {list(SUPPORTED_FORMAL_CHARGES)}"
        )


def encode_atom(
    symbol: str,
    *,
    formal_charge: int = 0,
    aromatic: bool = False,
    degree: int = 0,
) -> torch.Tensor:
    """
    Encode one atom as a fixed-size feature vector.

    Layout:
        [ atom_type one-hot over C,H,N,O,S,F,Cl,Br,
          formal_charge one-hot over -1,0,+1,
          aromatic flag,
          atom degree ]

    The degree is supplied by graph construction from the explicit bond list.
    """
    _require_supported_atom(symbol)
    _require_supported_charge(int(formal_charge))
    if degree < 0:
        raise ValueError(f"Atom degree must be non-negative, got {degree!r}.")

    vec = torch.zeros(ATOM_FEATURE_DIM, dtype=torch.float32)
    vec[ATOM_TO_INDEX[symbol]] = 1.0

    charge_offset = len(SUPPORTED_ATOMS)
    vec[charge_offset + FORMAL_CHARGE_TO_INDEX[int(formal_charge)]] = 1.0

    aromatic_offset = charge_offset + len(SUPPORTED_FORMAL_CHARGES)
    vec[aromatic_offset] = 1.0 if aromatic else 0.0
    vec[aromatic_offset + 1] = float(degree)
    return vec


def build_molecule_graph(
    graph_id: str,
    atoms: Iterable[dict | str],
    bonds: Iterable[tuple[int, int, str]],
    label: int,
) -> dict:
    """
    Build a deterministic molecule graph dictionary for the demo pipeline.

    Parameters
    ----------
    graph_id : stable molecule/example identifier
    atoms    : iterable of either symbols ("C") or dictionaries with keys
               symbol, formal_charge, aromatic
    bonds    : iterable of (i, j, bond_type) tuples using zero-based indices
    label    : demo task label, e.g. 1 if a molecule contains a hetero atom

    Returns
    -------
    dict containing x, adj, edge_attr, label, graph_id, and chemistry metadata.
    """
    atom_specs: list[dict] = []
    for atom in atoms:
        spec = {"symbol": atom} if isinstance(atom, str) else dict(atom)
        symbol = spec.get("symbol")
        formal_charge = int(spec.get("formal_charge", 0))
        aromatic = bool(spec.get("aromatic", False))
        _require_supported_atom(symbol)
        _require_supported_charge(formal_charge)
        atom_specs.append(
            {
                "symbol": symbol,
                "formal_charge": formal_charge,
                "aromatic": aromatic,
            }
        )

    n = len(atom_specs)
    if n == 0:
        raise ValueError("Molecule graph must contain at least one atom.")

    adj = torch.zeros(n, n, dtype=torch.float32)
    edge_attr = torch.zeros(n, n, BOND_FEATURE_DIM, dtype=torch.float32)
    degrees = [0 for _ in range(n)]
    bond_records: list[tuple[int, int, str]] = []

    for i, j, bond_type in bonds:
        i = int(i)
        j = int(j)
        _require_supported_bond(bond_type)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f"Bond ({i}, {j}, {bond_type!r}) references an atom outside 0..{n - 1}."
            )
        if i == j:
            raise ValueError("Self-bonds are not supported in this toy chemistry subset.")
        if adj[i, j] != 0:
            raise ValueError(f"Duplicate bond between atoms {i} and {j}.")

        idx = BOND_TO_INDEX[bond_type]
        adj[i, j] = adj[j, i] = 1.0
        edge_attr[i, j, idx] = 1.0
        edge_attr[j, i, idx] = 1.0
        degrees[i] += 1
        degrees[j] += 1
        bond_records.append((i, j, bond_type))

    x = torch.stack(
        [
            encode_atom(
                spec["symbol"],
                formal_charge=spec["formal_charge"],
                aromatic=spec["aromatic"],
                degree=degrees[i],
            )
            for i, spec in enumerate(atom_specs)
        ],
        dim=0,
    )

    return {
        "graph_id": graph_id,
        "x": x,
        "adj": adj,
        "edge_attr": edge_attr,
        "label": int(label),
        "type": "molecule",
        "n": n,
        "task": "contains_hetero_atom",
        "atoms": atom_specs,
        "bonds": bond_records,
        "feature_dim": ATOM_FEATURE_DIM,
        "edge_feature_dim": BOND_FEATURE_DIM,
    }


def toy_molecule_dataset() -> list[dict]:
    """
    Return a deterministic toy molecule dataset.

    Demo label: 1 when the molecule contains a hetero atom from
    {N, O, S, F, Cl, Br}; otherwise 0. The examples intentionally cover the
    restricted atom and bond vocabularies without attempting chemical breadth.
    """
    return [
        build_molecule_graph(
            "methane", ["C", "H", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single"), (0, 4, "single")],
            label=0,
        ),
        build_molecule_graph(
            "ethene", ["C", "C", "H", "H", "H", "H"],
            [(0, 1, "double"), (0, 2, "single"), (0, 3, "single"), (1, 4, "single"), (1, 5, "single")],
            label=0,
        ),
        build_molecule_graph(
            "ethyne", ["C", "C", "H", "H"],
            [(0, 1, "triple"), (0, 2, "single"), (1, 3, "single")],
            label=0,
        ),
        build_molecule_graph(
            "benzene", [{"symbol": "C", "aromatic": True} for _ in range(6)],
            [(0, 1, "aromatic"), (1, 2, "aromatic"), (2, 3, "aromatic"),
             (3, 4, "aromatic"), (4, 5, "aromatic"), (5, 0, "aromatic")],
            label=0,
        ),
        build_molecule_graph(
            "water", ["O", "H", "H"],
            [(0, 1, "single"), (0, 2, "single")],
            label=1,
        ),
        build_molecule_graph(
            "ammonia", ["N", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single")],
            label=1,
        ),
        build_molecule_graph(
            "ethanol", ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            [(0, 1, "single"), (1, 2, "single"), (0, 3, "single"), (0, 4, "single"),
             (0, 5, "single"), (1, 6, "single"), (1, 7, "single"), (2, 8, "single")],
            label=1,
        ),
        build_molecule_graph(
            "thiol", ["C", "S", "H", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single"),
             (0, 4, "single"), (1, 5, "single")],
            label=1,
        ),
        build_molecule_graph(
            "fluoromethane", ["C", "F", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single"), (0, 4, "single")],
            label=1,
        ),
        build_molecule_graph(
            "chloromethane", ["C", "Cl", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single"), (0, 4, "single")],
            label=1,
        ),
        build_molecule_graph(
            "bromomethane", ["C", "Br", "H", "H", "H"],
            [(0, 1, "single"), (0, 2, "single"), (0, 3, "single"), (0, 4, "single")],
            label=1,
        ),
    ]
