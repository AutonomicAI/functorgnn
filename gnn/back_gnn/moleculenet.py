"""
MoleculeNet benchmark adapters for the Functor GNN demo.

This module intentionally keeps benchmark-specific loading and RDKit parsing
outside dataset.py. It is a small first adapter for ESOL and BACE, not a full
cheminformatics platform.

Returned graph dictionaries match the existing Functor GNN pipeline shape:
    graph_id, adj, x, optional edge_attr, label, type, n, task
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import torch


SUPPORTED_KINDS = {"esol", "bace"}
BOND_TYPES = ("single", "double", "triple", "aromatic")
ESOL_CSV_CANDIDATES = (
    "esol.csv",
    "delaney-processed.csv",
    "esol/raw/delaney-processed.csv",
)
ESOL_SMILES_COLUMNS = ("smiles", "SMILES", "smiles_str")
ESOL_LABEL_COLUMNS = (
    "measured log solubility in mols per litre",
    "measured_log_solubility",
    "logS",
    "label",
)


def _require_rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading MoleculeNet ESOL/BACE requires RDKit. Install rdkit "
            "(for example, `conda install -c conda-forge rdkit`) and retry."
        ) from exc
    return Chem


def _require_deepchem():
    try:
        import deepchem as dc
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DeepChem import failed; this environment may also require TensorFlow. "
            "Prefer local CSV loading for ESOL."
        ) from exc
    return dc


def _require_rdkit_for_esol_csv():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ESOL CSV loading requires RDKit. Install with `pip install rdkit`."
        ) from exc
    return Chem


def _atom_features(atom: Any) -> torch.Tensor:
    """Return compact deterministic RDKit atom features."""
    hybridization = str(atom.GetHybridization()).upper()
    return torch.tensor(
        [
            float(atom.GetAtomicNum()),
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(hybridization == "SP"),
            float(hybridization == "SP2"),
            float(hybridization == "SP3"),
            float(atom.GetMass() * 0.01),
        ],
        dtype=torch.float32,
    )


def _bond_index(Chem: Any, bond: Any) -> int:
    bond_type = bond.GetBondType()
    if bond.GetIsAromatic():
        return 3
    if bond_type == Chem.BondType.SINGLE:
        return 0
    if bond_type == Chem.BondType.DOUBLE:
        return 1
    if bond_type == Chem.BondType.TRIPLE:
        return 2
    # Keep unsupported/rare bond classes deterministic rather than failing the
    # whole benchmark adapter. They remain connected but get an all-zero edge.
    return -1


def _smiles_to_graph(
    smiles: str,
    *,
    graph_id: str,
    label: int | float,
    task: str,
    graph_type: str = "molecule",
    Chem: Any | None = None,
) -> dict:
    Chem = Chem or _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {task} graph {graph_id!r}: {smiles!r}")

    n = int(mol.GetNumAtoms())
    if n == 0:
        raise ValueError(f"MoleculeNet graph {graph_id!r} has no atoms.")

    adj = torch.zeros(n, n, dtype=torch.float32)
    edge_attr = torch.zeros(n, n, len(BOND_TYPES), dtype=torch.float32)

    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        adj[i, j] = adj[j, i] = 1.0
        idx = _bond_index(Chem, bond)
        if idx >= 0:
            edge_attr[i, j, idx] = 1.0
            edge_attr[j, i, idx] = 1.0

    x = torch.stack([_atom_features(atom) for atom in mol.GetAtoms()], dim=0)
    return {
        "graph_id": graph_id,
        "adj": adj,
        "x": x,
        "edge_attr": edge_attr,
        "label": label,
        "type": graph_type,
        "n": n,
        "task": task,
    }


def _find_local_esol_csv(data_dir: str) -> Path | None:
    root = Path(data_dir)
    for rel in ESOL_CSV_CANDIDATES:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _select_column(fieldnames: Iterable[str] | None, accepted: tuple[str, ...], role: str) -> str:
    fields = list(fieldnames or [])
    for name in accepted:
        if name in fields:
            return name

    # Keep the advertised exact names as the primary contract, but accept
    # harmless capitalization/spacing differences in local CSV exports.
    normalized = {field.strip().lower(): field for field in fields}
    for name in accepted:
        match = normalized.get(name.strip().lower())
        if match is not None:
            return match

    raise ValueError(
        f"ESOL CSV is missing a {role} column. Accepted {role} columns: {list(accepted)}. "
        f"Available columns: {fields}"
    )


def _load_esol_csv(path: Path) -> list[dict]:
    Chem = _require_rdkit_for_esol_csv()
    graphs: list[dict] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        smiles_column = _select_column(reader.fieldnames, ESOL_SMILES_COLUMNS, "SMILES")
        label_column = _select_column(reader.fieldnames, ESOL_LABEL_COLUMNS, "label")

        for row_idx, row in enumerate(reader):
            smiles = (row.get(smiles_column) or "").strip()
            label_text = (row.get(label_column) or "").strip()
            if not smiles or not label_text:
                continue
            try:
                label = float(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ESOL label at {path}:{row_idx + 2}: {label_text!r}"
                ) from exc

            graphs.append(
                _smiles_to_graph(
                    smiles,
                    graph_id=f"esol_{len(graphs):05d}",
                    label=label,
                    task="esol",
                    graph_type="moleculenet",
                    Chem=Chem,
                )
            )

    return graphs


def _flatten_datasets(datasets: Any) -> Iterable[Any]:
    if datasets is None:
        return
    if hasattr(datasets, "itersamples"):
        yield datasets
        return
    for ds in datasets:
        if ds is not None:
            yield ds


def _load_deepchem_raw(kind: str, data_dir: str) -> tuple[list[Any], list[Any], str]:
    """
    Load raw MoleculeNet records through DeepChem.

    Returns (datasets, tasks, smiles_field_name). DeepChem's loader names have
    varied over time, so this function isolates those details from callers.
    """
    dc = _require_deepchem()

    if kind == "esol":
        loader = getattr(dc.molnet, "load_delaney", None) or getattr(dc.molnet, "load_esol", None)
        if loader is None:
            raise ImportError("DeepChem does not expose an ESOL/Delaney MoleculeNet loader.")
        tasks, datasets, _transformers = loader(
            featurizer="Raw", splitter=None, data_dir=data_dir, reload=True
        )
        return list(_flatten_datasets(datasets)), list(tasks), "smiles"

    if kind == "bace":
        loader = getattr(dc.molnet, "load_bace_classification", None) or getattr(dc.molnet, "load_bace", None)
        if loader is None:
            raise ImportError("DeepChem does not expose a BACE MoleculeNet loader.")
        tasks, datasets, _transformers = loader(
            featurizer="Raw", splitter=None, data_dir=data_dir, reload=True
        )
        return list(_flatten_datasets(datasets)), list(tasks), "smiles"

    raise ValueError(f"Unsupported MoleculeNet kind {kind!r}. Expected one of {sorted(SUPPORTED_KINDS)}.")


def _coerce_smiles(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    # DeepChem Raw featurizers typically produce strings, but keep a defensive
    # path for one-element numpy/object containers.
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def _iter_records(kind: str, data_dir: str) -> Iterable[tuple[str, int | float]]:
    datasets, _tasks, _smiles_field = _load_deepchem_raw(kind, data_dir)
    for dataset in datasets:
        for smiles, y, w, sample_id in dataset.itersamples():
            # Respect DeepChem weights when present; zero weight commonly marks
            # missing labels in benchmark tasks.
            if hasattr(w, "__len__") and len(w) and float(w[0]) == 0.0:
                continue
            label_value = y[0] if hasattr(y, "__len__") else y
            label = float(label_value) if kind == "esol" else int(label_value)

            # Raw DeepChem loaders generally put SMILES in X; some versions
            # keep the canonical SMILES in ids. Prefer X when string-like and
            # otherwise fall back to the sample id.
            candidate = smiles if isinstance(smiles, (str, bytes)) else sample_id
            yield _coerce_smiles(candidate), label


def load_moleculenet_dataset(kind: str, data_dir: str = "data") -> list[dict]:
    """
    Load an ESOL or BACE MoleculeNet dataset as Functor GNN graph dicts.

    Parameters
    ----------
    kind:
        "esol" for aqueous solubility regression labels or "bace" for BACE
        classification labels.
    data_dir:
        Directory used for local ESOL CSV files and by DeepChem for
        MoleculeNet downloads/cache files.

    Raises
    ------
    ImportError
        If DeepChem or RDKit is unavailable.
    ValueError
        If an unsupported kind is requested or a SMILES record cannot be parsed.
    """
    kind = kind.lower()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"kind must be one of {sorted(SUPPORTED_KINDS)} for MoleculeNet datasets.")

    if kind == "esol":
        local_csv = _find_local_esol_csv(data_dir)
        if local_csv is not None:
            return _load_esol_csv(local_csv)

    # No local ESOL CSV was found, or this is BACE. Fall back to the original
    # DeepChem path for benchmark loading.
    _require_deepchem()
    _require_rdkit()

    graphs: list[dict] = []
    for idx, (smiles, label) in enumerate(_iter_records(kind, data_dir)):
        graphs.append(
            _smiles_to_graph(
                smiles,
                graph_id=f"{kind}_{idx:05d}",
                label=label,
                task=kind,
            )
        )
    return graphs
