"""
CommitStore — version-controlled registry of model weight snapshots.

Implements the projection  π : C → W  from GNNEquiv (Definition 7):
    π(commit_id)  →  weight state dict  ∈  W

Each commit records:
    - commit_id  : SHA-256 hex digest of the serialised state dict (first 16 chars)
    - parent_id  : previous HEAD commit (forms a directed commit chain)
    - message    : human-readable label (e.g. "epoch 10")
    - timestamp  : UTC ISO-8601
    - metadata   : arbitrary dict (e.g. {"epoch": 10, "loss": 0.42, "acc": 0.85})

Weights are stored as individual .pt files: commits/<commit_id>.pt
The commit chain is stored as commits/registry.json.
"""

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_state(state_dict: dict) -> str:
    """SHA-256 of the serialised state dict, truncated to 16 hex chars."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def _safe_load(path: Path) -> dict:
    """Load a state dict; handles both PyTorch < 2.4 and >= 2.4 APIs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# CommitStore
# ---------------------------------------------------------------------------

class CommitStore:
    """
    Version-controlled store for FunctorGNN weight snapshots.

    The projection  π : C → W  is implemented by `project(commit_id)`.
    `load_into(model, commit_id)` is the in-place variant used by inference.
    """

    _REGISTRY_FILE = "registry.json"
    _CONFIG_FILE   = "model_config.json"

    def __init__(self, store_dir: str = "commits"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._reg_path = self.store_dir / self._REGISTRY_FILE
        self._registry = self._load_registry()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        if self._reg_path.exists():
            with open(self._reg_path) as f:
                return json.load(f)
        return {"head": None, "commits": []}

    def _save_registry(self) -> None:
        with open(self._reg_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    # ------------------------------------------------------------------
    # Core commit API
    # ------------------------------------------------------------------

    def commit(
        self,
        model: torch.nn.Module,
        message: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Snapshot current model weights and append a commit record.

        Returns the commit_id (hash of the weight snapshot).
        Idempotent: if the same weights are committed twice the file is
        overwritten but the registry gets a new timestamped entry.
        """
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        commit_id = _hash_state(state_dict)

        weights_path = self.store_dir / f"{commit_id}.pt"
        torch.save(state_dict, weights_path)

        record = {
            "commit_id": commit_id,
            "parent_id": self._registry["head"],
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata":  metadata or {},
        }
        self._registry["commits"].append(record)
        self._registry["head"] = commit_id
        self._save_registry()
        return commit_id

    # ------------------------------------------------------------------
    # Projection  π : C → W
    # ------------------------------------------------------------------

    def project(self, commit_id: str) -> dict:
        """
        π(commit_id) → weight state dict.

        This is the formal projection from GNNEquiv Definition 7.
        Raises ValueError for unknown commit_ids.
        """
        path = self.store_dir / f"{commit_id}.pt"
        if not path.exists():
            raise ValueError(
                f"Commit {commit_id!r} not found in store at {self.store_dir}."
            )
        return _safe_load(path)

    def load_into(self, model: torch.nn.Module, commit_id: str) -> torch.nn.Module:
        """
        Apply π(commit_id) to model in-place.

        This is what inference.run_inference calls — it corresponds to
        Theorem 9: F̃(c, G, ω) = S(π(c), G, ω).
        """
        model.load_state_dict(self.project(commit_id))
        return model

    # ------------------------------------------------------------------
    # Commit introspection
    # ------------------------------------------------------------------

    def head(self) -> Optional[str]:
        """Return the current HEAD commit_id, or None if no commits exist."""
        return self._registry["head"]

    def list_commits(self) -> list:
        """Return all commit records in insertion order."""
        return list(self._registry["commits"])

    def get_commit(self, commit_id: str) -> Optional[dict]:
        for c in self._registry["commits"]:
            if c["commit_id"] == commit_id:
                return c
        return None

    def nth_commit(self, n: int) -> Optional[str]:
        """Return the commit_id at position n (0-indexed). None if out of range."""
        commits = self._registry["commits"]
        if 0 <= n < len(commits):
            return commits[n]["commit_id"]
        return None

    def count(self) -> int:
        return len(self._registry["commits"])

    # ------------------------------------------------------------------
    # Model config persistence
    # ------------------------------------------------------------------

    def save_model_config(self, config: dict) -> None:
        with open(self.store_dir / self._CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

    def load_model_config(self) -> dict:
        path = self.store_dir / self._CONFIG_FILE
        if not path.exists():
            raise FileNotFoundError(
                "model_config.json not found — run 'train' first."
            )
        with open(path) as f:
            return json.load(f)
