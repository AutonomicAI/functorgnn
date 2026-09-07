"""
Structured inference logger.

Each inference event is appended as a JSON line to logs/inference.jsonl.

Record schema:
    run_id         : str  — 8-char hex UUID for this inference call
    commit_id      : str  — 16-char weight snapshot hash (the commit used)
    seed           : int  — random seed (ω in GNNEquiv notation)
    graph_id       : str  — identifier of the input graph
    predicted_class/probs/logits for classification, or prediction/task for
    ESOL regression
    timestamp      : str  — UTC ISO-8601
"""

import json
from pathlib import Path

_LOG_DIR  = Path("logs")
_LOG_FILE = _LOG_DIR / "inference.jsonl"


def log_inference_event(record: dict) -> None:
    """Append a structured inference record to the JSONL log."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_log(tail: int = 50) -> list:
    """Return the last `tail` inference records from the log."""
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-tail:]]
