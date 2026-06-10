"""
Prediction Logger
=================
Appends each prediction (inputs + output + timestamp) to a JSONL log file.
Provides a simple in-process store for the /history endpoint.

This is intentionally lightweight — no database dependency.
In a production system you'd swap this for DynamoDB or CloudWatch Logs.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

LOG_FILE = os.environ.get("PREDICTION_LOG_PATH", "predictions.jsonl")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY_ENTRIES", "500"))

# Thread lock so concurrent requests don't corrupt the log file
_lock = threading.Lock()


def log_prediction(inputs: dict[str, Any], prediction: str, error: str | None = None) -> None:
    """
    Append one prediction record to the JSONL log file.

    Args:
        inputs:     Dict of feature names → raw input values from the request
        prediction: The model's predicted quality score as a string
        error:      Error message if the prediction failed, else None
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "prediction": prediction,
        "error": error,
    }

    with _lock:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Trim the file if it grows too large (keep last MAX_HISTORY lines)
            _trim_log_if_needed()
        except OSError as e:
            # Never crash the API because of a logging failure
            print(f"[logger] Warning: could not write to log — {e}")


def get_history(n: int = 20) -> list[dict]:
    """
    Return the last `n` prediction records in reverse-chronological order.

    Args:
        n: Number of recent records to return (default 20, max 100)

    Returns:
        List of prediction record dicts, newest first.
    """
    n = min(n, 100)

    with _lock:
        try:
            if not os.path.exists(LOG_FILE):
                return []

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()

            records = []
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if len(records) >= n:
                    break

            return records
        except OSError:
            return []


def get_stats() -> dict:
    """
    Compute simple summary statistics over all logged predictions.

    Returns:
        Dict with total_predictions, avg_score, min_score, max_score
    """
    with _lock:
        try:
            if not os.path.exists(LOG_FILE):
                return {"total_predictions": 0}

            scores = []
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("error") is None:
                            scores.append(float(rec["prediction"]))
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

            if not scores:
                return {"total_predictions": 0}

            return {
                "total_predictions": len(scores),
                "avg_score": round(sum(scores) / len(scores), 2),
                "min_score": round(min(scores), 2),
                "max_score": round(max(scores), 2),
            }
        except OSError:
            return {"total_predictions": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trim_log_if_needed() -> None:
    """Keep the log file from growing unboundedly. Trim to MAX_HISTORY lines."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) > MAX_HISTORY:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_HISTORY:])
    except OSError:
        pass
