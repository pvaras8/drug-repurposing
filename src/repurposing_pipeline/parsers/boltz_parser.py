"""Parsers for Boltz JSON outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any


def parse_boltz_metrics(json_path: Path) -> dict[str, Any]:
    """Extract affinity metrics from a Boltz output JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"Boltz JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "boltz_affinity_pred_value": data.get("affinity_pred_value"),
        "boltz_affinity_probability_binary": data.get("affinity_probability_binary"),
        "boltz_affinity_pred_value1": data.get("affinity_pred_value1"),
        "boltz_affinity_probability_binary1": data.get("affinity_probability_binary1"),
        "boltz_affinity_pred_value2": data.get("affinity_pred_value2"),
        "boltz_affinity_probability_binary2": data.get("affinity_probability_binary2"),
    }


def parse_boltz_metrics_from_csv(csv_path: Path, smiles: str) -> dict[str, Any]:
    """Extract affinity metrics from boltz.py CSV output for a specific SMILES."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Boltz CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in reversed(rows):
        if (row.get("SMILES") or "") != smiles:
            continue
        row_error = (row.get("error") or "").strip()
        if row_error:
            raise ValueError(f"Boltz reported molecule error: {row_error}")
        return {
            "boltz_affinity_pred_value": _to_float(row.get("affinity_pred_value")),
            "boltz_affinity_probability_binary": _to_float(row.get("affinity_probability_binary")),
            "boltz_affinity_pred_value1": _to_float(row.get("affinity_pred_value1")),
            "boltz_affinity_probability_binary1": _to_float(row.get("affinity_probability_binary1")),
            "boltz_affinity_pred_value2": _to_float(row.get("affinity_pred_value2")),
            "boltz_affinity_probability_binary2": _to_float(row.get("affinity_probability_binary2")),
        }

    raise ValueError(f"No Boltz CSV row found for SMILES: {smiles}")


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
