"""Protocol scoring for one or more Boltz targets."""

from __future__ import annotations

from math import isfinite, prod
from typing import Any


def score_boltz(row: dict[str, Any], *, prefix: str = "", min_probability: float = 0.45) -> float | None:
    """Return B × (6 − A) only for a passing Boltz prediction."""
    if row.get(f"{prefix}boltz_status") != "completed":
        return None
    try:
        affinity = float(row[f"{prefix}boltz_affinity_pred_value"])
        probability = float(row[f"{prefix}boltz_affinity_probability_binary"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (isfinite(affinity) and isfinite(probability)):
        return None
    if affinity >= 0 or probability < min_probability or not 0 <= probability <= 1:
        return None
    return probability * (6 - affinity)


def _pic50(row: dict[str, Any], prefix: str = "") -> float | None:
    """Convert Boltz affinity prediction A into pIC50 = 6 - A."""
    try:
        affinity = float(row[f"{prefix}boltz_affinity_pred_value"])
    except (KeyError, TypeError, ValueError):
        return None
    return 6 - affinity if isfinite(affinity) else None


def apply_ranking(results: list[dict[str, Any]], target_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Rank passing candidates by Boltz score or the geometric mean across targets."""
    names = target_names or []
    for row in results:
        if names:
            scores = [score_boltz(row, prefix=f"{name}_") for name in names]
            for name, score in zip(names, scores):
                row[f"{name}_pIC50"] = _pic50(row, prefix=f"{name}_")
                row[f"{name}_boltz_score"] = score
            row["final_score"] = prod(scores) ** (1 / len(scores)) if all(s is not None for s in scores) else None
        else:
            row["pIC50"] = _pic50(row)
            row["boltz_score"] = score_boltz(row)
            row["final_score"] = row["boltz_score"]
    ranked = sorted((row for row in results if row["final_score"] is not None), key=lambda row: (-row["final_score"], row["molecule_id"]))
    for rank, row in enumerate(ranked, 1):
        row["final_rank"] = rank
        row["boltz_rank"] = rank if not names else None
    for row in results:
        row.setdefault("final_rank", None)
    return sorted(results, key=lambda row: (row["final_rank"] is None, row["final_rank"] or 0, row["molecule_id"]))
