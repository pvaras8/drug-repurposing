"""Ranking helpers for repurposing results."""

from __future__ import annotations

from typing import Any


def apply_ranking(results: list[dict[str, Any]], method: str = "separate") -> list[dict[str, Any]]:
    """Apply ranking metadata without docking fusion in this phase.

    method='separate' ranks only by Boltz affinity when available.
    """
    if method != "separate":
        for row in results:
            row["ranking_note"] = "Only 'separate' ranking is active in Phase 1"
        return results

    scored = [row for row in results if row.get("boltz_affinity_pred_value") is not None]
    scored_sorted = sorted(scored, key=lambda row: row["boltz_affinity_pred_value"])

    rank_map = {row["molecule_id"]: index + 1 for index, row in enumerate(scored_sorted)}
    for row in results:
        row["boltz_rank"] = rank_map.get(row["molecule_id"])
        row["final_rank"] = row["boltz_rank"]
    return results
