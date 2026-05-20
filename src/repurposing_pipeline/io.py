"""I/O helpers for the repurposing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Any


REQUIRED_COLUMNS = {"molecule_id", "smiles"}
EXCLUDED_OUTPUT_COLUMNS = {
    "log",
    "boltz_log",
    "boltz_note",
    "boltz_output_dir",
    "ligand_prep_error",
}


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for one pipeline run."""

    root: Path
    prepared: Path
    vina_results: Path
    boltz_results: Path
    output: Path
    logs: Path
    checkpoints: Path


def read_input_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read and validate molecule rows from the input CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        headers = set(reader.fieldnames)
        missing = REQUIRED_COLUMNS - headers
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Input CSV missing required columns: {missing_text}")

        rows: list[dict[str, str]] = []
        for row in reader:
            molecule_id = (row.get("molecule_id") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            if not molecule_id or not smiles:
                continue
            rows.append({key: (value or "").strip() for key, value in row.items()})

    if not rows:
        raise ValueError("Input CSV has no valid rows with molecule_id and smiles")
    return rows


def ensure_run_paths(runs_root: Path, run_id: str) -> RunPaths:
    """Create the run directory structure under runs/run_XXX."""
    root = runs_root / run_id
    prepared = root / "prepared"
    vina_results = root / "vina_results"
    boltz_results = root / "boltz_results"
    output = root / "output"
    logs = root / "logs"
    checkpoints = root / ".checkpoints"

    for path in (root, prepared, vina_results, boltz_results, output, logs, checkpoints):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        root=root,
        prepared=prepared,
        vina_results=vina_results,
        boltz_results=boltz_results,
        output=output,
        logs=logs,
        checkpoints=checkpoints,
    )


def write_final_results(results: list[dict[str, Any]], output_path: Path) -> Path:
    """Write a normalized final results CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        headers = ["molecule_id", "smiles", "status", "error"]
    else:
        header_set: set[str] = set()
        for result in results:
            header_set.update(result.keys())
        header_set -= EXCLUDED_OUTPUT_COLUMNS
        preferred = [
            "molecule_id",
            "smiles",
            "ligand_prep_status",
            "docking_status",
            "boltz_status",
            "status",
            "error",
            "boltz_affinity_pred_value",
            "boltz_affinity_probability_binary",
            "boltz_rank",
            "final_rank",
        ]
        rest = sorted(key for key in header_set if key not in preferred)
        headers = [key for key in preferred if key in header_set] + rest

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in headers})

    return output_path


def checkpoint_path(checkpoints_dir: Path, stage: str) -> Path:
    """Return checkpoint file path for a stage."""
    return checkpoints_dir / f"{stage}.json"


def load_checkpoint(checkpoints_dir: Path, stage: str) -> dict[str, Any]:
    """Load stage checkpoint state if it exists."""
    path = checkpoint_path(checkpoints_dir, stage)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(checkpoints_dir: Path, stage: str, data: dict[str, Any]) -> Path:
    """Persist stage checkpoint state."""
    path = checkpoint_path(checkpoints_dir, stage)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path
