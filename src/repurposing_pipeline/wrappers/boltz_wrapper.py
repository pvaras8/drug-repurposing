"""Wrapper to integrate existing boltz.py script without modifying src/boltz."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
from typing import Any

from repurposing_pipeline.parsers.boltz_parser import parse_boltz_metrics, parse_boltz_metrics_from_csv


def run_boltz_with_existing_wrapper(
    repo_root: Path,
    molecule_id: str,
    smiles: str,
    run_boltz: bool,
    logs_dir: Path,
    boltz_results_dir: Path,
) -> dict[str, Any]:
    """Run Boltz via the repository-level boltz.py wrapper.

    This adapter runs boltz.py per molecule through environment variables and
    parses returned metrics into pipeline-normalized fields.
    """
    log_path = logs_dir / f"{molecule_id}.log"
    molecule_out_dir = boltz_results_dir / molecule_id
    molecule_out_dir.mkdir(parents=True, exist_ok=True)

    if not run_boltz:
        return {
            "boltz_status": "skipped",
            "boltz_note": "Boltz execution disabled in this run",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    boltz_script = repo_root / "boltz.py"
    if not boltz_script.exists():
        return {
            "boltz_status": "failed",
            "error": f"boltz.py not found at {boltz_script}",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    output_csv_path = molecule_out_dir / "boltz_output.csv"
    output_results_dir = molecule_out_dir / "boltz_results_affinity_tmp"
    output_json_path = output_results_dir / "predictions/affinity_tmp/affinity_affinity_tmp.json"
    env = os.environ.copy()
    env["BOLTZ_SINGLE_SMILES"] = smiles
    env["BOLTZ_OUTPUT_CSV"] = str(output_csv_path)
    env["BOLTZ_JSON_PATH"] = str(output_json_path)
    env["BOLTZ_RESULTS_DIR"] = str(output_results_dir)
    env["BOLTZ_TMP_YAML"] = str(molecule_out_dir / "affinity_tmp.yaml")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("[BOLTZ] Attempting repository wrapper execution via boltz.py\n")
        log_file.write(f"[BOLTZ] molecule_id={molecule_id} smiles={smiles}\n")
        process = subprocess.run(
            [sys.executable, str(boltz_script)],
            cwd=str(repo_root),
            stdout=log_file,
            stderr=log_file,
            check=False,
            text=True,
            env=env,
        )

    if process.returncode != 0:
        return {
            "boltz_status": "failed",
            "error": f"boltz.py exited with code {process.returncode}",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    metrics: dict[str, Any]
    try:
        metrics = parse_boltz_metrics(output_json_path)
    except Exception:
        try:
            metrics = parse_boltz_metrics_from_csv(output_csv_path, smiles=smiles)
        except Exception as exc:
            return {
                "boltz_status": "failed",
                "error": f"boltz output parse failed: {exc}",
                "boltz_log": str(log_path),
                "boltz_output_dir": str(molecule_out_dir),
            }

    return {
        **metrics,
        "boltz_status": "completed",
        "boltz_note": "boltz.py executed and metrics parsed",
        "boltz_log": str(log_path),
        "boltz_output_dir": str(molecule_out_dir),
    }
