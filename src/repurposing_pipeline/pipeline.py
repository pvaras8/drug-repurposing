"""Main orchestration for the repurposing pipeline.

Vina docking is intentionally not implemented in this phase.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

from repurposing_pipeline.io import (
    ensure_run_paths,
    load_checkpoint,
    read_input_csv,
    save_checkpoint,
    write_final_results,
)
from repurposing_pipeline.ligand_prep import prepare_ligands
from repurposing_pipeline.ranking import apply_ranking
from repurposing_pipeline.wrappers.boltz_wrapper import run_boltz_with_existing_wrapper


def _build_run_logger(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("repurposing_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_path = logs_dir / "run.log"
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def run_pipeline(
    input_csv: Path,
    receptor_path: Path | None,
    runs_root: Path,
    run_id: str,
    run_boltz: bool = False,
) -> Path:
    """Execute the Phase 1 pipeline and write final_results.csv."""
    run_paths = ensure_run_paths(runs_root, run_id)
    logger = _build_run_logger(run_paths.logs)
    logger.info("Starting run_id=%s input_csv=%s", run_id, input_csv)

    rows = read_input_csv(input_csv)
    logger.info("Loaded %s valid input rows", len(rows))

    prep_checkpoint = load_checkpoint(run_paths.checkpoints, "ligand_prep")
    if prep_checkpoint:
        prepared_rows = prep_checkpoint.get("rows", [])
        logger.info("Loaded ligand prep checkpoint with %s rows", len(prepared_rows))
    else:
        prepared_rows = prepare_ligands(rows, run_paths.prepared)
        save_checkpoint(run_paths.checkpoints, "ligand_prep", {"rows": prepared_rows})
        logger.info("Ligand preparation completed")

    results: list[dict[str, Any]] = []
    boltz_checkpoint = load_checkpoint(run_paths.checkpoints, "boltz")
    boltz_by_id = boltz_checkpoint.get("by_molecule", {}) if boltz_checkpoint else {}

    for row in prepared_rows:
        molecule_id = row["molecule_id"]
        input_smiles = row["smiles"]
        canonical_smiles = row.get("canonical_smiles")
        base_result: dict[str, Any] = {
            "molecule_id": molecule_id,
            "smiles": input_smiles,
            "canonical_smiles": canonical_smiles,
            "ligand_prep_status": row.get("ligand_prep_status"),
            "ligand_prep_error": row.get("ligand_prep_error"),
            "docking_status": "pending_phase_2",
            "status": "completed",
            "error": "",
        }

        if row.get("ligand_prep_status") != "completed":
            base_result["status"] = "failed"
            base_result["error"] = row.get("ligand_prep_error") or "Ligand prep failed"
            results.append(base_result)
            continue

        # TODO: Vina integration lives in Phase 2.
        if molecule_id in boltz_by_id:
            boltz_result = boltz_by_id[molecule_id]
        else:
            boltz_result = run_boltz_with_existing_wrapper(
                repo_root=Path(__file__).resolve().parents[2],
                molecule_id=molecule_id,
                smiles=input_smiles,
                run_boltz=run_boltz,
                logs_dir=run_paths.logs,
                boltz_results_dir=run_paths.boltz_results,
            )
            boltz_by_id[molecule_id] = boltz_result

        merged = {**base_result, **boltz_result}
        if merged.get("boltz_status") == "failed":
            merged["status"] = "failed"
            merged["error"] = merged.get("error", "Boltz stage failed")
        results.append(merged)

    save_checkpoint(run_paths.checkpoints, "boltz", {"by_molecule": boltz_by_id})

    ranked = apply_ranking(results, method="separate")
    final_csv = write_final_results(ranked, run_paths.output / "final_results.csv")

    logger.info("Run completed output_csv=%s receptor=%s", final_csv, receptor_path)
    return final_csv
