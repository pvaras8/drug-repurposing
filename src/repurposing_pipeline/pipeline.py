"""Main orchestration for the repurposing pipeline."""

from __future__ import annotations

from pathlib import Path
import csv
import logging
from statistics import median
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
from repurposing_pipeline.wrappers.vina_wrapper import run_vina_parallel


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
    docking_setup: dict[str, Any] | None = None,
    run_vina: bool = True,
    vina_num_processors: int = -1,
    vina_cpu_per_job: int = 1,
    vina_exhaustiveness: int = 16,
    vina_n_poses: int = 10,
    vina_write_n_poses: int = 10,
    vina_energy_range: float = 3.0,
    vina_fallback_score: float = -1.0,
    vina_timeout_seconds: int = 300,
    vina_embed_seed: int = 42,
    vina_seed: int = 12345,
    run_boltz: bool = False,
) -> Path:
    """Execute the repurposing pipeline and write final_results.csv."""
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

    vina_checkpoint = load_checkpoint(run_paths.checkpoints, "vina")
    vina_by_id: dict[str, Any] = vina_checkpoint.get("by_molecule", {}) if vina_checkpoint else {}
    if run_vina and receptor_path is not None and docking_setup is not None:
        pending_rows: list[dict[str, Any]] = []
        for row in prepared_rows:
            if row.get("ligand_prep_status") != "completed":
                continue
            if row["molecule_id"] in vina_by_id:
                continue
            pending_rows.append(row)

        if pending_rows:
            logger.info("Starting Vina docking for %s molecules", len(pending_rows))
            vina_results = run_vina_parallel(
                rows=pending_rows,
                receptor_pdbqt=receptor_path,
                center=tuple(float(x) for x in docking_setup["box_center"]),
                box_size=tuple(float(x) for x in docking_setup["box_size"]),
                vina_results_dir=run_paths.vina_results,
                num_processors=vina_num_processors,
                vina_cpu_per_job=vina_cpu_per_job,
                exhaustiveness=vina_exhaustiveness,
                n_poses=vina_n_poses,
                write_n_poses=vina_write_n_poses,
                energy_range=vina_energy_range,
                fallback_score=vina_fallback_score,
                timeout_seconds=vina_timeout_seconds,
                embed_seed=vina_embed_seed,
                vina_seed=vina_seed,
            )
            vina_by_id.update(vina_results)
            save_checkpoint(run_paths.checkpoints, "vina", {"by_molecule": vina_by_id})
            logger.info("Vina docking completed")
    elif run_vina and receptor_path is None:
        logger.info("Skipping Vina docking: receptor not provided")
    elif run_vina and docking_setup is None:
        logger.info("Skipping Vina docking: docking setup metadata missing")
    completed_scores = [
        float(item.get("vina_score"))
        for item in vina_by_id.values()
        if item.get("docking_status") == "completed" and item.get("vina_score") is not None
    ]
    docking_median: float | None = median(completed_scores) if completed_scores else None
    docking_threshold: float | None = (
        (0.75 * docking_median) if docking_median is not None else None
    )

    boltz_selected_ids: set[str] | None = None
    if docking_threshold is not None:
        boltz_selected_ids = set()
        for molecule_id, docking_result in vina_by_id.items():
            score = docking_result.get("vina_score")
            if score is None:
                continue
            score_f = float(score)
            if docking_median is not None and docking_median < 0:
                if score_f <= docking_threshold:
                    boltz_selected_ids.add(molecule_id)
            elif score_f >= docking_threshold:
                boltz_selected_ids.add(molecule_id)

    if run_vina and receptor_path is not None and docking_setup is not None:
        vina_csv_path = run_paths.output / "vina_results.csv"
        rows_for_csv: list[dict[str, Any]] = []
        for row in prepared_rows:
            mol_id = row["molecule_id"]
            docking = vina_by_id.get(mol_id, {})
            rows_for_csv.append(
                {
                    "molecule_id": mol_id,
                    "drugbank_id": row.get("drugbank_id") or row.get("external_id") or "",
                    "smiles": row.get("smiles", ""),
                    "vina_score": docking.get("vina_score", ""),
                    "docking_status": docking.get("docking_status", "missing"),
                    "vina_error": docking.get("vina_error", ""),
                    "vina_ligand_pdbqt": docking.get("vina_ligand_pdbqt", ""),
                    "vina_pose_pdbqt": docking.get("vina_pose_pdbqt", ""),
                    "pass_to_boltz": bool(boltz_selected_ids and mol_id in boltz_selected_ids),
                    "median_vina_score": docking_median if docking_median is not None else "",
                    "threshold_3_over_4_median": (
                        docking_threshold if docking_threshold is not None else ""
                    ),
                }
            )

        with vina_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "molecule_id",
                    "drugbank_id",
                    "smiles",
                    "vina_score",
                    "docking_status",
                    "vina_error",
                    "vina_ligand_pdbqt",
                    "vina_pose_pdbqt",
                    "pass_to_boltz",
                    "median_vina_score",
                    "threshold_3_over_4_median",
                ],
            )
            writer.writeheader()
            writer.writerows(rows_for_csv)
        logger.info("Vina CSV written to %s", vina_csv_path)

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
            "docking_status": "skipped",
            "status": "completed",
            "error": "",
        }

        if row.get("ligand_prep_status") != "completed":
            base_result["status"] = "failed"
            base_result["error"] = row.get("ligand_prep_error") or "Ligand prep failed"
            results.append(base_result)
            continue

        docking_result = vina_by_id.get(molecule_id)
        if docking_result is not None:
            base_result.update(docking_result)
        elif run_vina and receptor_path is not None and docking_setup is not None:
            base_result["docking_status"] = "failed"
            base_result["vina_error"] = "Docking result missing"
            base_result["vina_score"] = vina_fallback_score
        elif run_vina and receptor_path is None:
            base_result["docking_status"] = "skipped_no_receptor"
        elif run_vina and docking_setup is None:
            base_result["docking_status"] = "skipped_no_box"

        pass_to_boltz = True
        if boltz_selected_ids is not None:
            pass_to_boltz = molecule_id in boltz_selected_ids

        if not pass_to_boltz:
            boltz_result = {
                "boltz_status": "filtered_out_by_vina",
                "error": "Filtered out by Vina median threshold",
            }
        elif molecule_id in boltz_by_id:
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
        if merged.get("docking_status") == "failed":
            merged["status"] = "failed"
            merged["error"] = merged.get("vina_error", "Vina stage failed")
        if merged.get("boltz_status") == "filtered_out_by_vina":
            merged["status"] = "filtered"
            merged["error"] = "Did not pass Vina threshold for Boltz"
        if merged.get("boltz_status") == "failed":
            merged["status"] = "failed"
            merged["error"] = merged.get("error", "Boltz stage failed")
        results.append(merged)

    save_checkpoint(run_paths.checkpoints, "boltz", {"by_molecule": boltz_by_id})

    ranked = apply_ranking(results, method="separate")
    final_csv = write_final_results(ranked, run_paths.output / "final_results.csv")

    logger.info("Run completed output_csv=%s receptor=%s", final_csv, receptor_path)
    return final_csv
