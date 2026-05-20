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
from repurposing_pipeline.wrappers.affinity_template_wrapper import build_affinity_template_from_pdb
from repurposing_pipeline.wrappers.boltz_wrapper import run_boltz_with_existing_wrapper
from repurposing_pipeline.wrappers.vina_wrapper import run_vina_parallel


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear percentile interpolation for q in [0,1]."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


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


def _normalize_minmax(values: list[float]) -> list[float]:
    """Min-max normalize values into [0, 1]."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    scale = vmax - vmin
    return [(value - vmin) / scale for value in values]


def _write_multiplicative_ranking_csv(results: list[dict[str, Any]], output_path: Path) -> Path | None:
    """Write multiplicative ranking CSV based on normalized D, P and B metrics."""
    eligible_rows: list[dict[str, Any]] = []
    for row in results:
        if row.get("docking_status") != "completed":
            continue
        if row.get("boltz_status") != "completed":
            continue

        vina_score = row.get("vina_score")
        affinity_pred = row.get("boltz_affinity_pred_value")
        affinity_prob = row.get("boltz_affinity_probability_binary")
        if vina_score is None or affinity_pred is None or affinity_prob is None:
            continue

        try:
            docking_strength = -float(vina_score)
            pred_value = float(affinity_pred)
            binder_prob = float(affinity_prob)
        except (TypeError, ValueError):
            continue

        pchembl_kcal = (6.0 - pred_value) * 1.364
        enriched = dict(row)
        enriched["D_docking_strength"] = docking_strength
        enriched["P_pChEMBL_kcal"] = pchembl_kcal
        enriched["B_binder_probability"] = binder_prob
        eligible_rows.append(enriched)

    if not eligible_rows:
        return None

    d_norm = _normalize_minmax([float(row["D_docking_strength"]) for row in eligible_rows])
    p_norm = _normalize_minmax([float(row["P_pChEMBL_kcal"]) for row in eligible_rows])
    b_norm = _normalize_minmax([float(row["B_binder_probability"]) for row in eligible_rows])

    for idx, row in enumerate(eligible_rows):
        row["D_norm"] = d_norm[idx]
        row["P_norm"] = p_norm[idx]
        row["B_norm"] = b_norm[idx]
        row["multiplicative_score"] = row["D_norm"] * row["P_norm"] * row["B_norm"]

    eligible_rows.sort(key=lambda item: float(item["multiplicative_score"]), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule_id",
        "drugbank_id",
        "smiles",
        "vina_score",
        "boltz_affinity_pred_value",
        "boltz_affinity_probability_binary",
        "D_docking_strength",
        "P_pChEMBL_kcal",
        "B_binder_probability",
        "D_norm",
        "P_norm",
        "B_norm",
        "multiplicative_score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in eligible_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    return output_path


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
    vina_hard_timeout: bool = False,
    vina_embed_seed: int = 42,
    vina_seed: int = 12345,
    run_boltz: bool = False,
    boltz_conda_env: str | None = None,
    boltz_python_executable: str | None = None,
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
                hard_timeout=vina_hard_timeout,
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
    completed_scores_with_id = [
        (molecule_id, float(item.get("vina_score")))
        for molecule_id, item in vina_by_id.items()
        if item.get("docking_status") == "completed" and item.get("vina_score") is not None
    ]
    completed_scores = [score for _, score in completed_scores_with_id]
    docking_median: float | None = median(completed_scores) if completed_scores else None
    quartile_threshold: float | None = None

    boltz_selected_ids: set[str] | None = None
    if completed_scores and docking_median is not None:
        sorted_scores = sorted(completed_scores)
        # For docking scores, lower (more negative) is typically better.
        if docking_median < 0:
            quartile_threshold = _percentile(sorted_scores, 0.25)
            boltz_selected_ids = {
                molecule_id
                for molecule_id, score in completed_scores_with_id
                if score <= quartile_threshold
            }
        else:
            quartile_threshold = _percentile(sorted_scores, 0.75)
            boltz_selected_ids = {
                molecule_id
                for molecule_id, score in completed_scores_with_id
                if score >= quartile_threshold
            }

    if boltz_selected_ids is not None and boltz_selected_ids:
        print("Vina completed. Median docking score:", f"{docking_median:.4f}")
        print("Selected last quartile molecules for Boltz:", len(boltz_selected_ids))
        selected_rows = [
            row for row in prepared_rows if row["molecule_id"] in boltz_selected_ids
        ]
        selected_rows.sort(key=lambda row: float(vina_by_id[row["molecule_id"]]["vina_score"]))
        for row in selected_rows:
            mol_id = row["molecule_id"]
            score = float(vina_by_id[mol_id]["vina_score"])
            dbid = row.get("drugbank_id") or row.get("external_id") or ""
            label = f"{mol_id} ({dbid})" if dbid else mol_id
            print(f"  - {label}: vina_score={score:.4f}")

    boltz_template_path: Path | None = None
    if run_boltz and boltz_selected_ids is not None and boltz_selected_ids:
        receptor_source: Path | None = None
        receptor_pdb_raw = (docking_setup or {}).get("receptor_pdb") if docking_setup else None
        if receptor_pdb_raw:
            candidate = Path(str(receptor_pdb_raw))
            if candidate.exists():
                receptor_source = candidate

        # Fallback: derive contacts/sequence from the receptor PDBQT if raw PDB is not provided.
        if receptor_source is None and receptor_path is not None and receptor_path.exists():
            receptor_source = receptor_path

        if receptor_source is not None:
            boltz_template_path = Path(__file__).resolve().parents[2] / "examples" / "affinity.yaml"
            summary = build_affinity_template_from_pdb(
                pdb_path=receptor_source,
                output_yaml=boltz_template_path,
                center=tuple(float(x) for x in docking_setup["box_center"]),
                radius=8.0,
                chain_id="A",
            )
            print(
                "Boltz template overwritten:",
                str(boltz_template_path),
                f"(source={receptor_source.name}, contacts={summary['contacts_count']}, seq_len={summary['sequence_length']})",
            )
        else:
            print("WARNING: no receptor file available to rebuild affinity template; using existing template.")

    if quartile_threshold is not None:
        pass
    elif docking_median is not None:
        boltz_selected_ids = set()

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
                    "last_quartile_threshold": (
                        quartile_threshold if quartile_threshold is not None else ""
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
                    "last_quartile_threshold",
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
                "error": "Filtered out by Vina last quartile threshold",
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
                template_path=boltz_template_path,
                boltz_conda_env=boltz_conda_env,
                boltz_python_executable=boltz_python_executable,
            )
            boltz_by_id[molecule_id] = boltz_result

        merged = {**base_result, **boltz_result}
        if merged.get("docking_status") == "failed":
            merged["status"] = "failed"
            merged["error"] = merged.get("vina_error", "Vina stage failed")
        if merged.get("boltz_status") == "filtered_out_by_vina":
            merged["status"] = "filtered"
            merged["error"] = "Did not pass Vina last quartile threshold for Boltz"
        if merged.get("boltz_status") == "failed":
            merged["status"] = "failed"
            merged["error"] = merged.get("error", "Boltz stage failed")
        results.append(merged)

    save_checkpoint(run_paths.checkpoints, "boltz", {"by_molecule": boltz_by_id})

    ranked = apply_ranking(results, method="separate")
    final_csv = write_final_results(ranked, run_paths.output / "final_results.csv")

    multiplicative_csv = _write_multiplicative_ranking_csv(
        results=ranked,
        output_path=run_paths.output / "final_multiplicative_ranked.csv",
    )
    if multiplicative_csv is not None:
        print("Multiplicative ranking CSV:", multiplicative_csv)

    logger.info("Run completed output_csv=%s receptor=%s", final_csv, receptor_path)
    return final_csv
