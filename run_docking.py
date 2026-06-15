#!/usr/bin/env python3
"""Run docking + downstream pipeline without the Click CLI entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repurposing_pipeline.io import ensure_run_paths
from repurposing_pipeline.pipeline import run_pipeline
from repurposing_pipeline.wrappers.meeko_wrapper import (
    parse_triplet,
    receptor_seems_protonated,
    validate_box,
)


def _load_vina_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid config format in {path}: expected JSON object")
    return loaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vina docking pipeline")
    parser.add_argument(
        "--input-csv",
        default=str(REPO_ROOT / "input_csvs/drugbank_smiles_con_id.csv"),
        help="Input CSV with at least molecule_id, smiles columns",
    )
    parser.add_argument(
        "--receptor-ready-pdbqt",
        required=True,
        help="Prepared receptor PDBQT path (required)",
    )
    parser.add_argument(
        "--receptor-pdb",
        default="",
        help="Optional original receptor PDB path (if omitted, PDBQT is used to rebuild Boltz template)",
    )
    parser.add_argument(
        "--pocket-center",
        default="-27.66,51.89,20.25",
        help="Pocket center x,y,z",
    )
    parser.add_argument(
        "--pocket-size",
        default="20,20,20",
        help="Pocket size x,y,z",
    )
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"), help="Runs root")
    parser.add_argument("--run-id", default="run_001", help="Run id")
    parser.add_argument(
        "--vina-config",
        default=str(REPO_ROOT / "config/vina.json"),
        help="Vina JSON config path",
    )
    parser.add_argument("--run-boltz", action="store_true", help="Run boltz stage")
    parser.add_argument(
        "--boltz-conda-env",
        default="",
        help="Conda env name where Boltz should run (e.g., boltz2)",
    )
    parser.add_argument(
        "--boltz-python-executable",
        default="",
        help="Absolute Python path for Boltz env (overrides --boltz-conda-env)",
    )
    parser.add_argument(
        "--allow-non-protonated-receptor",
        action="store_true",
        help="Continue even if receptor appears weakly protonated",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive confirmation prompts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N valid input rows (0 means all)",
    )
    return parser.parse_args()


def _build_limited_csv(input_csv: Path, output_csv: Path, limit: int) -> Path:
    """Write a subset CSV containing the first N valid rows by SMILES presence."""
    if limit <= 0:
        return input_csv

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        fieldnames = list(reader.fieldnames)
        smiles_key = "smiles" if "smiles" in fieldnames else "SMILES" if "SMILES" in fieldnames else None
        if smiles_key is None:
            raise ValueError("Input CSV missing SMILES/smiles column")

        rows: list[dict[str, str]] = []
        for row in reader:
            smiles = (row.get(smiles_key) or "").strip()
            if not smiles:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError("No valid rows found while building limited input CSV")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def main() -> None:
    args = _parse_args()

    input_csv = Path(args.input_csv).resolve()
    receptor_path = Path(args.receptor_ready_pdbqt).resolve()
    receptor_pdb_path = Path(args.receptor_pdb).resolve() if args.receptor_pdb else None
    runs_root = Path(args.runs_root).resolve()
    vina_config_path = Path(args.vina_config).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not receptor_path.exists():
        raise FileNotFoundError(f"Receptor PDBQT not found: {receptor_path}")
    if receptor_pdb_path is not None and not receptor_pdb_path.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb_path}")

    center = parse_triplet(args.pocket_center)
    box_size = parse_triplet(args.pocket_size)
    validate_box(center, box_size)

    protonated, atom_count, hydrogen_count, hydrogen_ratio = receptor_seems_protonated(receptor_path)
    if not protonated:
        print(
            "WARNING: receptor seems weakly protonated "
            f"(atoms={atom_count}, H={hydrogen_count}, ratio={hydrogen_ratio:.3f})."
        )
        print("Se recomienda protonar antes de docking.")
        if not args.allow_non_protonated_receptor:
            if args.no_prompt:
                raise RuntimeError(
                    "Receptor may be non-protonated. Use --allow-non-protonated-receptor to continue"
                )
            reply = input("Quieres continuar de todas formas? [y/N]: ").strip().lower()
            if reply not in {"y", "yes", "s", "si"}:
                raise RuntimeError("Docking cancelled by user")

    vina_cfg = _load_vina_config(vina_config_path)

    run_vina = bool(vina_cfg.get("run_vina", True))
    vina_num_processors = int(vina_cfg.get("num_processors", -1))
    vina_cpu_per_job = int(vina_cfg.get("vina_cpu_per_job", 1))
    vina_exhaustiveness = int(vina_cfg.get("exhaustiveness", 16))
    vina_n_poses = int(vina_cfg.get("n_poses", 10))
    vina_write_n_poses = int(vina_cfg.get("write_n_poses", 10))
    vina_energy_range = float(vina_cfg.get("energy_range", 3.0))
    vina_fallback_score = float(vina_cfg.get("fallback_score", -1.0))
    vina_timeout_seconds = int(vina_cfg.get("timeout_seconds", 300))
    vina_max_mw = float(vina_cfg.get("max_mw", 600.0))
    vina_sf_name = str(vina_cfg.get("sf_name", "vina"))
    vina_embed_seed = int(vina_cfg.get("embed_seed", 42))
    vina_seed = int(vina_cfg.get("vina_seed", 12345))
    vina_save_every = int(vina_cfg.get("save_every", 25))
    boltz_conda_env = str(vina_cfg.get("boltz_conda_env", "")).strip()
    boltz_python_executable = str(vina_cfg.get("boltz_python_executable", "")).strip()

    # CLI arguments override config when provided.
    if args.boltz_conda_env.strip():
        boltz_conda_env = args.boltz_conda_env.strip()
    if args.boltz_python_executable.strip():
        boltz_python_executable = args.boltz_python_executable.strip()

    run_paths = ensure_run_paths(runs_root, args.run_id)

    input_for_run = input_csv
    if args.limit > 0:
        subset_csv = run_paths.root / f"input_first_{args.limit}.csv"
        input_for_run = _build_limited_csv(input_csv=input_csv, output_csv=subset_csv, limit=args.limit)
        print(f"Using limited input CSV: {input_for_run}")

    docking_setup = {
        "receptor_mode": "ready",
        "receptor_pdbqt": str(receptor_path),
        "receptor_pdb": str(receptor_pdb_path) if receptor_pdb_path is not None else "",
        "box_center": [center[0], center[1], center[2]],
        "box_size": [box_size[0], box_size[1], box_size[2]],
        "box_source": "manual",
        "selected_ligand": None,
        "protonated_check": {
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "hydrogen_ratio": hydrogen_ratio,
            "seems_protonated": protonated,
        },
    }

    setup_json = run_paths.output / "docking_setup.json"
    setup_json.write_text(json.dumps(docking_setup, indent=2), encoding="utf-8")

    final_csv = run_pipeline(
        input_csv=input_for_run,
        receptor_path=receptor_path,
        docking_setup=docking_setup,
        runs_root=runs_root,
        run_id=args.run_id,
        run_vina=run_vina,
        vina_num_processors=vina_num_processors,
        vina_cpu_per_job=vina_cpu_per_job,
        vina_exhaustiveness=vina_exhaustiveness,
        vina_n_poses=vina_n_poses,
        vina_write_n_poses=vina_write_n_poses,
        vina_energy_range=vina_energy_range,
        vina_fallback_score=vina_fallback_score,
        vina_timeout_seconds=vina_timeout_seconds,
        vina_max_mw=vina_max_mw,
        vina_sf_name=vina_sf_name,
        vina_embed_seed=vina_embed_seed,
        vina_seed=vina_seed,
        vina_save_every=vina_save_every,
        run_boltz=args.run_boltz,
        boltz_conda_env=(boltz_conda_env or None),
        boltz_python_executable=(boltz_python_executable or None),
    )

    print(f"Docking setup: {setup_json}")
    print(f"Pipeline finished. Results: {final_csv}")


if __name__ == "__main__":
    main()
