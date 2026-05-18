#!/usr/bin/env python3
"""Run the repurposing pipeline with the first N rows from a DrugBank-like CSV.

Usage:
    python run_first5_drugbank.py
    python run_first5_drugbank.py drugbank_smiles_con_id.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repurposing_pipeline.pipeline import run_pipeline


def _pick_smiles(row: dict[str, str]) -> str:
    return (row.get("smiles") or row.get("SMILES") or "").strip()


def _pick_base_id(row: dict[str, str], index_1_based: int) -> str:
    base = (
        row.get("molecule_id")
        or row.get("drugbank_id")
        or row.get("id")
        or row.get("ID")
        or ""
    ).strip()
    if base:
        return base
    return f"row_{index_1_based:05d}"


def build_subset_csv(input_csv: Path, output_csv: Path, limit: int) -> int:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader, start=1):
            smiles = _pick_smiles(row)
            if not smiles:
                continue

            base_id = _pick_base_id(row, idx)
            # Keep molecule_id unique per row to avoid collisions in checkpoints/logs.
            molecule_id = f"{base_id}_{idx:05d}"
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "external_id": base_id,
                }
            )
            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError("No valid rows with SMILES were found in the input CSV")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles", "external_id"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-N DrugBank molecules through pipeline")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="drugbank_smiles_con_id.csv",
        help="Input CSV path (default: drugbank_smiles_con_id.csv)",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of rows to run (default: 5)")
    parser.add_argument("--run-id", default="hpc_first5_001", help="Run ID under runs root")
    parser.add_argument("--runs-root", default="runs", help="Runs root directory")
    parser.add_argument(
        "--no-boltz",
        action="store_true",
        help="Disable Boltz execution (for dry run checks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    runs_root = Path(args.runs_root).resolve()
    run_root = runs_root / args.run_id
    subset_csv = run_root / f"input_first_{args.n}.csv"

    selected = build_subset_csv(input_csv=input_csv, output_csv=subset_csv, limit=args.n)

    final_csv = run_pipeline(
        input_csv=subset_csv,
        receptor_path=None,
        runs_root=runs_root,
        run_id=args.run_id,
        run_boltz=not args.no_boltz,
    )

    print(f"Input source : {input_csv}")
    print(f"Subset CSV   : {subset_csv}")
    print(f"Rows selected: {selected}")
    print(f"Final output : {final_csv}")


if __name__ == "__main__":
    main()
