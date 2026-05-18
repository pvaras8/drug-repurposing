#!/usr/bin/env python

import csv
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import pandas as pd
import yaml


DEFAULT_TEMPLATE = Path("/LUSTRE/users/pvaras/boltz/examples/affinity.yaml")
DEFAULT_SCRATCH_ROOT = Path("/LUSTRE/users/pvaras/boltz/")
DEFAULT_TMP_YAML = Path("/tmp/affinity_tmp.yaml")
DEFAULT_RESULTS_DIR = DEFAULT_SCRATCH_ROOT / "boltz_results_affinity_tmp"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "predictions/affinity_tmp/affinity_affinity_tmp.json"
DEFAULT_CSV_PATH = Path("boltz_results_bace_de_novo.csv")
DEFAULT_INPUT_CSV = Path("molecules_filtered_strict.csv")


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _read_smiles_list() -> list[str]:
    single_smiles = os.getenv("BOLTZ_SINGLE_SMILES")
    if single_smiles:
        return [single_smiles]

    input_csv = _env_path("BOLTZ_INPUT_CSV", DEFAULT_INPUT_CSV)
    df = pd.read_csv(input_csv)
    return df["SMILES"].dropna().astype(str).tolist()


def _ensure_csv_header(csv_path: Path) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(
                [
                    "SMILES",
                    "affinity_pred_value",
                    "affinity_probability_binary",
                    "affinity_pred_value1",
                    "affinity_probability_binary1",
                    "affinity_pred_value2",
                    "affinity_probability_binary2",
                    "error",
                ]
            )


def main() -> None:
    template_path = _env_path("BOLTZ_TEMPLATE_PATH", DEFAULT_TEMPLATE)
    tmp_yaml = _env_path("BOLTZ_TMP_YAML", DEFAULT_TMP_YAML)
    results_dir = _env_path("BOLTZ_RESULTS_DIR", DEFAULT_RESULTS_DIR)
    json_path = _env_path("BOLTZ_JSON_PATH", DEFAULT_JSON_PATH)
    csv_path = _env_path("BOLTZ_OUTPUT_CSV", DEFAULT_CSV_PATH)
    smiles_list = _read_smiles_list()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_yaml.parent.mkdir(parents=True, exist_ok=True)
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_csv_header(csv_path)

    for smiles in smiles_list:
        print(f"\\n[boltz.py] SMILES: {smiles}")
        try:
            data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            data["sequences"][1]["ligand"]["smiles"] = smiles

            with tmp_yaml.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle)

            subprocess.run(
                [
                    "boltz",
                    "predict",
                    str(tmp_yaml),
                    "--out_dir",
                    str(results_dir.parent),
                    "--override",
                ],
                check=True,
            )

            for _ in range(20):
                if json_path.exists():
                    break
                time.sleep(0.5)
            else:
                raise FileNotFoundError(f"Boltz JSON not found: {json_path}")

            with json_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)

            row = [
                smiles,
                metrics.get("affinity_pred_value"),
                metrics.get("affinity_probability_binary"),
                metrics.get("affinity_pred_value1"),
                metrics.get("affinity_probability_binary1"),
                metrics.get("affinity_pred_value2"),
                metrics.get("affinity_probability_binary2"),
                "",
            ]
            print("  [ok] prediction completed")
        except Exception as exc:
            print("  [error] molecule failed")
            print(f"  {exc}")
            traceback.print_exc()
            row = [smiles, None, None, None, None, None, None, str(exc)]
        finally:
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)

    print("\\n[boltz.py] predictions finished")
    print("[boltz.py] CSV saved at:", csv_path.resolve())


if __name__ == "__main__":
    main()
