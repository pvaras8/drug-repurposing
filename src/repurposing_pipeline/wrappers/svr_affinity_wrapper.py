"""SVR affinity prediction wrapper (post-Vina, pose-based).

Mirrors the affinity prediction logic from PockLigGPT's reward_meeko_vina.py,
adapted to the dict-based data model of drug-repurposing.

The external ``predict_affinity.py`` script is invoked via subprocess with a
manifest CSV (ligand_id, ligand_mol2, vina_pdbqt, protein_pdb).  Ligand mol2
files are generated from the Vina input PDBQT using obabel.
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_abs(path_str: str, base_dir: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(Path(base_dir) / p)


def _pdbqt_to_mol2(
    pdbqt_path: str,
    mol2_path: str,
    obabel_bin: str,
    conda_env: str = "",
    activation: str = "conda_run",
    conda_bin: str = "conda",
) -> None:
    """Convert a ligand PDBQT to mol2 using obabel."""
    obabel_cmd = (
        f"{shlex.quote(obabel_bin)} -ipdbqt {shlex.quote(pdbqt_path)} "
        f"-omol2 -O {shlex.quote(mol2_path)}"
    )
    if conda_env:
        if activation == "source_activate":
            cmd = f"source activate {shlex.quote(conda_env)} && {obabel_cmd}"
        else:
            cmd = f"{shlex.quote(conda_bin)} run -n {shlex.quote(conda_env)} {obabel_cmd}"
        completed = subprocess.run(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        completed = subprocess.run(
            [obabel_bin, "-ipdbqt", pdbqt_path, "-omol2", "-O", mol2_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if completed.returncode != 0 or not os.path.exists(mol2_path):
        raise RuntimeError(
            f"obabel failed for {pdbqt_path}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _build_affinity_command(
    affinity: dict[str, Any],
    manifest_path: str,
    output_path: str,
) -> str:
    python_bin = str(affinity.get("python_bin", "python"))
    script = str(affinity["script"])
    parts = [
        shlex.quote(python_bin),
        shlex.quote(script),
        "--manifest", shlex.quote(manifest_path),
        "--model", shlex.quote(str(affinity["model"])),
        "--output", shlex.quote(output_path),
    ]

    project_dir = str(affinity["project_dir"])
    if affinity["model"] == "svr_fp_pose":
        svr_model = _maybe_abs(
            affinity.get("svr_model", "checkpoints/svr_affinity_fp_pose_model.joblib"),
            project_dir,
        )
        svr_meta = _maybe_abs(
            affinity.get("svr_meta", "checkpoints/svr_affinity_fp_pose_metadata.json"),
            project_dir,
        )
        parts.extend(["--svr-model", shlex.quote(svr_model)])
        parts.extend(["--svr-meta", shlex.quote(svr_meta)])
    else:
        dimenet_model = _maybe_abs(
            affinity.get("dimenet_model", "checkpoints/dimenet_model.pt"),
            project_dir,
        )
        parts.extend(["--dimenet-model", shlex.quote(dimenet_model)])
        parts.extend(["--device", shlex.quote(str(affinity.get("device", "cpu")))])

    cmd = " ".join(parts)
    conda_env = str(affinity.get("conda_env", "")).strip()
    if conda_env:
        activation = str(affinity.get("activation", "conda_run"))
        conda_bin = str(affinity.get("conda_bin", "conda"))
        if activation == "source_activate":
            cmd = f"source activate {shlex.quote(conda_env)} && {cmd}"
        else:
            cmd = f"{shlex.quote(conda_bin)} run -n {shlex.quote(conda_env)} {cmd}"
    return cmd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_affinity_cfg(affinity: dict[str, Any]) -> None:
    """Raise if the affinity config block is invalid. No-op when disabled."""
    if not affinity.get("enabled", False):
        return
    model = str(affinity.get("model", "svr_fp_pose"))
    if model not in {"svr_fp_pose", "dimenet"}:
        raise ValueError("affinity.model must be 'svr_fp_pose' or 'dimenet'")
    selection = str(affinity.get("selection", "min"))
    if selection not in {"min", "max", "first"}:
        raise ValueError("affinity.selection must be 'min', 'max', or 'first'")
    activation = str(affinity.get("activation", "conda_run"))
    if activation not in {"conda_run", "source_activate"}:
        raise ValueError("affinity.activation must be 'conda_run' or 'source_activate'")
    script = str(affinity.get("script", ""))
    if not script:
        raise ValueError("affinity.script is required when affinity is enabled")
    if not os.path.exists(script):
        raise FileNotFoundError(f"predict_affinity.py not found: {script}")
    if model == "dimenet" and not affinity.get("protein_pdb"):
        raise ValueError("affinity.protein_pdb is required when model='dimenet'")


def run_svr_affinity(
    vina_by_id: dict[str, dict[str, Any]],
    affinity: dict[str, Any],
    output_dir: Path,
    protein_pdb: str = "",
) -> dict[str, float]:
    """Run SVR affinity prediction on completed Vina poses.

    Returns ``{molecule_id: predicted_score}`` for molecules that received a
    prediction.  Molecules that fail mol2 conversion or have no prediction are
    silently excluded (the caller decides whether to fall back to Vina scores).
    Returns an empty dict when ``affinity["enabled"]`` is ``False``.
    """
    if not affinity.get("enabled", False):
        return {}

    candidates = {
        mol_id: info
        for mol_id, info in vina_by_id.items()
        if info.get("docking_status") == "completed"
        and info.get("vina_ligand_pdbqt")
        and info.get("vina_pose_pdbqt")
    }
    if not candidates:
        return {}

    obabel_bin = str(affinity.get("obabel_bin", "obabel"))
    obabel_conda_env = str(affinity.get("obabel_conda_env", affinity.get("conda_env", ""))).strip()
    activation = str(affinity.get("activation", "conda_run"))
    conda_bin = str(affinity.get("conda_bin", "conda"))

    mol2_dir = output_dir / "svr_mol2"
    mol2_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for mol_id, info in candidates.items():
        mol2_path = str(mol2_dir / f"{mol_id}.mol2")
        try:
            _pdbqt_to_mol2(
                pdbqt_path=str(info["vina_ligand_pdbqt"]),
                mol2_path=mol2_path,
                obabel_bin=obabel_bin,
                conda_env=obabel_conda_env,
                activation=activation,
                conda_bin=conda_bin,
            )
        except RuntimeError:
            continue
        manifest_rows.append({
            "ligand_id": mol_id,
            "ligand_mol2": mol2_path,
            "vina_pdbqt": str(os.path.abspath(info["vina_pose_pdbqt"])),
            "protein_pdb": str(os.path.abspath(protein_pdb)) if protein_pdb else "",
        })

    if not manifest_rows:
        return {}

    manifest_path = str(output_dir / "svr_manifest.csv")
    prediction_path = str(output_dir / "svr_predictions.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["ligand_id", "ligand_mol2", "vina_pdbqt", "protein_pdb"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    cmd = _build_affinity_command(affinity, manifest_path, prediction_path)
    completed = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=str(affinity["project_dir"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        if affinity.get("fallback_to_vina", True):
            return {}
        raise RuntimeError(
            f"SVR affinity prediction failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    if not os.path.exists(prediction_path):
        if affinity.get("fallback_to_vina", True):
            return {}
        raise FileNotFoundError(
            f"predict_affinity.py did not produce expected CSV: {prediction_path}"
        )

    score_col = str(affinity.get("score_column", "predicted_affinity_kcal_mol"))
    selection = str(affinity.get("selection", "min"))

    pred_rows: list[dict[str, str]] = []
    with open(prediction_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pred_rows.append(dict(row))

    if not pred_rows or score_col not in (pred_rows[0] if pred_rows else {}):
        return {}

    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for idx, row in enumerate(pred_rows):
        try:
            score = float(row[score_col])
        except (ValueError, KeyError):
            continue
        groups[str(row["ligand_id"])].append((score, idx))

    result: dict[str, float] = {}
    for mol_id, score_list in groups.items():
        if not score_list:
            continue
        if selection == "min":
            result[mol_id] = min(s for s, _ in score_list)
        elif selection == "max":
            result[mol_id] = max(s for s, _ in score_list)
        else:  # first
            result[mol_id] = sorted(score_list, key=lambda t: t[1])[0][0]

    return result
