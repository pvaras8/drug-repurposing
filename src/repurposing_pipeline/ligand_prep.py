"""Ligand preparation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repurposing_pipeline.wrappers.rdkit_wrapper import canonicalize_smiles


def prepare_ligands(rows: list[dict[str, str]], prepared_dir: Path) -> list[dict[str, Any]]:
    """Prepare ligands and persist canonical SMILES artifacts per molecule."""
    prepared_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        molecule_id = row["molecule_id"]
        smiles = row["smiles"]
        canonical, is_valid, error = canonicalize_smiles(smiles)

        prepared_path = prepared_dir / f"{molecule_id}.smi"
        if is_valid:
            prepared_path.write_text(canonical + "\n", encoding="utf-8")
            ligand_prep_status = "completed"
            prep_error = None
        else:
            ligand_prep_status = "failed"
            prep_error = error or "Invalid SMILES"

        prepared_rows.append(
            {
                **row,
                "canonical_smiles": canonical,
                "prepared_smiles_path": str(prepared_path),
                "ligand_prep_status": ligand_prep_status,
                "ligand_prep_error": prep_error,
            }
        )

    return prepared_rows
