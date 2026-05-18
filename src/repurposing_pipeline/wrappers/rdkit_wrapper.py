"""RDKit helper wrapper.

This module is intentionally defensive: if RDKit is not installed,
it returns status information instead of crashing the full run.
"""

from __future__ import annotations

from typing import Tuple


def canonicalize_smiles(smiles: str) -> Tuple[str, bool, str | None]:
    """Return canonical SMILES when RDKit is available.

    Returns tuple: (smiles_out, is_valid, error_message)
    """
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - environment-dependent
        return smiles, True, f"RDKit unavailable: {exc}"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False, "Invalid SMILES"

    canonical = Chem.MolToSmiles(mol, canonical=True)
    return canonical, True, None
