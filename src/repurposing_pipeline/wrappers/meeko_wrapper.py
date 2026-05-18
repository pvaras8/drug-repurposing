"""MEEKO wrapper placeholder.

Vina integration is intentionally deferred to Phase 2.
"""

from __future__ import annotations

from pathlib import Path


def prepare_ligand_for_vina(input_path: Path, output_path: Path) -> Path:
    """Placeholder for SDF/MOL2 -> PDBQT conversion.

    This function is intentionally not implemented in this phase.
    """
    raise NotImplementedError(
        "Vina preparation is deferred. Implement in Phase 2 after the Vina design decision."
    )
