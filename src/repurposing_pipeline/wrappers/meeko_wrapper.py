"""MEEKO-related helpers for docking input validation."""

from __future__ import annotations

from pathlib import Path
import re


def parse_triplet(value: str) -> tuple[float, float, float]:
    """Parse coordinate triplets from comma or whitespace-separated text."""
    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if len(tokens) != 3:
        raise ValueError("Expected 3 values (x, y, z)")
    return float(tokens[0]), float(tokens[1]), float(tokens[2])


def validate_box(center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    """Validate docking center and size values."""
    for axis, value in zip(("x", "y", "z"), center):
        if not (-10000.0 < value < 10000.0):
            raise ValueError(f"Box center {axis} out of range: {value}")
    for axis, value in zip(("x", "y", "z"), size):
        if value <= 0:
            raise ValueError(f"Box size {axis} must be > 0")
        if value > 150:
            raise ValueError(f"Box size {axis} too large: {value}")


def receptor_seems_protonated(pdbqt_path: Path, min_ratio: float = 0.03) -> tuple[bool, int, int, float]:
    """Heuristic check for receptor protonation based on hydrogen atom ratio."""
    atom_count = 0
    hydrogen_count = 0

    with pdbqt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_count += 1
            atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
            if atom_name.startswith("H"):
                hydrogen_count += 1

    ratio = (hydrogen_count / atom_count) if atom_count else 0.0
    return ratio >= min_ratio, atom_count, hydrogen_count, ratio


def prepare_ligand_for_vina(input_path: Path, output_path: Path) -> Path:
    """Placeholder for SDF/MOL2 -> PDBQT conversion."""
    raise NotImplementedError(
        "Vina preparation is deferred. Implement in Phase 2 after the Vina design decision."
    )
