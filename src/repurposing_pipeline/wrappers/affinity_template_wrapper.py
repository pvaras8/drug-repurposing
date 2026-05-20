"""Generate Boltz affinity template from receptor PDB and pocket center."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import yaml

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "CYX": "C",
    "CME": "C",
    "CSO": "C",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


def _parse_atoms(pdb_lines: list[str]) -> list[tuple[str, int, str, str, tuple[float, float, float]]]:
    atoms: list[tuple[str, int, str, str, tuple[float, float, float]]] = []
    for line in pdb_lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        chain = line[21].strip() or "A"
        resseq = int(line[22:26].strip())
        icode = line[26].strip()
        resname = line[17:20].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        atoms.append((chain, resseq, icode, resname, (x, y, z)))
    return atoms


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _residues_within_radius(
    atoms: list[tuple[str, int, str, str, tuple[float, float, float]]],
    center: tuple[float, float, float],
    radius: float,
) -> list[tuple[str, int, str, str, float]]:
    best: dict[tuple[str, int, str, str], float] = {}
    for chain, resseq, icode, resname, xyz in atoms:
        d = _distance(xyz, center)
        if d <= radius:
            key = (chain, resseq, icode, resname)
            best[key] = min(best.get(key, float("inf")), d)

    items = sorted(best.items(), key=lambda kv: kv[1])
    return [(chain, resseq, icode, resname, dmin) for (chain, resseq, icode, resname), dmin in items]


def _extract_sequence(
    atoms: list[tuple[str, int, str, str, tuple[float, float, float]]],
    chain_id: str,
) -> tuple[str, list[tuple[int, str, str]]]:
    seen: set[tuple[int, str]] = set()
    residues: list[tuple[int, str, str]] = []

    for chain, resseq, icode, resname, _ in atoms:
        if chain != chain_id:
            continue
        key = (resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        residues.append((resseq, icode, resname))

    residues.sort(key=lambda t: (t[0], t[1]))

    sequence_chars: list[str] = []
    unknown: list[tuple[int, str, str]] = []
    for resseq, icode, resname in residues:
        aa = AA3_TO_1.get(resname, "X")
        sequence_chars.append(aa)
        if aa == "X":
            unknown.append((resseq, icode, resname))

    return "".join(sequence_chars), unknown


def build_affinity_template_from_pdb(
    pdb_path: Path,
    output_yaml: Path,
    center: tuple[float, float, float],
    radius: float = 8.0,
    chain_id: str = "A",
) -> dict[str, Any]:
    """Generate and overwrite Boltz affinity template YAML from receptor PDB."""
    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    atoms = _parse_atoms(lines)
    if not atoms:
        raise ValueError(f"No atoms parsed from PDB: {pdb_path}")

    residues = _residues_within_radius(atoms=atoms, center=center, radius=radius)
    if not residues:
        raise ValueError("No residues found within radius for Boltz pocket constraints")

    sequence, unknown = _extract_sequence(atoms=atoms, chain_id=chain_id)
    if not sequence:
        raise ValueError(f"Could not extract sequence for chain {chain_id}")

    contacts = [[chain, int(resseq)] for chain, resseq, _, _, _ in residues]

    data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "msa": "empty",
                }
            },
            {
                "ligand": {
                    "id": "L",
                    "smiles": "C",
                }
            },
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": "L",
                    "contacts": contacts,
                    "max_distance": 6,
                    "force": True,
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "L",
                }
            }
        ],
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return {
        "contacts_count": len(contacts),
        "sequence_length": len(sequence),
        "unknown_residues": unknown,
    }
