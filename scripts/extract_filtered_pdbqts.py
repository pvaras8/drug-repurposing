#!/usr/bin/env python3
"""Copy Vina poses selected by multiplicative-ranking CSV files.

The script only uses the Python standard library.  It discovers ranking CSVs
below a runs directory, reads their ``molecule_id`` column, and copies the
matching ``<molecule_id>_out.pdbqt`` files into one folder per experiment.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV_PATTERN = "final_multiplicative_ranked.csv"


@dataclass(frozen=True)
class ManifestRow:
    experiment: str
    molecule_id: str
    status: str
    source_csv: str
    source_pdbqt: str
    copied_to: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae las poses PDBQT de las moléculas presentes en los CSV "
            "de ranking final de todos los experimentos."
        )
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Carpeta que contiene las carpetas de los experimentos.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Carpeta de destino (se creará si no existe).",
    )
    parser.add_argument(
        "--csv-pattern",
        default=DEFAULT_CSV_PATTERN,
        help=(
            "Nombre o patrón glob de los CSV (por defecto: "
            f"{DEFAULT_CSV_PATTERN}). Ejemplo: '*multiplicative*ranked.csv'."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="RUN",
        help=(
            "Procesa únicamente estas carpetas de experimento (nombres relativos "
            "a runs_dir). Si se omite, procesa todos los experimentos."
        ),
    )
    parser.add_argument(
        "--id-column",
        default="molecule_id",
        help="Columna del CSV que contiene el identificador (por defecto: molecule_id).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe PDBQT que ya existan en el destino.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Comprueba y genera el informe sin copiar archivos.",
    )
    return parser.parse_args()


def experiment_root(csv_path: Path) -> Path:
    """Return the run root for both run/output/file.csv and run/file.csv."""
    if csv_path.parent.name.lower() == "output":
        return csv_path.parent.parent
    return csv_path.parent


def experiment_name(run_root: Path, runs_dir: Path) -> str:
    """Create a stable, collision-resistant directory name."""
    relative = run_root.relative_to(runs_dir)
    return "__".join(relative.parts) or run_root.name


def read_ids(csv_path: Path, id_column: str) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or []) or "ninguna"
            raise ValueError(
                f"falta la columna {id_column!r}; columnas disponibles: {available}"
            )

        ids: list[str] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            molecule_id = (row.get(id_column) or "").strip()
            if not molecule_id:
                print(
                    f"AVISO: {csv_path}:{line_number}: identificador vacío; se omite",
                    file=sys.stderr,
                )
                continue
            if molecule_id not in seen:
                ids.append(molecule_id)
                seen.add(molecule_id)
        return ids


def pose_index(run_root: Path) -> dict[str, list[Path]]:
    """Index output poses inside one run, preferring the standard directory."""
    standard_dir = run_root / "vina_results" / "poses"
    files = standard_dir.glob("*_out.pdbqt") if standard_dir.is_dir() else run_root.rglob("*_out.pdbqt")
    index: dict[str, list[Path]] = {}
    for path in files:
        molecule_id = path.name[: -len("_out.pdbqt")]
        index.setdefault(molecule_id, []).append(path)
    return index


def process_csv(
    csv_path: Path,
    runs_dir: Path,
    output_dir: Path,
    id_column: str,
    overwrite: bool,
    dry_run: bool,
) -> list[ManifestRow]:
    run_root = experiment_root(csv_path)
    experiment = experiment_name(run_root, runs_dir)
    destination_dir = output_dir / experiment
    ids = read_ids(csv_path, id_column)
    index = pose_index(run_root)
    rows: list[ManifestRow] = []

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for molecule_id in ids:
        matches = index.get(molecule_id, [])
        source = matches[0] if len(matches) == 1 else None
        destination = destination_dir / f"{molecule_id}_out.pdbqt"

        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "ambiguous"
        elif destination.exists() and not overwrite:
            status = "already_exists"
        else:
            status = "would_copy" if dry_run else "copied"
            if not dry_run:
                shutil.copy2(source, destination)

        rows.append(
            ManifestRow(
                experiment=experiment,
                molecule_id=molecule_id,
                status=status,
                source_csv=str(csv_path),
                source_pdbqt=(";".join(str(path) for path in matches)),
                copied_to=str(destination) if source else "",
            )
        )
    return rows


def write_manifest(output_dir: Path, rows: list[ManifestRow]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    return manifest_path


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not runs_dir.is_dir():
        print(f"ERROR: no existe la carpeta de runs: {runs_dir}", file=sys.stderr)
        return 2
    if output_dir == runs_dir or runs_dir in output_dir.parents:
        print(
            "ERROR: la carpeta de salida debe estar fuera de la carpeta de runs "
            "para evitar que vuelva a ser escaneada.",
            file=sys.stderr,
        )
        return 2

    if args.experiments:
        selected_roots: list[Path] = []
        invalid_experiments: list[str] = []
        for name in args.experiments:
            candidate = (runs_dir / name).resolve()
            if runs_dir not in candidate.parents or not candidate.is_dir():
                invalid_experiments.append(name)
            else:
                selected_roots.append(candidate)
        if invalid_experiments:
            print(
                "ERROR: no existen estas carpetas de experimento dentro de "
                f"{runs_dir}: {', '.join(invalid_experiments)}",
                file=sys.stderr,
            )
            return 2
        csv_paths = sorted(
            csv_path
            for selected_root in selected_roots
            for csv_path in selected_root.rglob(args.csv_pattern)
        )
    else:
        csv_paths = sorted(runs_dir.rglob(args.csv_pattern))
    if not csv_paths:
        print(
            f"ERROR: no se encontraron CSV con el patrón {args.csv_pattern!r} "
            f"dentro de {runs_dir}",
            file=sys.stderr,
        )
        return 1

    manifest_rows: list[ManifestRow] = []
    csv_errors = 0
    for csv_path in csv_paths:
        try:
            rows = process_csv(
                csv_path,
                runs_dir,
                output_dir,
                args.id_column,
                args.overwrite,
                args.dry_run,
            )
        except (OSError, csv.Error, ValueError) as exc:
            csv_errors += 1
            print(f"ERROR: {csv_path}: {exc}", file=sys.stderr)
            continue
        manifest_rows.extend(rows)
        print(f"{experiment_name(experiment_root(csv_path), runs_dir)}: {len(rows)} moléculas")

    manifest_path = write_manifest(output_dir, manifest_rows)
    counts: dict[str, int] = {}
    for row in manifest_rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    print(f"\nCSV procesados: {len(csv_paths) - csv_errors}/{len(csv_paths)}")
    print(f"Moléculas totales: {len(manifest_rows)}")
    for status in ("copied", "would_copy", "already_exists", "missing", "ambiguous"):
        if counts.get(status):
            print(f"{status}: {counts[status]}")
    print(f"Informe: {manifest_path}")

    return 1 if csv_errors or counts.get("missing") or counts.get("ambiguous") else 0


if __name__ == "__main__":
    raise SystemExit(main())
