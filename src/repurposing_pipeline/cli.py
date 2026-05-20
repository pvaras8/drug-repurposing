"""CLI entrypoint for the repurposing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import click
from click.core import ParameterSource

from repurposing_pipeline.io import ensure_run_paths
from repurposing_pipeline.pipeline import run_pipeline
from repurposing_pipeline.wrappers.meeko_wrapper import (
    parse_triplet,
    receptor_seems_protonated,
    validate_box,
)


def _prompt_path(text: str, must_exist: bool = True) -> Path:
    """Prompt for filesystem path and return resolved absolute Path."""
    return Path(
        click.prompt(
            text,
            type=click.Path(path_type=Path, exists=must_exist, file_okay=True, dir_okay=False),
        )
    ).resolve()


def _load_vina_config(path: Path) -> dict[str, object]:
    """Load Vina config from JSON file if it exists."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid Vina config format in {path}: expected a mapping")
    return loaded


def _resolve_docking_inputs(
    receptor: Path | None,
    receptor_ready_pdbqt: Path | None,
    pocket_center: str | None,
    pocket_size: str,
    interactive_docking_setup: bool,
    allow_non_protonated_receptor: bool,
) -> tuple[Path | None, dict[str, object] | None]:
    """Validate docking receptor + pocket setup before pipeline run."""
    ready_receptor = receptor_ready_pdbqt or receptor

    if interactive_docking_setup and ready_receptor is None:
        ready_receptor = _prompt_path("Ruta al receptor PDBQT preparado")

    if ready_receptor is None:
        return None, None

    box_center = parse_triplet(pocket_center) if pocket_center else None
    box_size_values = parse_triplet(pocket_size)

    if box_center is None and interactive_docking_setup:
        has_coords = click.confirm("Tienes coordenadas del pocket (centro y tamano)?", default=False)
        if has_coords:
            box_center = parse_triplet(click.prompt("Centro del pocket (x,y,z)"))
            box_size_values = parse_triplet(click.prompt("Tamano del pocket (x,y,z)", default=pocket_size))

    ready_receptor = ready_receptor.resolve()
    if not ready_receptor.exists():
        raise click.ClickException(f"Receptor PDBQT not found: {ready_receptor}")
    if box_center is None:
        raise click.ClickException(
            "For a preprocessed receptor, provide pocket coordinates with --pocket-center "
            "or answer the interactive pocket prompt"
        )

    validate_box(box_center, box_size_values)

    protonated, atom_count, hydrogen_count, hydrogen_ratio = receptor_seems_protonated(ready_receptor)
    if not protonated:
        warning = (
            "Receptor seems weakly protonated "
            f"(atoms={atom_count}, H={hydrogen_count}, ratio={hydrogen_ratio:.3f}). "
            "Se recomienda protonar antes de docking."
        )
        click.echo(f"WARNING: {warning}")
        if interactive_docking_setup and not allow_non_protonated_receptor:
            should_continue = click.confirm("Quieres continuar de todas formas?", default=False)
            if not should_continue:
                raise click.ClickException("Docking cancelled: receptor not protonated")
        elif not allow_non_protonated_receptor:
            raise click.ClickException(
                "Receptor may be non-protonated. Re-run with --allow-non-protonated-receptor to continue"
            )

    metadata = {
        "receptor_mode": "ready",
        "receptor_pdbqt": str(ready_receptor),
        "box_center": list(box_center),
        "box_size": list(box_size_values),
        "box_source": "manual",
        "selected_ligand": None,
        "protonated_check": {
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "hydrogen_ratio": hydrogen_ratio,
            "seems_protonated": protonated,
        },
    }
    return ready_receptor, metadata


@click.command()
@click.option("--input-csv", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--receptor", type=click.Path(path_type=Path, exists=True), required=False)
@click.option("--receptor-ready-pdbqt", type=click.Path(path_type=Path, exists=True), required=False)
@click.option("--pocket-center", type=str, required=False, help="Pocket center triplet: 'x,y,z'")
@click.option("--pocket-size", type=str, default="20,20,20", show_default=True)
@click.option(
    "--allow-non-protonated-receptor",
    is_flag=True,
    default=False,
    help="Continue even if receptor appears non-protonated.",
)
@click.option(
    "--interactive-docking-setup/--no-interactive-docking-setup",
    default=True,
    show_default=True,
    help="Ask interactive docking setup questions when receptor data is missing.",
)
@click.option("--runs-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--run-id", type=str, default="run_001", show_default=True)
@click.option(
    "--vina-config",
    type=click.Path(path_type=Path, exists=False),
    default=Path("config/vina.json"),
    show_default=True,
    help="Path to Vina JSON config file.",
)
@click.option(
    "--run-vina/--no-run-vina",
    default=True,
    show_default=True,
    help="Run Vina docking when receptor + pocket setup is available.",
)
@click.option("--vina-num-processors", type=int, default=-1, show_default=True)
@click.option("--vina-cpu-per-job", type=int, default=1, show_default=True)
@click.option("--vina-exhaustiveness", type=int, default=16, show_default=True)
@click.option("--vina-n-poses", type=int, default=10, show_default=True)
@click.option("--vina-write-n-poses", type=int, default=10, show_default=True)
@click.option("--vina-energy-range", type=float, default=3.0, show_default=True)
@click.option("--vina-fallback-score", type=float, default=-1.0, show_default=True)
@click.option("--vina-timeout-seconds", type=int, default=300, show_default=True)
@click.option("--vina-embed-seed", type=int, default=42, show_default=True)
@click.option("--vina-seed", type=int, default=12345, show_default=True)
@click.option(
    "--run-boltz",
    is_flag=True,
    default=False,
    help="Attempt execution via repository-level boltz.py wrapper.",
)
@click.pass_context
def main(
    ctx: click.Context,
    input_csv: Path,
    receptor: Path | None,
    receptor_ready_pdbqt: Path | None,
    pocket_center: str | None,
    pocket_size: str,
    allow_non_protonated_receptor: bool,
    interactive_docking_setup: bool,
    runs_root: Path,
    run_id: str,
    vina_config: Path,
    run_vina: bool,
    vina_num_processors: int,
    vina_cpu_per_job: int,
    vina_exhaustiveness: int,
    vina_n_poses: int,
    vina_write_n_poses: int,
    vina_energy_range: float,
    vina_fallback_score: float,
    vina_timeout_seconds: int,
    vina_embed_seed: int,
    vina_seed: int,
    run_boltz: bool,
) -> None:
    """Run the repurposing pipeline with optional receptor/pocket setup for docking."""
    config_path = vina_config.resolve()
    try:
        vina_cfg = _load_vina_config(config_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    # If user did not pass a CLI value explicitly, take value from config file.
    if ctx.get_parameter_source("run_vina") == ParameterSource.DEFAULT:
        run_vina = bool(vina_cfg.get("run_vina", run_vina))
    if ctx.get_parameter_source("vina_num_processors") == ParameterSource.DEFAULT:
        vina_num_processors = int(vina_cfg.get("num_processors", vina_num_processors))
    if ctx.get_parameter_source("vina_cpu_per_job") == ParameterSource.DEFAULT:
        vina_cpu_per_job = int(vina_cfg.get("vina_cpu_per_job", vina_cpu_per_job))
    if ctx.get_parameter_source("vina_exhaustiveness") == ParameterSource.DEFAULT:
        vina_exhaustiveness = int(vina_cfg.get("exhaustiveness", vina_exhaustiveness))
    if ctx.get_parameter_source("vina_n_poses") == ParameterSource.DEFAULT:
        vina_n_poses = int(vina_cfg.get("n_poses", vina_n_poses))
    if ctx.get_parameter_source("vina_write_n_poses") == ParameterSource.DEFAULT:
        vina_write_n_poses = int(vina_cfg.get("write_n_poses", vina_write_n_poses))
    if ctx.get_parameter_source("vina_energy_range") == ParameterSource.DEFAULT:
        vina_energy_range = float(vina_cfg.get("energy_range", vina_energy_range))
    if ctx.get_parameter_source("vina_fallback_score") == ParameterSource.DEFAULT:
        vina_fallback_score = float(vina_cfg.get("fallback_score", vina_fallback_score))
    if ctx.get_parameter_source("vina_timeout_seconds") == ParameterSource.DEFAULT:
        vina_timeout_seconds = int(vina_cfg.get("timeout_seconds", vina_timeout_seconds))
    if ctx.get_parameter_source("vina_embed_seed") == ParameterSource.DEFAULT:
        vina_embed_seed = int(vina_cfg.get("embed_seed", vina_embed_seed))
    if ctx.get_parameter_source("vina_seed") == ParameterSource.DEFAULT:
        vina_seed = int(vina_cfg.get("vina_seed", vina_seed))

    run_paths = ensure_run_paths(runs_root, run_id)

    try:
        receptor_path, docking_setup = _resolve_docking_inputs(
            receptor=receptor,
            receptor_ready_pdbqt=receptor_ready_pdbqt,
            pocket_center=pocket_center,
            pocket_size=pocket_size,
            interactive_docking_setup=interactive_docking_setup,
            allow_non_protonated_receptor=allow_non_protonated_receptor,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if run_vina and receptor_path is None:
        raise click.ClickException(
            "Vina requires a prepared receptor PDBQT. Provide --receptor-ready-pdbqt (or --receptor)."
        )

    if docking_setup is not None:
        setup_json = run_paths.output / "docking_setup.json"
        setup_json.write_text(json.dumps(docking_setup, indent=2), encoding="utf-8")
        center = docking_setup["box_center"]
        size = docking_setup["box_size"]
        click.echo("PDB listo para docking, y coordenadas listas")
        click.echo(f"Centro pocket: {center}")
        click.echo(f"Tamano pocket: {size}")

    final_csv = run_pipeline(
        input_csv=input_csv,
        receptor_path=receptor_path,
        docking_setup=docking_setup,
        runs_root=runs_root,
        run_id=run_id,
        run_vina=run_vina,
        vina_num_processors=vina_num_processors,
        vina_cpu_per_job=vina_cpu_per_job,
        vina_exhaustiveness=vina_exhaustiveness,
        vina_n_poses=vina_n_poses,
        vina_write_n_poses=vina_write_n_poses,
        vina_energy_range=vina_energy_range,
        vina_fallback_score=vina_fallback_score,
        vina_timeout_seconds=vina_timeout_seconds,
        vina_embed_seed=vina_embed_seed,
        vina_seed=vina_seed,
        run_boltz=run_boltz,
    )
    click.echo(f"Pipeline finished. Results: {final_csv}")


if __name__ == "__main__":
    main()
