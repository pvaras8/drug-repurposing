"""CLI entrypoint for the repurposing pipeline."""

from __future__ import annotations

from pathlib import Path
import click

from repurposing_pipeline.pipeline import run_pipeline


@click.command()
@click.option("--input-csv", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--receptor", type=click.Path(path_type=Path, exists=False), required=False)
@click.option("--runs-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
@click.option("--run-id", type=str, default="run_001", show_default=True)
@click.option(
    "--run-boltz",
    is_flag=True,
    default=False,
    help="Attempt execution via repository-level boltz.py wrapper.",
)
def main(input_csv: Path, receptor: Path | None, runs_root: Path, run_id: str, run_boltz: bool) -> None:
    """Run the Phase 1 repurposing pipeline without Vina integration."""
    final_csv = run_pipeline(
        input_csv=input_csv,
        receptor_path=receptor,
        runs_root=runs_root,
        run_id=run_id,
        run_boltz=run_boltz,
    )
    click.echo(f"Pipeline finished. Results: {final_csv}")


if __name__ == "__main__":
    main()
