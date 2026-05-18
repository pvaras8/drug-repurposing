# Vina Integration Options (Phase 0)

This document compares options to integrate docking in the repurposing pipeline.

## Scope and constraints

- Keep src/boltz unchanged.
- Use boltz.py as the integration wrapper for Boltz stage.
- Support local and HPC execution.
- Keep run outputs under runs/run_XXX/.
- Do not implement Vina yet in this phase.

## Option 1: AutoDock Vina CLI via subprocess

Summary:
- Use vina executable directly from pipeline wrappers.

Pros:
- Stable and common in HPC environments.
- Easy to track exact command for reproducibility.
- No Python API lock-in.

Cons:
- Requires external binary installation management.
- Parsing output files and logs is needed.

Operational notes:
- Best fit for batch systems and job arrays.
- Add timeout and retry wrappers around subprocess calls.

## Option 2: Python vina bindings

Summary:
- Use Python package API to control docking from code.

Pros:
- Programmatic integration and less shell glue.
- Potentially cleaner unit tests around in-memory calls.

Cons:
- Environment compatibility can vary across clusters.
- Some workflows still depend on external prep utilities.

Operational notes:
- Good for local prototyping.
- Validate portability before HPC adoption.

## Option 3: Meeko prep + Vina CLI docking

Summary:
- Use Meeko for ligand/receptor preparation and Vina CLI for docking.

Pros:
- Clear separation of prep and docking.
- Widely adopted practical workflow.
- Good compatibility with existing PDBQT-based pipelines.

Cons:
- More moving parts and version coordination.
- Requires strict artifact contracts between prep and docking steps.

Operational notes:
- Strong candidate for reproducible production workflows.

## Option 4: Abstraction supporting smina/gnina backends

Summary:
- Define common docking interface, with pluggable engines.

Pros:
- Future-proof architecture and easier benchmarking.
- Enables phased backend adoption.

Cons:
- Slightly more design overhead up front.
- Need backend-specific parser differences.

Operational notes:
- Recommended as architectural direction even if Vina is first backend.

## Decision matrix (high level)

- Installation on local + HPC: Option 1 and 3 are usually strongest.
- Reproducibility: Option 1 and 3 are strong when command + versions are logged.
- Performance and batch operation: Option 1 and 3 fit HPC patterns well.
- Maintainability: Option 4 improves long-term maintainability.

## Recommendation

Recommended path:
1. Use Option 3 as primary implementation path (Meeko prep + Vina CLI).
2. Implement under a backend abstraction (Option 4) to keep room for smina/gnina.
3. Keep Python bindings (Option 2) as optional local experimentation path.

## Proposed interface (no backend implementation yet)

```python
class DockingBackend:
    def prepare(self, molecule_id: str, smiles: str, prepared_dir: Path) -> dict:
        ...

    def dock(self, molecule_id: str, prepared_artifacts: dict, receptor_path: Path, out_dir: Path) -> dict:
        ...

    def parse(self, molecule_id: str, raw_output_dir: Path) -> dict:
        ...
```

This interface will be implemented in Phase 2 after approval.
