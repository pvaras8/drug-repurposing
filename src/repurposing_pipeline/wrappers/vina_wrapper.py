"""Vina docking wrapper with Meeko ligand preparation and multiprocessing."""

from __future__ import annotations

from multiprocessing import Pool, Process, Queue, cpu_count
from pathlib import Path
from typing import Any

WORKER_CFG: dict[str, Any] = {}


def _resolve_worker_count(num_processors: int) -> int:
    if num_processors == -1:
        return max(1, cpu_count() - 1)
    return max(1, num_processors)


def _init_worker(config: dict[str, Any]) -> None:
    global WORKER_CFG
    WORKER_CFG = config


def _smiles_to_3d_mol(smiles: str, seed: int) -> Any:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("RDKit failed to generate 3D conformer")

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

    return mol


def _mol_to_pdbqt_string(mol: Any) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    if not setups:
        raise RuntimeError("Meeko did not produce a MoleculeSetup")

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise RuntimeError(f"Meeko PDBQT export failed: {error_msg}")

    return pdbqt_string


def _failure_result(molecule_id: str, ligand_pdbqt_path: Path, pose_pdbqt_path: Path, message: str) -> dict[str, Any]:
    return {
        "molecule_id": molecule_id,
        "docking_status": "failed",
        "vina_score": float(WORKER_CFG["fallback_score"]),
        "vina_ligand_pdbqt": str(ligand_pdbqt_path),
        "vina_pose_pdbqt": str(pose_pdbqt_path),
        "vina_error": message,
    }


def _dock_task_impl(task: tuple[int, str, str, str]) -> dict[str, Any]:
    idx, molecule_id, smiles, canonical_smiles = task
    ligands_dir = Path(WORKER_CFG["ligands_dir"])
    poses_dir = Path(WORKER_CFG["poses_dir"])

    ligand_pdbqt_path = ligands_dir / f"{molecule_id}_input.pdbqt"
    pose_pdbqt_path = poses_dir / f"{molecule_id}_out.pdbqt"

    from vina import Vina

    smiles_for_docking = canonical_smiles or smiles
    mol = _smiles_to_3d_mol(
        smiles=smiles_for_docking,
        seed=int(WORKER_CFG["embed_seed"]) + idx,
    )
    ligand_pdbqt = _mol_to_pdbqt_string(mol)
    ligand_pdbqt_path.write_text(ligand_pdbqt, encoding="utf-8")

    vina_obj = Vina(
        sf_name=WORKER_CFG["sf_name"],
        cpu=int(WORKER_CFG["vina_cpu_per_job"]),
        seed=int(WORKER_CFG["vina_seed"]) + idx,
        verbosity=0,
    )
    vina_obj.set_receptor(str(WORKER_CFG["receptor_pdbqt"]))
    vina_obj.compute_vina_maps(
        center=list(WORKER_CFG["center"]),
        box_size=list(WORKER_CFG["box_size"]),
    )

    vina_obj.set_ligand_from_string(ligand_pdbqt)
    vina_obj.dock(
        exhaustiveness=int(WORKER_CFG["exhaustiveness"]),
        n_poses=int(WORKER_CFG["n_poses"]),
    )

    energies = vina_obj.energies(
        n_poses=int(WORKER_CFG["n_poses"]),
        energy_range=float(WORKER_CFG["energy_range"]),
    )
    if energies is None or len(energies) == 0:
        raise RuntimeError("Vina returned no energies")

    best_score = float(energies[0][0])

    vina_obj.write_poses(
        str(pose_pdbqt_path),
        n_poses=int(WORKER_CFG["write_n_poses"]),
        energy_range=float(WORKER_CFG["energy_range"]),
        overwrite=True,
    )

    return {
        "molecule_id": molecule_id,
        "docking_status": "completed",
        "vina_score": best_score,
        "vina_ligand_pdbqt": str(ligand_pdbqt_path),
        "vina_pose_pdbqt": str(pose_pdbqt_path),
        "vina_error": "",
    }


def _dock_task_subprocess(task: tuple[int, str, str, str], cfg: dict[str, Any], queue: Queue) -> None:
    global WORKER_CFG
    WORKER_CFG = cfg
    try:
        result = _dock_task_impl(task)
    except Exception as exc:  # noqa: BLE001
        _, molecule_id, _, _ = task
        ligands_dir = Path(WORKER_CFG["ligands_dir"])
        poses_dir = Path(WORKER_CFG["poses_dir"])
        ligand_pdbqt_path = ligands_dir / f"{molecule_id}_input.pdbqt"
        pose_pdbqt_path = poses_dir / f"{molecule_id}_out.pdbqt"
        result = _failure_result(molecule_id, ligand_pdbqt_path, pose_pdbqt_path, str(exc))
    queue.put(result)


def _dock_task(task: tuple[int, str, str, str]) -> dict[str, Any]:
    _, molecule_id, _, _ = task
    ligands_dir = Path(WORKER_CFG["ligands_dir"])
    poses_dir = Path(WORKER_CFG["poses_dir"])

    ligand_pdbqt_path = ligands_dir / f"{molecule_id}_input.pdbqt"
    pose_pdbqt_path = poses_dir / f"{molecule_id}_out.pdbqt"

    try:
        timeout_seconds = int(WORKER_CFG["timeout_seconds"])
        hard_timeout = bool(WORKER_CFG.get("hard_timeout", False))
        if hard_timeout and timeout_seconds > 0:
            queue: Queue = Queue(maxsize=1)
            child = Process(target=_dock_task_subprocess, args=(task, dict(WORKER_CFG), queue))
            child.start()
            child.join(timeout_seconds)
            if child.is_alive():
                child.terminate()
                child.join()
                return _failure_result(
                    molecule_id,
                    ligand_pdbqt_path,
                    pose_pdbqt_path,
                    f"Hard timeout after {timeout_seconds}s",
                )
            if queue.empty():
                return _failure_result(
                    molecule_id,
                    ligand_pdbqt_path,
                    pose_pdbqt_path,
                    "Docking subprocess ended without result",
                )
            return queue.get_nowait()

        return _dock_task_impl(task)
    except Exception as exc:
        return _failure_result(molecule_id, ligand_pdbqt_path, pose_pdbqt_path, str(exc))


def run_vina_parallel(
    rows: list[dict[str, Any]],
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    vina_results_dir: Path,
    num_processors: int = -1,
    vina_cpu_per_job: int = 1,
    exhaustiveness: int = 16,
    n_poses: int = 10,
    write_n_poses: int = 10,
    energy_range: float = 3.0,
    fallback_score: float = -1.0,
    timeout_seconds: int = 300,
    hard_timeout: bool = False,
    sf_name: str = "vina",
    embed_seed: int = 42,
    vina_seed: int = 12345,
) -> dict[str, dict[str, Any]]:
    """Dock multiple molecules with Vina and return results indexed by molecule_id."""
    worker_count = _resolve_worker_count(num_processors)
    receptor_resolved = receptor_pdbqt.resolve()
    if not receptor_resolved.exists():
        raise FileNotFoundError(f"Receptor PDBQT not found: {receptor_resolved}")

    # Import checks in parent process so we fail fast with a clear message.
    try:
        import meeko  # noqa: F401
        import vina  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing docking dependencies. Install: pip install -U meeko vina rdkit"
        ) from exc

    ligands_dir = vina_results_dir / "ligands"
    poses_dir = vina_results_dir / "poses"
    ligands_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    worker_cfg: dict[str, Any] = {
        "receptor_pdbqt": str(receptor_resolved),
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "box_size": [float(box_size[0]), float(box_size[1]), float(box_size[2])],
        "exhaustiveness": int(exhaustiveness),
        "n_poses": int(n_poses),
        "write_n_poses": int(write_n_poses),
        "energy_range": float(energy_range),
        "vina_cpu_per_job": int(vina_cpu_per_job),
        "fallback_score": float(fallback_score),
        "timeout_seconds": int(timeout_seconds),
        "hard_timeout": bool(hard_timeout),
        "sf_name": str(sf_name),
        "embed_seed": int(embed_seed),
        "vina_seed": int(vina_seed),
        "ligands_dir": str(ligands_dir),
        "poses_dir": str(poses_dir),
    }

    tasks: list[tuple[int, str, str, str]] = []
    for idx, row in enumerate(rows):
        tasks.append(
            (
                idx,
                str(row["molecule_id"]),
                str(row.get("smiles", "")),
                str(row.get("canonical_smiles", "")),
            )
        )

    by_molecule: dict[str, dict[str, Any]] = {}
    with Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(worker_cfg,),
    ) as pool:
        for result in pool.imap_unordered(_dock_task, tasks):
            by_molecule[result["molecule_id"]] = result

    return by_molecule
