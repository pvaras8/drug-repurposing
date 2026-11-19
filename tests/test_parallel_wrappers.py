from pathlib import Path
from types import SimpleNamespace

from repurposing_pipeline.wrappers import svr_affinity_wrapper
from repurposing_pipeline.wrappers.boltz_wrapper import run_boltz_parallel


def test_parallel_boltz_reports_missing_template_per_molecule(tmp_path: Path) -> None:
    result = run_boltz_parallel(
        repo_root=tmp_path,
        molecules=[("mol_a", "CC"), ("mol_b", "CCC")],
        logs_dir=tmp_path / "logs",
        boltz_results_dir=tmp_path / "results",
        template_path=None,
        gpu_ids=[0, 1],
    )

    assert list(result) == ["mol_a", "mol_b"]
    assert all(item["boltz_status"] == "failed" for item in result.values())


def test_parallel_svr_preserves_manifest_order_and_loads_model_once(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "affinity_project"
    project.mkdir()
    script = project / "predict_affinity.py"
    script.write_text("# test stub", encoding="utf-8")
    model = project / "model.joblib"
    meta = project / "meta.json"
    model.touch()
    meta.touch()
    calls = {"prediction": 0}

    def fake_run(args, **kwargs):
        command = args[-1] if args[:2] == ["bash", "-lc"] else " ".join(args)
        if "obabel" in command:
            output = command.split(" -O ", 1)[1].strip("'")
            Path(output).write_text("mol2", encoding="utf-8")
        else:
            calls["prediction"] += 1
            manifest = tmp_path / "output" / "svr_manifest.csv"
            prediction = tmp_path / "output" / "svr_predictions.csv"
            ids = [line.split(",", 1)[0] for line in manifest.read_text().splitlines()[1:]]
            prediction.write_text(
                "ligand_id,predicted_affinity_kcal_mol\n"
                + "".join(f"{mol_id},-{index + 1}.0\n" for index, mol_id in enumerate(ids)),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(svr_affinity_wrapper.subprocess, "run", fake_run)
    vina = {}
    for mol_id in ("mol_b", "mol_a", "mol_c"):
        ligand = tmp_path / f"{mol_id}_ligand.pdbqt"
        pose = tmp_path / f"{mol_id}_pose.pdbqt"
        ligand.touch()
        pose.touch()
        vina[mol_id] = {
            "docking_status": "completed",
            "vina_ligand_pdbqt": str(ligand),
            "vina_pose_pdbqt": str(pose),
        }

    result = svr_affinity_wrapper.run_svr_affinity(
        vina_by_id=vina,
        affinity={
            "enabled": True,
            "model": "svr_fp_pose",
            "project_dir": str(project),
            "script": str(script),
            "svr_model": str(model),
            "svr_meta": str(meta),
            "obabel_bin": "obabel",
        },
        output_dir=tmp_path / "output",
        num_workers=3,
    )

    assert list(result) == ["mol_b", "mol_a", "mol_c"]
    assert calls["prediction"] == 1
