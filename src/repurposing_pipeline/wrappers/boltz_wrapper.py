"""Wrapper to integrate existing boltz.py script without modifying src/boltz."""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

from repurposing_pipeline.parsers.boltz_parser import parse_boltz_metrics, parse_boltz_metrics_from_csv


def _boltz_base_command(
    boltz_conda_env: str | None,
    boltz_python_executable: str | None,
) -> list[str]:
    if boltz_python_executable:
        return [boltz_python_executable, "-m", "boltz.main"]
    if boltz_conda_env:
        conda_bin = shutil.which("conda")
        if conda_bin is None:
            raise FileNotFoundError("conda executable not found")
        return [conda_bin, "run", "-n", boltz_conda_env, "boltz"]
    return [sys.executable, "-m", "boltz.main"]


def run_boltz_parallel(
    repo_root: Path,
    molecules: list[tuple[str, str]],
    logs_dir: Path,
    boltz_results_dir: Path,
    template_path: Path | None,
    gpu_ids: list[int],
    boltz_conda_env: str | None = None,
    boltz_python_executable: str | None = None,
    min_gpu_memory_mb: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run one multi-ligand Boltz batch per GPU and merge by molecule id.

    A batch lets Boltz load each structure and affinity model once per GPU.
    Round-robin shards are deterministic and have isolated inputs, outputs,
    temporary files and logs. Existing valid JSON outputs are reused.
    """
    if not molecules:
        return {}
    if not gpu_ids:
        raise ValueError("At least one Boltz GPU id is required")
    if template_path is None or not template_path.exists():
        return {
            molecule_id: {
                "boltz_status": "failed",
                "error": "Boltz template is missing",
            }
            for molecule_id, _ in molecules
        }

    logs_dir.mkdir(parents=True, exist_ok=True)
    boltz_results_dir.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    shards: list[list[tuple[str, str]]] = [[] for _ in gpu_ids]
    for index, molecule in enumerate(molecules):
        shards[index % len(gpu_ids)].append(molecule)

    free_memory: dict[int, int] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        probe = subprocess.run(
            [nvidia_smi, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if probe.returncode == 0:
            for line in probe.stdout.splitlines():
                try:
                    gpu_text, memory_text = line.split(",", 1)
                    free_memory[int(gpu_text.strip())] = int(memory_text.strip())
                except ValueError:
                    continue

    def run_shard(worker_index: int) -> dict[str, dict[str, Any]]:
        shard_started = time.perf_counter()
        gpu_id = gpu_ids[worker_index]
        shard = shards[worker_index]
        if not shard:
            return {}
        worker_dir = boltz_results_dir / f"worker_{worker_index}_gpu_{gpu_id}"
        inputs_dir = worker_dir / "inputs"
        output_root = worker_dir / "output"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"boltz_worker_{worker_index}_gpu_{gpu_id}.log"
        available_mb = free_memory.get(gpu_id)
        if available_mb is not None and available_mb < max(0, int(min_gpu_memory_mb)):
            return {
                molecule_id: {
                    "boltz_status": "failed",
                    "error": (
                        f"GPU {gpu_id} has {available_mb} MiB free; "
                        f"requires at least {min_gpu_memory_mb} MiB"
                    ),
                    "boltz_gpu_id": gpu_id,
                    "boltz_worker": worker_index,
                }
                for molecule_id, _ in shard
            }

        name_by_id: dict[str, str] = {}
        expected_input_paths: set[Path] = set()
        for molecule_id, smiles in shard:
            # File names are controlled locally even when molecule IDs contain
            # whitespace or path separators.
            input_name = "ligand_" + hashlib.sha256(molecule_id.encode("utf-8")).hexdigest()[:16]
            name_by_id[molecule_id] = input_name
            payload = yaml.safe_load(yaml.safe_dump(template))
            payload["sequences"][1]["ligand"]["smiles"] = smiles
            input_path = inputs_dir / f"{input_name}.yaml"
            expected_input_paths.add(input_path)
            input_path.write_text(
                yaml.safe_dump(payload), encoding="utf-8"
            )
        for stale_input in inputs_dir.glob("*.yaml"):
            if stale_input not in expected_input_paths:
                stale_input.unlink()

        result_dir = output_root / f"boltz_results_{inputs_dir.stem}"
        predictions_dir = result_dir / "predictions"
        missing = []
        for molecule_id, _ in shard:
            name = name_by_id[molecule_id]
            json_path = predictions_dir / name / f"affinity_{name}.json"
            if not json_path.exists():
                missing.append(molecule_id)

        process_returncode = 0
        process_error = ""
        if missing:
            try:
                cmd = _boltz_base_command(boltz_conda_env, boltz_python_executable)
                cmd.extend(["predict", str(inputs_dir), "--out_dir", str(output_root)])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"[BOLTZ] physical_gpu={gpu_id} visible_device=0\n")
                    log_file.write(f"[BOLTZ] free_gpu_memory_mb={available_mb}\n")
                    log_file.write("[BOLTZ] command=" + " ".join(cmd) + "\n")
                    completed = subprocess.run(
                        cmd, cwd=str(repo_root), stdout=log_file, stderr=log_file,
                        check=False, text=True, env=env,
                    )
                process_returncode = completed.returncode
            except Exception as exc:
                process_returncode = -1
                process_error = str(exc)

        shard_results: dict[str, dict[str, Any]] = {}
        for molecule_id, _ in shard:
            name = name_by_id[molecule_id]
            json_path = predictions_dir / name / f"affinity_{name}.json"
            try:
                metrics = parse_boltz_metrics(json_path)
                shard_results[molecule_id] = {
                    **metrics,
                    "boltz_status": "completed",
                    "boltz_gpu_id": gpu_id,
                    "boltz_worker": worker_index,
                    "boltz_log": str(log_path),
                    "boltz_output_dir": str(result_dir),
                }
            except Exception as exc:
                detail = process_error or f"Boltz exit code {process_returncode}; {exc}"
                shard_results[molecule_id] = {
                    "boltz_status": "failed",
                    "error": detail,
                    "boltz_gpu_id": gpu_id,
                    "boltz_worker": worker_index,
                    "boltz_log": str(log_path),
                    "boltz_output_dir": str(result_dir),
                }
        elapsed = time.perf_counter() - shard_started
        with log_path.open("a", encoding="utf-8") as log_file:
            throughput = (len(shard) * 60.0 / elapsed) if elapsed > 0 else 0.0
            log_file.write(
                f"[BOLTZ] wall_seconds={elapsed:.3f} molecules={len(shard)} "
                f"molecules_per_minute={throughput:.4f}\n"
            )
        return shard_results

    merged: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(molecules))) as pool:
        futures = {
            pool.submit(run_shard, index): index
            for index, shard in enumerate(shards)
            if shard
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            try:
                merged.update(future.result())
            except Exception as exc:
                gpu_id = gpu_ids[worker_index]
                for molecule_id, _ in shards[worker_index]:
                    merged[molecule_id] = {
                        "boltz_status": "failed",
                        "error": f"Boltz worker {worker_index} failed: {exc}",
                        "boltz_gpu_id": gpu_id,
                        "boltz_worker": worker_index,
                    }
    return merged


def run_boltz_with_existing_wrapper(
    repo_root: Path,
    molecule_id: str,
    smiles: str,
    run_boltz: bool,
    logs_dir: Path,
    boltz_results_dir: Path,
    template_path: Path | None = None,
    boltz_conda_env: str | None = None,
    boltz_python_executable: str | None = None,
) -> dict[str, Any]:
    """Run Boltz via the repository-level boltz.py wrapper.

    This adapter runs boltz.py per molecule through environment variables and
    parses returned metrics into pipeline-normalized fields.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{molecule_id}.log"
    molecule_out_dir = boltz_results_dir / molecule_id
    molecule_out_dir.mkdir(parents=True, exist_ok=True)

    if not run_boltz:
        return {
            "boltz_status": "skipped",
            "boltz_note": "Boltz execution disabled in this run",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    boltz_script = repo_root / "boltz.py"
    if not boltz_script.exists():
        return {
            "boltz_status": "failed",
            "error": f"boltz.py not found at {boltz_script}",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    output_csv_path = molecule_out_dir / "boltz_output.csv"
    output_results_dir = molecule_out_dir / "boltz_results_affinity_tmp"
    output_json_path = output_results_dir / "predictions/affinity_tmp/affinity_affinity_tmp.json"
    env = os.environ.copy()
    env["BOLTZ_SINGLE_SMILES"] = smiles
    env["BOLTZ_OUTPUT_CSV"] = str(output_csv_path)
    env["BOLTZ_JSON_PATH"] = str(output_json_path)
    env["BOLTZ_RESULTS_DIR"] = str(output_results_dir)
    env["BOLTZ_TMP_YAML"] = str(molecule_out_dir / "affinity_tmp.yaml")
    if template_path is not None:
        env["BOLTZ_TEMPLATE_PATH"] = str(template_path)

    if boltz_python_executable:
        boltz_cmd = [boltz_python_executable, str(boltz_script)]
    elif boltz_conda_env:
        conda_bin = shutil.which("conda")
        if conda_bin is None:
            return {
                "boltz_status": "failed",
                "error": "conda executable not found, cannot run Boltz in alternate environment",
                "boltz_log": str(log_path),
                "boltz_output_dir": str(molecule_out_dir),
            }
        boltz_cmd = [conda_bin, "run", "-n", boltz_conda_env, "python", str(boltz_script)]
    else:
        boltz_cmd = [sys.executable, str(boltz_script)]

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("[BOLTZ] Attempting repository wrapper execution via boltz.py\n")
        log_file.write(f"[BOLTZ] molecule_id={molecule_id} smiles={smiles}\n")
        log_file.write("[BOLTZ] command=" + " ".join(boltz_cmd) + "\n")
        process = subprocess.run(
            boltz_cmd,
            cwd=str(repo_root),
            stdout=log_file,
            stderr=log_file,
            check=False,
            text=True,
            env=env,
        )

    if process.returncode != 0:
        return {
            "boltz_status": "failed",
            "error": f"boltz.py exited with code {process.returncode}",
            "boltz_log": str(log_path),
            "boltz_output_dir": str(molecule_out_dir),
        }

    metrics: dict[str, Any]
    try:
        metrics = parse_boltz_metrics(output_json_path)
    except Exception:
        try:
            metrics = parse_boltz_metrics_from_csv(output_csv_path, smiles=smiles)
        except Exception as exc:
            return {
                "boltz_status": "failed",
                "error": f"boltz output parse failed: {exc}",
                "boltz_log": str(log_path),
                "boltz_output_dir": str(molecule_out_dir),
            }

    return {
        **metrics,
        "boltz_status": "completed",
        "boltz_note": "boltz.py executed and metrics parsed",
        "boltz_log": str(log_path),
        "boltz_output_dir": str(molecule_out_dir),
    }
