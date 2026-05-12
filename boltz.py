#!/usr/bin/env python
# batch_boltz.py

import subprocess, shutil, yaml, csv, json, pathlib, os, time
import pandas as pd
import traceback

# --- Rutas fijas ---
TEMPLATE = pathlib.Path(
    "/LUSTRE/users/pvaras/boltz/examples/affinity.yaml"
)
SCRATCH_ROOT = pathlib.Path(
    "/LUSTRE/users/pvaras/boltz/"
)
TMP_YAML = "/tmp/affinity_tmp.yaml"
RESULTS_DIR = SCRATCH_ROOT / "boltz_results_affinity_tmp"
JSON_PATH = RESULTS_DIR / "predictions/affinity_tmp/affinity_affinity_tmp.json"
CSV_PATH = pathlib.Path("boltz_results_bace_de_novo.csv")

# --- Cargar SMILES ---
df = pd.read_csv("molecules_filtered_strict.csv")
smiles_list = df["SMILES"].dropna().astype(str).tolist()

# --- Preparar CSV ---
write_header = not CSV_PATH.exists()
with CSV_PATH.open("a", newline="") as fh:
    writer = csv.writer(fh)
    if write_header:
        writer.writerow([
            "SMILES",
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
            "affinity_pred_value2",
            "affinity_probability_binary2",
            "error"
        ])

# --- Bucle principal ---
for smiles in smiles_list:

    print(f"\n▶ SMILES: {smiles}")

    try:
        # 1) Generar YAML
        data = yaml.safe_load(TEMPLATE.read_text())
        data["sequences"][1]["ligand"]["smiles"] = smiles

        with open(TMP_YAML, "w") as f:
            yaml.safe_dump(data, f)

        # 2) Ejecutar Boltz
        subprocess.run(
            ["boltz", "predict", TMP_YAML],
            check=True,
        )

        # 3) Esperar JSON
        for _ in range(20):
            if JSON_PATH.exists():
                break
            time.sleep(0.5)
        else:
            raise FileNotFoundError(f"No se encontró {JSON_PATH}")

        # 4) Leer métricas
        with JSON_PATH.open() as jf:
            metrics = json.load(jf)

        row = [
            smiles,
            metrics.get("affinity_pred_value"),
            metrics.get("affinity_probability_binary"),
            metrics.get("affinity_pred_value1"),
            metrics.get("affinity_probability_binary1"),
            metrics.get("affinity_pred_value2"),
            metrics.get("affinity_probability_binary2"),
            ""
        ]

        print("  ✓ hecho correctamente")

    except Exception as e:
        print("  ✗ ERROR con esta molécula")
        print("  ", str(e))

        # Opcional: imprimir traceback completo
        traceback.print_exc()

        row = [
            smiles,
            None, None, None, None, None, None,
            str(e)
        ]

    finally:
        # Guardar fila (aunque haya error)
        with CSV_PATH.open("a", newline="") as fh:
            csv.writer(fh).writerow(row)

        # Limpiar resultados para siguiente iteración
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
        if os.path.exists(JSON_PATH):
            try:
                os.remove(JSON_PATH)
            except:
                pass

print("\n🏁 Predicciones terminadas.")
print("CSV guardado en:", CSV_PATH.resolve())
