## Task Spec: Pipeline de Drug Repurposing (Sin implementar Vina aun)

### Objetivo
Construir la base de un pipeline unico para reposicionamiento de farmacos que procese un CSV y orqueste etapas de preparacion, docking, scoring con Boltz y ranking final.

### Restricciones obligatorias
1. Mantener src/boltz intacto.
2. No modificar codigo interno de Boltz salvo necesidad estricta y justificada.
3. Reutilizar boltz.py como wrapper principal de integracion Boltz.
4. No implementar Vina todavia.
5. Preparar la arquitectura para integrar Vina despues de una decision tecnica.
6. CSV de entrada sin columna name obligatoria.

### Estructura objetivo

```text
DRUG-REPURPOSING/
├── boltz.py
├── src/
│   ├── boltz/
│   └── repurposing_pipeline/
│       ├── __init__.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── io.py
│       ├── ligand_prep.py
│       ├── ranking.py
│       ├── wrappers/
│       │   ├── __init__.py
│       │   ├── boltz_wrapper.py
│       │   ├── rdkit_wrapper.py
│       │   └── meeko_wrapper.py
│       └── parsers/
│           ├── __init__.py
│           └── boltz_parser.py
└── runs/
    └── run_001/
        ├── prepared/
        ├── vina_results/
        ├── boltz_results/
        ├── output/
        └── logs/
```

Nota: vina_wrapper.py y vina_parser.py se reservan para la fase posterior de implementacion.

### Contrato de entrada y salida

Entrada CSV:
1. Columnas minimas: molecule_id, smiles
2. Opcionales: external_id, metadata

Salida por corrida:
1. runs/run_001/prepared
2. runs/run_001/vina_results
3. runs/run_001/boltz_results
4. runs/run_001/output
5. runs/run_001/logs

### Fase 0 (ahora): Evaluar opciones para implementar Vina

Generar un documento de decision tecnica con opciones de integracion, sin codificar Vina aun.

Opciones a comparar:
1. CLI nativo de AutoDock Vina via subprocess.
2. Python bindings de vina (si aplican a entorno objetivo).
3. Flujo con Meeko para preparacion + Vina CLI para docking.
4. Backends alternativos compatibles (smina o gnina) manteniendo interfaz comun.

Criterios de evaluacion:
1. Facilidad de instalacion en local y HPC.
2. Reproducibilidad (seed, versionado, formato de salida).
3. Rendimiento y paralelizacion.
4. Robustez operativa (timeouts, retries, errores por molecula).
5. Facilidad de parseo de poses y scores.
6. Mantenibilidad en el tiempo.

Entregable de Fase 0:
1. Documento breve en docs/vina_options.md con recomendacion final.
2. Interfaz abstracta propuesta para docking (sin backend implementado).

### Fase 1: Skeleton funcional sin Vina

Implementar solo esto:
1. src/repurposing_pipeline/io.py
2. src/repurposing_pipeline/ligand_prep.py
3. src/repurposing_pipeline/ranking.py
4. src/repurposing_pipeline/pipeline.py
5. src/repurposing_pipeline/cli.py
6. src/repurposing_pipeline/wrappers/boltz_wrapper.py
7. src/repurposing_pipeline/wrappers/rdkit_wrapper.py
8. src/repurposing_pipeline/wrappers/meeko_wrapper.py
9. src/repurposing_pipeline/parsers/boltz_parser.py

Comportamiento esperado en Fase 1:
1. Leer CSV y validar esquema.
2. Crear estructura runs/run_001/*.
3. Preparar ligandos (hasta donde permitan dependencias instaladas).
4. Ejecutar Boltz via boltz.py y parsear resultados.
5. Generar output tabular con estado por molecula.
6. Dejar placeholders claros para etapa de docking.

### Fase 2 (despues de decision): Integrar Vina

Solo cuando se apruebe la opcion de Fase 0:
1. Crear vina_wrapper.py
2. Crear vina_parser.py
3. Conectar docking real en pipeline.py

### Checklist de implementacion

1. Crear paquete src/repurposing_pipeline con __init__.py.
2. Definir modelos de datos minimos para registro por molecula.
3. Implementar manejo de run_id y rutas runs/run_XXX.
4. Implementar logging por molecula en runs/run_XXX/logs.
5. Implementar checkpoints de avance por etapa.
6. Integrar llamada a boltz.py sin tocar src/boltz.
7. Exportar resultados a runs/run_XXX/output/final_results.csv.
8. Incluir seccion TODO Vina claramente delimitada.

### Criterios de aceptacion

1. src/boltz permanece sin cambios.
2. El pipeline corre con CSV sin columna name.
3. Se crea correctamente la jerarquia runs/run_001/*.
4. Se produce final_results.csv con estado por molecula.
5. Existe docs/vina_options.md con comparativa y recomendacion.
6. No existe implementacion funcional de Vina en esta fase.

### Verificacion

1. Ejecutar con 5 a 10 moleculas y confirmar que no aborta corrida completa por fallos individuales.
2. Confirmar logs por molecula y log general de corrida.
3. Interrumpir y reanudar usando checkpoints.
4. Verificar salida final en runs/run_001/output.
5. Revisar documento de opciones Vina antes de aprobar la Fase 2.
