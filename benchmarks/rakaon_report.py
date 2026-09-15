"""Render measured endpoints and time-to-quality without averaging away failures."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parents[1]
    study = json.loads((root / "benchmarks/rakaon_time_to_quality.json").read_text())
    perf = json.loads((root / "benchmarks/rakaon_performance.json").read_text())
    groups = defaultdict(list)
    for row in study["runs"]:
        groups[row["arm"]].append(row)
    lines = ["# Rakaon: resultados medidos", "",
             "C=128, 2000 pasos, semillas 43/44, LR constante, evaluación cada 200 pasos.",
             "Datos completos en `benchmarks/rakaon_time_to_quality.json` y `rakaon_performance.json`.",
             "Este proxy no demuestra calidad perceptual en LoRA/fine-tuning real.", "",
             "## Endpoints (promedio de todas las semillas)", "",
             "| Optimizer | LR | Train | Validation | Gap | Estado lógico B/p |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, rows in groups.items():
        lines.append(f"| {arm} | {rows[0]['lr']:g} | {mean(r['tr'] for r in rows):.6f} | "
                     f"{mean(r['te'] for r in rows):.6f} | {mean(r['gap'] for r in rows):.6f} | {rows[0]['bpp']:.6f} |")
    lines += ["", "## Tiempo hasta calidad conjunta", "",
              "Segundos de entrenamiento reales (forward + backward + step), incluyendo calentamiento y excluyendo evaluación.",
              "Cada celda muestra todas las semillas. `—` significa no alcanzado en 2000 pasos; no se extrapola ni se promedian sólo los éxitos.",
              "Se muestra el primer checkpoint observado; una mejora aislada puede revertirse. Los JSON registran también dos checkpoints consecutivos.", "",
              "| Optimizer | Val < .10, abs(gap) < .007 | Val < .09, abs(gap) < .007 | Val < .08, abs(gap) < .007 | Val < .07, abs(gap) < .007 |",
              "|---|---|---|---|---|"]
    for arm, rows in groups.items():
        cells = []
        for target in (.10, .09, .08, .07):
            entries = []
            for row in rows:
                hit = row["targets"][f"val<{target:g}_abs_gap<0.007"]["first"]
                value = f"{hit['training_seconds']:.1f}s / {hit['step']} pasos" if hit else "—"
                entries.append(f"s{row['seed']}: {value}")
            cells.append("; ".join(entries))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Coste aislado del step", "",
              "Mediana de 40 steps tras 10 de calentamiento, GPU sin otro experimento de esta campaña.",
              "Memoria persistente CUDA incluye asignaciones/cache del optimizador; pico extra excluye activaciones del modelo.",
              "AdamW bf16 almacena sus momentos en bf16 en este control: no equivale a AdamW con master weights fp32.", "",
              "| Optimizer | Régimen | Dtype | ms/step | Persistente CUDA KiB | Pico extra KiB |",
              "|---|---|---|---:|---:|---:|"]
    for row in perf["runs"]:
        allocated = row.get("persistent_allocated_bytes")
        allocation = f"{allocated / 1024:.1f}" if allocated is not None else "no medido"
        lines.append(f"| {row['optimizer']} | {row['regime']} | {row['dtype'].removeprefix('torch.')} | "
                     f"{row['median_step_ms']:.3f} | {allocation} | {row['step_extra_peak_bytes']/1024:.1f} |")
    lines += ["", "## Alcance de las conclusiones", "",
              "No se demuestra superioridad universal. El orden de ejecuciones no fue aleatorizado; temperatura, carga del sistema y variación CUDA pueden afectar tiempos.",
              "Dos semillas no bastan para estimar con precisión una mejora pequeña. No mezclar estos endpoints con los del protocolo final-only.",
              "Priorizar tiempo hasta calidad dentro del presupuesto de memoria; si no se alcanza el objetivo, un step barato no cuenta como victoria.", ""]
    (root / "docs/research/rakaon-results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
