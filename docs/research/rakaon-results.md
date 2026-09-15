# Rakaon: resultados medidos

C=128, 2000 pasos, semillas 43/44, LR constante, evaluación cada 200 pasos.
Datos completos en `benchmarks/rakaon_time_to_quality.json` y `rakaon_performance.json`.
Este proxy no demuestra calidad perceptual en LoRA/fine-tuning real.

## Endpoints (promedio de todas las semillas)

| Optimizer | LR | Train | Validation | Gap | Estado lógico B/p |
|---|---:|---:|---:|---:|---:|
| Rakaon-1 | 0.0024 | 0.081199 | 0.089136 | 0.007938 | 0.000065 |
| AdamW | 0.0012 | 0.072675 | 0.082941 | 0.010267 | 8.000065 |
| Adakaon | 0.0012 | 0.078521 | 0.085852 | 0.007331 | 0.032095 |
| Nekaon | 0.0012 | 0.073267 | 0.080192 | 0.006925 | 0.563348 |

## Tiempo hasta calidad conjunta

Segundos de entrenamiento reales (forward + backward + step), incluyendo calentamiento y excluyendo evaluación.
Cada celda muestra todas las semillas. `—` significa no alcanzado en 2000 pasos; no se extrapola ni se promedian sólo los éxitos.
Se muestra el primer checkpoint observado; una mejora aislada puede revertirse. Los JSON registran también dos checkpoints consecutivos.

| Optimizer | Val < .10, abs(gap) < .007 | Val < .09, abs(gap) < .007 | Val < .08, abs(gap) < .007 | Val < .07, abs(gap) < .007 |
|---|---|---|---|---|
| Rakaon-1 | s43: 57.0s / 1800 pasos; s44: 66.2s / 1800 pasos | s43: —; s44: — | s43: —; s44: — | s43: —; s44: — |
| AdamW | s43: 35.2s / 1000 pasos; s44: 37.8s / 1400 pasos | s43: —; s44: — | s43: —; s44: — | s43: —; s44: — |
| Adakaon | s43: 60.1s / 2000 pasos; s44: 62.7s / 1400 pasos | s43: —; s44: — | s43: —; s44: — | s43: —; s44: — |
| Nekaon | s43: 37.5s / 800 pasos; s44: 38.8s / 800 pasos | s43: 89.6s / 1800 pasos; s44: 90.4s / 2000 pasos | s43: —; s44: 90.4s / 2000 pasos | s43: —; s44: — |

## Coste aislado del step

Mediana de 40 steps tras 10 de calentamiento, GPU sin otro experimento de esta campaña.
Memoria persistente CUDA incluye asignaciones/cache del optimizador; pico extra excluye activaciones del modelo.
AdamW bf16 almacena sus momentos en bf16 en este control: no equivale a AdamW con master weights fp32.

| Optimizer | Régimen | Dtype | ms/step | Persistente CUDA KiB | Pico extra KiB |
|---|---|---|---:|---:|---:|
| AdamW-fused | unet128 | float32 | 0.564 | 22289.0 | 0.0 |
| Adakaon | unet128 | float32 | 7.314 | 92.0 | 13885.5 |
| Nekaon | unet128 | float32 | 20.488 | 1606.5 | 29756.5 |
| Rakaon-0.1 | unet128 | float32 | 13.399 | 92.0 | 9474.0 |
| Rakaon-1 | unet128 | float32 | 12.243 | 23.0 | 7171.5 |
| AdamW-fused | adapter_bag | float32 | 1.029 | 768.0 | 0.0 |
| Adakaon | adapter_bag | float32 | 3.503 | 512.0 | 916.0 |
| Nekaon | adapter_bag | float32 | 6.138 | 1036.0 | 1746.0 |
| Rakaon-0.1 | adapter_bag | float32 | 204.654 | 512.0 | 3.5 |
| Rakaon-1 | adapter_bag | float32 | 2.693 | 256.0 | 520.0 |
| AdamW-fused | unet128 | bfloat16 | 0.299 | 11039.0 | 0.0 |
| Adakaon | unet128 | bfloat16 | 9.671 | 92.0 | 16188.5 |
| Nekaon | unet128 | bfloat16 | 22.565 | 1606.5 | 29756.5 |
| Rakaon-0.1 | unet128 | bfloat16 | 23.945 | 92.0 | 11522.0 |
| Rakaon-1 | unet128 | bfloat16 | 11.766 | 23.0 | 9218.0 |
| AdamW-fused | adapter_bag | bfloat16 | 1.830 | 768.0 | 0.0 |
| Adakaon | adapter_bag | bfloat16 | 4.984 | 512.0 | 1042.0 |
| Nekaon | adapter_bag | bfloat16 | 9.000 | 1036.0 | 1746.0 |
| Rakaon-0.1 | adapter_bag | bfloat16 | 261.342 | 512.0 | 4.0 |
| Rakaon-1 | adapter_bag | bfloat16 | 10.812 | 256.0 | 648.0 |

## Alcance de las conclusiones

No se demuestra superioridad universal. El orden de ejecuciones no fue aleatorizado; temperatura, carga del sistema y variación CUDA pueden afectar tiempos.
Dos semillas no bastan para estimar con precisión una mejora pequeña. No mezclar estos endpoints con los del protocolo final-only.
Priorizar tiempo hasta calidad dentro del presupuesto de memoria; si no se alcanza el objetivo, un step barato no cuenta como victoria.
