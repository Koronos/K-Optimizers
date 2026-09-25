# Re-medición post-fix de kahan8 (2026-09-24)

La primera re-medición (`bench_definitive.py` sobre 3c1cfc6) salió **contaminada**: wall total 557.9 s frente a ~310 s normales,
UNet fused SR 7.13 ms. Se descartó y sus tablas se borraron.

Bisect con commits intercalados (micro-bench, `abba/micro*.py`, 8 commits × 4 rondas) y el script definitive completo corrido
4 veces en orden ABBA (b427e0c, 3c1cfc6, 3c1cfc6, b427e0c; `abba/<n>_<rev>/`, `nvidia-smi` cada 1 s en `abba/smi_*.csv`, AC, máx 62 °C):
**no hay regresión** entre b427e0c y 3c1cfc6. Perfil de un paso: fused 8 eventos CUDA, 0 syncs, 0 memcpy en todos los commits.

| celda (mediana ms) | b427e0c #1 | 3c1cfc6 #2 | 3c1cfc6 #3 | b427e0c #4 |
|---|---|---|---|---|
| UNet fused SR | 0.540 | 0.587 | 0.605 | 0.547 |
| UNet fused kahan8 | 0.621 | 0.673 | 0.675 | 0.605 |
| UNet foreach SR | 4.780 | 4.171 | 4.768 | 4.075 |
| UNet 4bit fused SR | 0.690 | 0.715 | 0.721 | 0.613 |
| wall total | 317.7 s | 307.3 s | 312.6 s | 306.5 s |

Las cifras UNet fused de `../definitive/step_table.md` (2.66/2.97 ms, IQR bimodal) también estaban contaminadas a ratos y
sobrestiman el paso real ~4–5×; el overhead real de kahan8 en fused UNet es ≈ +10–15 % sobre SR.
Regla: registrar `nvidia-smi` durante toda la corrida y descartarla si el wall total se aleja de ~310 s.
