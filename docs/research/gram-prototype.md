# Referencia Gram emparejada

Implementación experimental en `benchmarks/gram_lora.py`, no exportada como
optimizador recomendado del paquete. Cada grupo contiene explícitamente [A,B]
con A[r,in] y B[out,r]. No se infieren parejas a partir de dimensiones o nombres.

Calcula simultáneamente, con los factores anteriores al step:

```
dA = solve(B.T @ B + damping * I, gradA)
dB = solve(A @ A.T + damping * I, gradB.T).T
A -= lr * dA
B -= lr * dB
```

Damping es una constante absoluta positiva. Permite resolver la inicialización
B=0 pero rompe la invariancia exacta al reescalado que tenía la referencia
ideal sin damping. El forward inicial no cambia. No se añaden momentum,
clipping, regularización de pesos ni normalización del update. El LR no es
directamente comparable al de Rakaon: requiere búsqueda controlada.

Las operaciones son FP32 y la escritura BF16 admite redondeo estocástico con
estado RNG propio. No se guarda un momento denso; sí hay temporales de los
factores y matrices r×r. No se ha medido aún latencia ni pico real de VRAM.
Ambos factores deben tener gradiente o ninguno; una pareja parcial se rechaza
antes de actualizar parámetros.

Ocho pruebas pasaron: referencia independiente FP64 para factores no nulos y
B=0; gradientes sin alterar; pérdida decreciente en un problema matricial
pequeño; reanudación BF16 exacta en CPU/CUDA; gradiente parcial y damping inválido.
La prueba pequeña no es difusión ni demuestra conservación de detalles.

La integración Rengu usa los objetos PEFT para encontrar pares, preserva sus LR
y exige alpha/r=1 en este piloto. Se rechazan parámetros no cubiertos, factores
reutilizados y parejas separadas entre grupos. Pasaron 21 pruebas de prototipo,
agrupación y configuración, incluida CUDA.

El smoke Anima de tres pasos completó 448 pares (34,635,776 parámetros),
con escala LoRA 1, pérdidas 0.116969/0.089265/0.127306 y pico CUDA 4.62 GiB.
Guardó step3 correctamente. Los tiempos activos de los pasos fueron
3.501/1.435/1.261 s; el primero incluye calentamiento. No son evidencia de
convergencia ni un benchmark comparable a los ensayos largos de Pets.

Se inició un barrido secuencial de cuatro corridas Anima/Pets: LR 0.001/0.01
y damping 0.001/0.01, seed43, 200 pasos, evaluación antes de entrenar y cada100,
32 imágenes por split, sin previews. Lanzador `benchmarks/anima/run_gram_screen.py`;
planes, configs, logs y reportes en `tmp/anima-gram43`. Se detiene en el primer
error para preservar el fallo. Es exploración, no confirmación independiente.

## Barrido terminado

Las cuatro corridas finalizaron con checkpoint y evaluación del paso200.
Se verificó coincidencia de modelo, dataset, huella de los adapters y pérdidas
iniciales con las referencias de seed43. Reportes:
[LR0.001](../../benchmarks/anima/gram43_lr0_results.md) y
[LR0.01](../../benchmarks/anima/gram43_lr1_results.md), con JSON completos contiguos.

| LR | Damping | Validation | Gap absoluto | Tiempo activo s | Pico GiB |
|---|---|---:|---:|---:|---:|
| 0.001 | 0.001 | 0.133673 | 0.010835 | 301.0 | 4.624 |
| 0.001 | 0.01 | 0.134747 | 0.010814 | 407.5 | 4.624 |
| 0.01 | 0.001 | 0.133006 | 0.010526 | 469.0 | 4.624 |
| 0.01 | 0.01 | 0.132821 | 0.010490 | 346.1 | 4.624 |

El mejor validation final de Gram queda ligeramente por encima de Nekaon
(0.132597, gap0.009018, 196.0s, 4.652GiB). Frente a Rakaon isotropic
(0.133067, gap0.009335, 181.4s, 4.624GiB), reduce algo validation pero empeora
gap y tiempo observado. No se demuestra ventaja global ni se cumplen los
umbrales originales. La variación temporal entre corridas exige cautela por
temperatura/carga del portátil; no fueron ensayos intercalados de throughput.
Los protocolos tienen distinta frecuencia de evaluación, excluida del tiempo
activo. Un menor cambio del gap respecto al paso0 no sustituye al gap absoluto.

La versión probada es SGD Gram con damping absoluto y solves por pareja,
sin momentum ni optimización de kernels; no equivale al método Riemannion completo.
Pendiente: evaluación perceptual con prompts y ruido fijos, incluida conservación
de detalles. Estas pérdidas no permiten concluir si los detalles mejoraron.
No admite gradient-release por parámetro sin una adaptación para esperar ambos
gradientes de cada par.
