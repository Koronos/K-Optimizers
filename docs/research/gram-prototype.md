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

Pendiente: resultados del barrido, comparación con inicialización idéntica y
evaluación perceptual con prompts y ruido fijos, incluida conservación de detalles.
No admite gradient-release por parámetro sin una adaptación para esperar ambos
gradientes de cada par.
