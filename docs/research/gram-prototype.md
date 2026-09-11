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

Pendiente: integración con pares LoRA explícitos de Rengu, comprobación del
factor alpha/r del adapter, barrido de LR/damping, comparación Anima con
inicialización idéntica y evaluación perceptual con prompts y ruido fijos.
No admite gradient-release por parámetro sin una adaptación para esperar ambos
gradientes de cada par.
