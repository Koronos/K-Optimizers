# Nekaon a LR bajo: verificación local

La limitación existe: el lookahead puede desaparecer por redondeo en BF16, sin
que se desactive el optimizador base. No hay un corte universal de LR.
MSAM aplica `e = -k * lr * momentum`, limitado por coordenada a
`abs(e) <= k * lr * clip_threshold`, y escribe el desplazamiento con redondeo
al más cercano. El update acumulativo de Adakaon puede usar redondeo estocástico.
Por tanto, que el lookahead sea cero no implica que todos los pesos dejen de aprender.

El changelog ya registra el problema y el aviso de lookahead inerte. Ese aviso
usa una muestra y magnitudes medias, con persistencia de 50 comprobaciones:
es una heurística, no mide diferencias reales de gradientes y no desactiva k.
Su frase de que no puede mover ningún peso es demasiado categórica para un
diagnóstico basado en medias. Ausencia de aviso no demuestra actividad completa.

## Reproducción CUDA

[`nekaon_low_lr_probe.py`](../../benchmarks/nekaon_low_lr_probe.py) ejecuta 20 pasos
con gradientes sintéticos constantes y separa el cambio de pesos base del cambio
`eval() -> train()` (lookahead). Usa k=1.5, momentum4bit, sin decay, cautious ni GC.

Con pesos BF16 inicialmente 0.01, LR1e-5 o 1e-6 produce **0% de coordenadas
desplazadas por lookahead**, aunque cambió el 95.7%/24.6% de los pesos base.
Con LR1e-4, se desplaza el 99.2%. Con pesos inicialmente 1, incluso LR1e-4
produce lookahead cero. FP32 representa el desplazamiento en el 99.2% en esos
casos, pero representarlo no garantiza una diferencia útil del gradiente.
[Resultados completos](../../benchmarks/nekaon_low_lr_probe.json).

## Capacidad de representación en el checkpoint Anima

Se analizaron los 34,635,776 parámetros LoRA optimizados del checkpoint Nekaon
seed43/step200; se excluyó llm_adapter congelado. Se probó `w ± 1.5*lr` y se
redondeó a BF16. Es un límite superior del porcentaje que podría moverse, usando
el desplazamiento máximo; no usa el momentum real. Los LR alternativos son
contrafactuales sobre ese checkpoint, no entrenamientos nuevos.

| LR | Máximo % movible A | Máximo % movible B |
|---|---:|---:|
| 1e-4 | 99.981 | 100.000 |
| 1e-5 | 21.903 | 98.793 |
| 1e-6 | 2.732 | 29.620 |

[Script](../../benchmarks/nekaon_bf16_capacity.py) y
[datos](../../benchmarks/nekaon_anima_bf16_capacity.json).

No hay evidencia para afirmar que Nekaon estuvo totalmente desactivado en nuestros
ensayos LR1e-4. Sí hay una limitación clara al trasladar esos pesos a LR menores,
especialmente en A. Incluso a LR1e-4 el momentum real puede ser menor que el
máximo y dejar muchas coordenadas intactas; falta medir el desplazamiento real
y su efecto en los gradientes durante Anima.

La prueba siguiente apropiada es comparar k=0/1.5 a LR bajos con parámetros LoRA
BF16 y FP32, conservando el modelo base BF16 y el LR. Esto distingue precisión
de utilidad del lookahead. No subir LR o k automáticamente: altera la dinámica.
No restaurar SR independiente en la ida/vuelta: el changelog documenta deriva
por las dos muestras aleatorias que no se cancelan.
