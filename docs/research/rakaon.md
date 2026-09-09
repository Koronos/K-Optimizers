# Rakaon: experimento de precondicionamiento regularizado

Estado: candidato experimental. No se ha demostrado superioridad ni calidad visual
en difusión real. La implementación y el protocolo permiten refutar la hipótesis.

## Objetivo y criterio de decisión

Priorizar pérdida de validación, vigilando el gap y el coste. Un gap pequeño con
ambas pérdidas altas es subentrenamiento, no éxito. Registrar por separado train,
validation y `gap = validation - train`; no seleccionar por train loss solamente.
El objetivo aspiracional es validation < 0.07 y |gap| < 0.007. Si el 0.07 original
se interpreta como train loss, reportamos también esa cifra sin confundir las dos.
Estos umbrales dependen del modelo, dataset, ruido, timesteps y ponderación de loss.

Se conserva LR constante: no hay warmup, horizonte prefijado ni scheduler externo.
La corrección de sesgo de los momentos depende del contador guardado. Un scheduler
cosine también puede reanudarse guardando su estado; el problema distinto es cambiar
el horizonte y, por tanto, la trayectoria de LR al extender un entrenamiento.

**Velocidad principal: tiempo real hasta una calidad común**, no menor latencia por
step. El estudio `rakaon_time_to_quality.json` evalúa cada 200 pasos y guarda pasos,
segundos de entrenamiento y segundos de pared para cada checkpoint. Se buscan cruces
de validation < .10/.09/.08/.077/.07, tanto sin restricción de gap como con
|gap| < .007. Se registra el primer cruce y el primero de dos checkpoints consecutivos
que cumplen la condición. Un objetivo no alcanzado se guarda como null: no se estima
cuándo se alcanzaría, ni se sustituye por el mejor loss final.

Los segundos de entrenamiento incluyen forward, backward, step y calentamiento;
excluyen evaluación. Los segundos de pared incluyen evaluación. Ambos empiezan después
de construir modelo/optimizador/dataset. La resolución temporal es de 200 pasos; no
se interpola. Dos checkpoints consecutivos no prueban estabilidad indefinida.

## Mecanismo

Para cada matriz (convoluciones se aplanan como `[out, resto]`):

1. EMA fp32 de las medias por fila y columna de `g² + eps`.
2. Reconstrucción factorizada `V = row[:,None] * col[None,:] / mean(row)`.
3. Regularización `V_s = ((1-s) V + s mean(V)) / (1-beta2**step)`.
4. Dirección `u = g / sqrt(V_s)`, con clipping RMS de `u` a 1 por defecto.
5. `p -= lr * (u + weight_decay * p)`; escritura bf16 con redondeo estocástico.

No hay momentum, copia maestra de pesos, EMA de pesos ni segundo forward/backward.
Los vectores usan varianza densa cuando `s < 1`.

La hipótesis: reducir la anisotropía extrema del precondicionador puede limitar
actualizaciones amplificadas en coordenadas de poca varianza sin otro estado denso.
No implica una garantía de generalización. El fundamento de factorización y clipping
es [Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html). La motivación de
mejorar estabilidad de estados comprimidos tiene precedentes como
[CAME](https://arxiv.org/abs/2307.02047); Rakaon no implementa CAME ni hereda sus resultados.

En `s=1`, la ecuación se reduce a normalización RMS por tensor. No hace falta
reconstruir ni guardar filas/columnas: basta un escalar fp32 por tensor. Se agrupan
tensores de igual forma/dispositivo/dtype, con presupuesto de 262144 elementos por
grupo de trabajo; un tensor individual mayor se procesa solo. No se afirma novedad
científica de este extremo isotrópico.

Como variante experimental de `s=1`, `block_size=64`, `256` o `1024` guarda un
valor fp32 por bloque contiguo del tensor aplanado. El último bloque usa su longitud
real, sin diluir su media con padding; el clipping se calcula como RMS global del
tensor después de aplicar las escalas de bloque. Los bloques son particiones
estadísticas y no representan grupos semánticos de la arquitectura. Esta variante
reduce el estado lógico aproximadamente a 4 bytes por bloque, pero puede añadir
temporales por el padding del último bloque y no constituye una geometría semántica.

## Uso y límites

```python
from kaon import Rakaon

optimizer = Rakaon(model.parameters(), lr=1e-4, shrinkage=0.1)
# shrinkage=1.0: memoria mínima, dinámica isotrópica distinta; requiere evaluar calidad.
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

El LR del ejemplo es sólo un punto de partida; los LRs del proxy no son recetas
para SDXL/FLUX. `shrinkage` en [0,1) se puede cambiar; cambiar hacia/desde 1 después
de inicializar estado se rechaza porque cambia su representación.

Parámetros reales y gradientes densos. El cálculo del optimizador es fp32; el camino
de baja precisión validado es bf16 con redondeo estocástico. No se ha validado fp16,
entrenamiento distribuido, CUDA graphs ni diferenciación a través del step.
Con AMP, usar `GradScaler.step` cuando corresponda para omitir pasos no finitos.

Guardar modelo y optimizer juntos, junto con RNG global, posición del dataloader y
estado de entrenamiento. El optimizador guarda su propio flujo RNG de redondeo y
restaura las estadísticas fp32 sin pasarlas por bf16. No requiere `opt.eval/train`.

Estado tensorial: `4*(R+C)` bytes por matriz en modo factorizado; `4*N` por vector.
Modo isotrópico: 4 bytes por tensor. Estas cuentas excluyen metadatos Python y el
pequeño estado RNG; no son memoria total de entrenamiento. Hay temporales densos
por tensor o grupo, además de pesos, gradientes y activaciones del modelo.
La memoria asignada real puede ser mayor: CUDA redondea asignaciones pequeñas.
`persistent_allocated_bytes` mide el incremento de memoria CUDA activa desde antes
de construir el optimizador hasta después de calentarlo, incluyendo caches persistentes;
`step_extra_peak_bytes` mide los temporales por encima de esa base. No confundir
los 4 bytes lógicos de un escalar con una asignación física de sólo 4 bytes.

## Protocolo reproducible

Ejecutar desde la raíz del worktree, con PyTorch y CUDA:

```bash
PYTHONPATH=src python -m pytest tests/test_rakaon.py
PYTHONPATH=src python benchmarks/rakaon_study.py
PYTHONPATH=src python benchmarks/rakaon_study.py --steps 2000 --channels 128 --seeds 43 44 --lrs .0012 --arms Adakaon Nekaon Rakaon-0 Rakaon-0.1 --output benchmarks/rakaon_confirm.json
PYTHONPATH=src python benchmarks/rakaon_performance.py
PYTHONPATH=src python benchmarks/rakaon_study.py --steps 2000 --channels 128 --seeds 43 44 --lrs .0012 --arm-lr Rakaon-1=.0024 --arms Rakaon-1 AdamW Adakaon Nekaon --eval-every 200 --output benchmarks/rakaon_time_to_quality.json
```

El estudio reutiliza dataset y U-Net DDPM del repositorio. El screening usa C=32,
400 pasos, semilla 42 y tres LRs iguales para cada candidato. La confirmación usa
C=128, 2000 pasos y semillas 43/44. Se mantienen batch=8, currículo 32/48/64 y
evaluación con ruido/timesteps congelados. Las semillas adicionales siguen usando
el mismo conjunto de validación: no constituyen un test externo independiente.

Adakaon sin momentum usa sus defaults excepto beta1=0 y cautious=False. Nekaon usa
beta1=.5, k=1.5, wd=.3 y momentum 4bit. Son recetas de referencia, no una ablación
de un único mecanismo: Rakaon también introduce corrección de sesgo. La ablación
es comparar Rakaon s=0 con s=.1/.5/1 al mismo LR y semilla.

No mezclar los resultados con rankings históricos de otra huella de dataset.
Los tiempos de entrenamiento son exploratorios; usar la medición aislada para
coste del optimizer.step. La bolsa de adaptadores mide lanzamientos y memoria,
no convergencia de LoRA. Estos JSON del proxy no entrenan modelos preentrenados
con imágenes reales ni miden KID, adherencia al prompt o memorización.

Los JSON guardan todas las ejecuciones, incluidas las que no favorecen al candidato.
El screening inicial precede a la compactación/agrupación del modo isotrópico; sus
cifras de memoria y velocidad de s=1 pertenecen a la implementación inicial.
El estudio largo s<1 conserva las mismas ecuaciones. La medición de rendimiento
usa la implementación final. Los nuevos estudios guardan el hash del código.

El modo isotrópico usa .0024 en el estudio temporal porque fue su mejor LR del
screening; las referencias usan .0012. AdamW es una referencia adicional con un
solo LR, no un AdamW exhaustivamente ajustado. El estudio temporal se añadió
después de la primera confirmación; reutiliza las semillas 43/44 y por tanto tampoco
es un test ciego. Su propósito es comparar curvas y tiempo, sin inventar una medida
de convergencia a partir de los endpoints anteriores.

El screening de bloques (`benchmarks/rakaon_blocks_screen.json`) usa C=64, 400 pasos,
semilla 42 y los tres LRs del screening. En ese protocolo ninguna variante de
`block_size=64/256/1024` vence a la mejor referencia Adakaon o Nekaon en validation;
los bloques sí reducen el estado lógico hasta aproximadamente 0.004 B/parámetro.
No se debe extrapolar este resultado a los screenings C=32 o C=128, que no se
ejecutaron para esta variante.

## Piloto Anima con imágenes reales

El protocolo de [Anima/Pets](../../benchmarks/anima/README.md) amplía la integración
a LoRA rank16 sobre Anima, BF16 y una GPU de 8 GB. La calibración de 200 pasos
terminó sin errores: validation pasó de .137754 a .129079 y train evaluado de
.150594 a .139225. Se conserva en `benchmarks/anima/calibration_results.json`.
No es una comparación emparejada: precede al fingerprint de inicialización.

El gap inicial ya era negativo (.012840 en magnitud). Por ello se reporta también
el cambio de gap respecto a step0, sin convertirlo en una garantía de generalización.
Las evaluaciones usan sólo ocho imágenes de cada partición; el protocolo necesita
ampliación y múltiples semillas antes de usar sus diferencias como recomendación.

El optimizador restaura su RNG de stochastic rounding; la reanudación del trainer
requiere además RNG de ruido, posición de datos y estado de modelo. Rengu actualmente
vuelve a aplicar `train_seed` al entrar al loop: LR constante por sí solo no garantiza
la misma trayectoria después de una reanudación. No se modificó el trainer.
