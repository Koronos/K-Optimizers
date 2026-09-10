# Anima y Rakaon: resultados acotados

Este documento separa tres protocolos que no deben combinarse.

## Piloto LoRA (seed 42)

El piloto usa semilla 42 y 16 imágenes de evaluación (8 train y 8 validation), cuatro
optimizadores, LoRA rank 16, BF16, 256 px y 200 pasos. Los resultados completos
están en [`comparison_results.md`](../../benchmarks/anima/comparison_results.md)
y [`comparison_results.json`](../../benchmarks/anima/comparison_results.json).

Rakaon isotropic quedó cercano a Nekaon en la evaluación: su pérdida validation
fue 0.129149 frente a 0.129074, con 255.0 s frente a 272.1 s de tiempo activo.
La diferencia es descriptiva y no establece significancia estadística ni una
ventaja general. El pico CUDA fue 5.767 GiB para Rakaon isotropic y 5.794 GiB
para Nekaon; esos picos incluyen asignaciones de previews.

## Smoke de full fine-tuning

El smoke de tres pasos se ejecutó sin adapter, con 1,956,405,248 parámetros
actualizados, BF16, 256 px y 24 bloques intercambiados. El mecanismo observado
incluye liberación de gradientes (`gradient_release`); el pico medido fue
1.56 GiB. Ocho matrices inspeccionadas cambiaron entre el checkpoint inicial y
el checkpoint del paso 3. Los datos están en
[`full_smoke_results.json`](../../benchmarks/anima/full_smoke_results.json) y
[`full_smoke_weight_changes.json`](../../benchmarks/anima/full_smoke_weight_changes.json).

Este smoke confirma una ruta de actualización de pesos completos bajo ese
ajuste de offload, pero tres pasos no permiten afirmar convergencia de full
fine-tuning ni calidad de generación.

## Confirmación y control AdamW

Se completaron seeds 43 y 44, con 32 imágenes de evaluación por split,
200 pasos, sin previews y orden alternado: seed 43 (`Nekaon → isotropic`) y
seed 44 (`isotropic → Nekaon`). AdamW fused se ejecutó después con las mismas
semillas y LR. El [reporte combinado](../../benchmarks/anima/paired_confirmation.md)
verifica modelo, dataset, protocolo, huella inicial de los adapters y pérdidas
iniciales dentro de cada semilla. No se deben mezclar estas pérdidas con el piloto
de seed 42.

| Seed | Optimizador | Validation | Gap absoluto | Tiempo activo (s) | Pico CUDA (GiB) |
|---|---|---:|---:|---:|---:|
| 43 | Nekaon | 0.132597 | 0.009018 | 196.0 | 4.652 |
| 43 | Rakaon isotropic | 0.133067 | 0.009335 | 181.4 | 4.624 |
| 43 | AdamW fused | 0.133572 | 0.009295 | 160.5 | 4.763 |
| 44 | Nekaon | 0.132850 | 0.009482 | 211.3 | 4.652 |
| 44 | Rakaon isotropic | 0.133323 | 0.009603 | 175.9 | 4.624 |
| 44 | AdamW fused | 0.134044 | 0.009608 | 153.1 | 4.763 |

Rakaon usa menos tiempo para completar 200 pasos que Nekaon (7.4% y 16.8%),
pero su pérdida validation es mayor en ambas semillas. AdamW completa los pasos
más rápido y tiene mayor pérdida. Esto describe un compromiso; no demuestra
menor tiempo hasta una calidad equivalente. Solo se evaluaron los pasos 0 y 200,
por lo que no se pueden interpolar curvas de convergencia ni tiempos a umbrales.
La diferencia de pico entre Rakaon y Nekaon es apenas 0.028 GiB en esta LoRA.

AdamW usa estados estándar BF16, betas 0.9/0.999 y weight decay 0.01; Nekaon
usa su receta cuantizada y Rakaon no usa momentum. No es un estudio de ajuste
exhaustivo ni una comparación que iguale todas las precisiones de estado.
Ningún brazo alcanza loss < 0.07 y gap absoluto < 0.007. El gap ya era negativo
antes de entrenar; restarle el valor inicial no equivale a cumplir ese objetivo.

## Inspección visual

Se cargaron los adapters del piloto seed 42, paso 200, y se generaron cuatro
prompts de razas a 512 px con 20 pasos Euler, guidance 4 y semilla 20260909.
Se inspeccionaron exclusivamente los previews del paso 0, anteriores a cualquier
actualización del trabajo de revisión. Ambos producen gatos, beagle y shiba inu
reconocibles; cambian poses, proporciones y detalles. Ambos beagles tienen estilo
de ilustración. No hay una ventaja visual consistente demostrada por esos ocho
ejemplos, ni métricas perceptuales o evaluación humana ciega. Los 32 ejemplos
de test reservados siguen sin utilizarse.

## Exploración de la mitad de pasos

Tras la confirmación se probó Rakaon isotropic con LR constante 0.0002 y 100
pasos, seed 43, mismos pesos iniciales y splits de 32 imágenes. Antes de obtener
el resultado se registró el [criterio](../../benchmarks/anima/fast43_protocol.json):
igualar o mejorar tanto validation como gap absoluto de Nekaon a 200 pasos,
con menor tiempo activo. Es un ensayo exploratorio elegido usando seed 43,
no una confirmación independiente.

El [resultado](../../benchmarks/anima/fast43_results.md) fue validation 0.135747,
gap absoluto 0.009612 y 85.4 s activos, frente a 0.132597, 0.009018 y 196.0 s
de Nekaon. Falla ambas condiciones de calidad; el menor tiempo no constituye
una aceleración hasta un resultado equivalente. Duplicar LR y reducir a la
mitad los pasos queda descartado como mejora demostrada en este ensayo.
