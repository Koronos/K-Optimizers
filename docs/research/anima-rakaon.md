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

## Confirmación en curso

Se están ejecutando seeds 43 y 44, con 32 imágenes de evaluación por split,
200 pasos, sin previews y orden alternado: seed 43 (`Nekaon → isotropic`) y
seed 44 (`isotropic → Nekaon`). Todavía no hay resultados; por tanto, no se
deben mezclar con el piloto de seed 42 ni extraer una conclusión de ranking.
