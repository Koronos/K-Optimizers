# Siguiente enfoque: dirección temporal y geometría LoRA

## Decisión

Priorizar un prototipo LoRA que reconozca el par A/B y precondicione en el espacio
del rango. Mantener Rakaon como control barato y hacer una ablación de momentum
antes de combinar mecanismos. Para full fine-tuning, estudiar por separado
ortogonalización matricial periódica con grupos semánticos explícitos.
Una interfaz común puede tener rutas diferentes; no hay evidencia para forzar
la misma actualización sobre factores LoRA y matrices completas.

## Qué sabemos y qué no

Los ensayos de Anima comparan recetas completas. Nekaon cambia momentum,
precondicionamiento, lookahead, weight decay, centralización y cautious masking.
Su ventaja no demuestra que el momentum sea la causa. AdamW usa estados BF16
en este protocolo: dos buffers cuestan aproximadamente 4 B/parámetro, no los
8 B/parámetro de estados FP32. Rakaon ahorró apenas 0.028 GiB de pico frente a
Nekaon en la LoRA real; merece aceptar algo de estado si mejora tiempo a calidad.

En el código isotrópico, sin decay ni redondeo, la dirección es exactamente:

`u = g / max(sqrt(max(v_hat, eps)), RMS(g)/clip_threshold)`.

El estado escalar cambia magnitud, nunca orientación. Cuando el clipping domina,
la magnitud histórica desaparece de esa actualización. Esto se deduce del código;
todavía no hemos medido con qué frecuencia ocurre en Anima. Instrumentar fracción
de clipping por factor y capa, RMS de gradientes y norma del cambio efectivo BA
en ejecuciones diagnósticas separadas de las mediciones de velocidad.

## Evidencia externa relevante

| Dirección | Evidencia primaria | Aplicabilidad y coste |
|---|---|---|
| Gram LoRA | [Riemannian Preconditioned LoRA, código de los autores](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA) evalúa precondicionadores r×r también en difusión texto-imagen. | Trabaja con factores emparejados; estudiar inicialización singular y regularización. No requiere materializar una matriz completa de actualización. |
| Geometría LoRA completa | [Riemannion](https://arxiv.org/html/2507.12142v2) incluye Stable Diffusion 2, DreamBooth y ranks 4/8/16, con CLIP/DINO. | Evidencia más próxima a personalización que un benchmark de lenguaje. Su inicialización y actualización geométrica forman un método conjunto; no atribuir todo a un reemplazo del optimizer. |
| Control barato A/B | [LoRA+](https://proceedings.mlr.press/v235/hayou24a.html) usa una razón fija entre los LR de los factores. | Control de escalas sin estado adicional. Sus resultados no prueban ventaja en Anima; mantiene LR constante por grupo. |
| Matrices completas | [CMuon](https://arxiv.org/abs/2608.02502) estudia separar matrices fusionadas antes de ortogonalizar en DiT. | Usar metadatos AdaLN/QKV, no adivinar particiones por dimensiones. No aplicar automáticamente por factor LoRA. |
| Coste amortizado | [Periodic Row-wise Muon](https://arxiv.org/abs/2608.20818v3) alterna pasos espectrales y por filas en DiT grandes. | Medir ciclos completos y transferencia bajo offload; sus mejoras distribuidas no equivalen a ganancias en nuestra GPU. |

[CAME](https://arxiv.org/abs/2307.02047) es otra referencia para estudiar confianza
y estados factorizados, pero la evidencia de su artículo es NLP. Lo dejo después
de separar momentum, clipping y geometría; añadir todos a la vez impediría saber
qué funcionó. [El estudio de escalado de LR LoRA](https://arxiv.org/abs/2602.06204)
refuerza que rank e inicialización deben registrarse al transferir una receta.

## Prueba local del mecanismo

Ejecuté [un probe CPU](../../benchmarks/lora_parameterization_probe.py) sobre una
matriz 64×48, rank 4, con factores de rango completo. Reemplazar A por cA y B por
B/c conserva BA, con error inicial menor que 2e-16. Sin embargo, para c=0.1/10,
el cambio efectivo de Rakaon isotropic difiere del caso c=1 en aproximadamente
6.19/6.11 veces la norma de referencia. El SGD precondicionado por Gram sin
regularización conserva ese cambio hasta aproximadamente 2e-11 de error relativo.
[Resultados reproducibles](../../benchmarks/lora_parameterization_probe.json).

Esto verifica sensibilidad a la representación en un paso; no demuestra mejor
loss, convergencia ni imágenes. Tampoco compara tamaños de paso entre métodos:
sus escalas son distintas. El control Gram falla con la inicialización estándar
B=0, pues BᵀB es singular. No se debe convertir este probe en código de producción.

## Prototipo y pruebas que deciden su continuidad

1. **Diagnóstico de momentum:** añadir únicamente EMA del update normalizado a
   Rakaon isotropic; comparar beta1=0/.5/.9 con idéntico clipping y decay=0.
   Primero FP32 para separar el mecanismo de la cuantización; solo si mejora,
   comparar almacenamiento de 4 bits. El coste lógico extra pasa de 4N bytes a
   aproximadamente 0.5N más escalas, sin prometer mismo pico CUDA ni calidad.
2. **Prototipo Gram:** para BA, calcular ambos updates con los factores anteriores:
   `dA=solve(B.T@B + lambda_B*I, gA)` y
   `dB=solve(A@A.T + lambda_A*I, gB.T).T`. Usar solves, no inversas explícitas.
   Elegir y probar damping en FP32, incluyendo B=0, rank degenerado y escalas
   extremas. El damping puede romper la invariancia del control ideal. Nunca
   inicializar ambos factores no nulos sin compensar el cambio del modelo base.
3. **Controles:** mismo presupuesto de búsqueda para AdamW/Nekaon, Rakaon,
   Rakaon con momentum y Gram. Incluir LoRA+ para saber si basta corregir escalas.
   No combinar Gram y momentum cuantizado hasta medirlos por separado.
4. **Anima:** curvas cada 50 pasos, primero 200 pasos de exploración en seed 43;
   fijar receta y objetivos antes de nuevas seeds 45/46. Mantener constante el LR
   por grupo y guardar optimizer, RNG y posición de datos. Evaluación determinista
   por nivel de ruido; registrar tiempo activo y total, picos y fallos.
5. **Selección:** menor tiempo observado hasta validation común, con gap reportado
   aparte y sin interpolar objetivos no alcanzados. No premiar gap pequeño por
   subentrenamiento. Confirmar adherencia, fidelidad y diversidad visual; Pets por
   raza no sustituye personalización por sujeto. Mantener test reservado.

El siguiente prototipo se justifica por mecanismos concretos y evidencia cercana,
no porque alguno de esos artículos garantice superar a Nekaon en nuestro caso.
