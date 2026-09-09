# Evidencia para el siguiente diseño — 8 de septiembre de 2026

Objetivo: una alternativa recomendable para LoRA de pocos datos y fine-tuning más
amplio, evaluada por calidad por tiempo y memoria. Este documento actualiza la
premisa del catálogo histórico: ya existen resultados publicados específicos de
difusión para la familia Muon. No justifican extrapolar a todos los regímenes.

## Evidencia primaria consultada

| Trabajo | Evidencia que aporta | Límite para nuestro objetivo |
|---|---|---|
| [CMuon, agosto 2026](https://arxiv.org/abs/2608.02502) | Sus autores separan matrices fusionadas de AdaLN/QKV antes de ortogonalizar. Reportan FID 1.18 en ImageNet 256 con DiT de 675M y más de 2x aceleración frente a AdamW. | Es entrenamiento de DiT en ese protocolo; no demuestra ventaja en LoRA de pocos ejemplos, ni con momentum cuantizado. |
| [Scaling Muon for Diffusion Transformers, versión 3](https://arxiv.org/abs/2608.20818v3) | Alterna ortogonalización NS5 periódica y actualizaciones por filas. Evalúa modelos de 1.3B–15B; reporta reducciones del coste del optimizador y del tiempo activo hasta calidad. | Los resultados dependen de implementación distribuida y modelos grandes. No asumir que las mismas ganancias se trasladan a una GPU de 8 GB. |
| [Optimization Benchmark for Diffusion Models on Dynamical Systems](https://arxiv.org/abs/2510.19376) | Compara optimizadores entrenando difusión para trayectorias; sus autores encuentran Muon y SOAP competitivos frente a AdamW. | No es generación de imágenes ni una validación de personalización por LoRA. |
| [Schedule-Free, repositorio de los autores](https://github.com/facebookresearch/schedule_free) | Permite entrenar sin fijar horizonte de scheduler y distingue pesos para gradiente/evaluación. Los autores advierten que requiere ajustar LR y regularización. | El promedio de iterados consume estado; evitar presentar ausencia de scheduler como ausencia de ajuste o garantía de calidad. |
| [Prodigy, repositorio de los autores](https://github.com/konstmish/prodigy) | Tiene recomendaciones específicas de configuración para difusión y advierte sobre adaptación insuficiente de escala. | Es un control de facilidad de uso, no prueba de superioridad perceptual ni de memoria mínima. |

Los números de publicaciones son resultados reportados por sus autores, no
reproducciones nuestras. No usar loss de un artículo y FID de otro para construir
un ranking combinado.

Se verificó posteriormente [Riemannion en arXiv](https://arxiv.org/abs/2507.12142):
sus autores evalúan LoRA sobre una variedad de rango fijo, con resultados en LLM
y difusión. Su inicialización y tratamiento conjunto de factores requieren una
comparación de integración completa. [LoRA-Muon](https://arxiv.org/abs/2606.12921)
propone actualizaciones espectrales con geometría de los factores y sin segundos
momentos; sus experimentos publicados aquí son TinyShakespeare. Es una dirección
para estudiar LoRA, no evidencia de victoria en Anima. Ambos trabajos refuerzan
que no basta con tratar los dos factores LoRA como matrices independientes.

## Lo aprendido localmente

La variante Rakaon sin momentum prueba una intervención sencilla y barata en estado:
reducir anisotropía del precondicionador. La confirmación de 2000 pasos, C=128 y
semillas 43/44 da estos promedios (LR constante .0012):

| Optimizer | Train | Validation | Gap | Estado lógico B/parámetro |
|---|---:|---:|---:|---:|
| Adakaon sin momentum | .076136 | .084103 | .007966 | .032095 |
| Nekaon | .073594 | .080341 | .006747 | .563348 |
| Rakaon s=0 | .079913 | .089768 | .009855 | .032095 |
| Rakaon s=.1 | .079338 | .087528 | .008190 | .032095 |

La regularización ayuda frente a la ablación s=0, pero no supera a Nekaon. Dos
semillas no permiten una afirmación estadística fuerte. El modo isotrópico reduce
mucho el estado lógico; su valor depende de la curva de tiempo hasta calidad, no
de ese cociente aislado. Ver los JSON e informe de [Rakaon](rakaon.md).

## Decisiones para continuar

1. **No declarar ganador a Rakaon.** Mantenerlo como candidato refutable y reutilizar
   las mejoras del protocolo aunque la hipótesis del optimizador falle.
2. **Separar regímenes reales.** En LoRA medir calidad/adherencia/memorización con
   partición por sujeto o fuente que evite fuga. En fine-tuning medir además memoria
   máxima y tiempo a calidad bajo el mismo presupuesto de VRAM y datos.
3. **No descartar Muon por el U-Net sintético.** La nueva evidencia favorece estudiar
   geometría de matrices y grupos semánticos en DiT. No inferir QKV/AdaLN sólo por
   dimensiones: el modelo debe proporcionar metadatos explícitos de agrupación.
4. **Explorar coste amortizado.** Las operaciones caras pueden ejecutarse periódicamente
   si mantienen calidad; medir el ciclo completo y el tiempo hasta objetivo. Esta es
   una hipótesis motivada por Periodic Row-wise Muon, no una ganancia ya obtenida aquí.
5. **Exigir una comparación fuerte.** Ajustar referencias con igual presupuesto de
   búsqueda; congelar decisiones antes de datos/semillas de confirmación; reportar
   variación, fallos y objetivos no alcanzados. Incluir varios budgets de pasos/tiempo.

Un optimizador puede ser la recomendación práctica predominante sin ganar cada
microbenchmark, pero esa recomendación requiere evidencia en ambos usos reales.
Por ahora no hay tal demostración. El proxy actual entrena un U-Net desde cero;
ampliar C no lo convierte en fine-tuning y una bolsa de tensores no demuestra LoRA.
