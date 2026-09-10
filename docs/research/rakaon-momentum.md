# Ablación de momentum — 10 septiembre 2026

Se añadió `beta1`, desactivado por defecto. Solo se admite con `shrinkage=1`
y sin bloques. Promedia el update ya normalizado y recortado, corrige el sesgo
con el número de updates del parámetro y aplica decay después. Cada tensor
agrega un buffer FP32. No es aún una implementación de momentum comprimido.
Los parámetros sin gradiente no avanzan ni sus momentos ni su contador.
No se permite cambiar beta1 una vez inicializado el estado; checkpoints
anteriores sin beta1 se interpretan como beta1=0.

## Validación

Pasaron 67 pruebas de Rakaon, bloques, reanudación entre procesos y momentum;
además, 9 pruebas de configuración Anima. Se verificaron ecuaciones contra una
referencia independiente, preservación del gradiente, BF16 en CPU/CUDA,
restauración del estado FP32, gradientes ausentes y checkpoints anteriores.
Ruff pasó en los archivos modificados.

## Cribado emparejado

Proxy U-Net C64, seed42, 400 pasos, batch8, LR constante y evaluación cada100.
Tres LR por variante: 0.0006/0.0012/0.0024. Selección descriptiva por menor
validation final dentro de ese mismo presupuesto; no confirmación independiente.
[Datos completos](../../benchmarks/rakaon_momentum_screen.json).

| Beta1 | LR elegido | Validation | Gap | Tiempo activo s | Estado lógico B/parámetro |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0024 | 0.101945 | 0.003435 | 6.723 | 0.000255 |
| 0.5 | 0.0012 | 0.101558 | 0.004596 | 7.925 | 4.000255 |
| 0.9 | 0.0012 | 0.106660 | 0.008386 | 6.982 | 4.000255 |

Momentum moderado mejora ligeramente la pérdida elegida pero usa más tiempo y
tiene mayor gap. Momentum alto no ayuda en esta rejilla. Ninguna variante cruza
validation<0.1 al final, ni alcanza el objetivo original 0.07. No hay una
aceleración hasta calidad equivalente demostrada por estos endpoints.
Los tiempos de la tabla son acumulaciones medidas, no estimaciones de throughput.
Estado lógico no es pico de VRAM; el buffer FP32 agrega aproximadamente 4 bytes
por parámetro.

## Ensayo real iniciado

Anima/Pets seed43, LoRA rank16, 256px, 200 pasos, LR constante 0.0001,
beta1=0.5, 32 imágenes por split de evaluación y previews desactivados.
La receta cambia solo beta1 respecto a Rakaon isotropic. Se evalúa antes de
entrenar y cada50 pasos; la frecuencia cambia frente a la referencia previa
que evaluó al principio y al final. La evaluación aísla/restaura RNG.
Se verificará la huella inicial y las pérdidas iniciales antes de comparar.
El mayor coste de evaluación debe aparecer en el tiempo total, separado del
tiempo activo. La búsqueda de LR del proxy no se transfiere como receta Anima.

La implementación Gram sigue pendiente: esta ablación permite evaluar el primer
mecanismo antes de combinarlo con cambios de parametrización.
