# Medición definitiva — bf16_method="kahan8" vs "stochastic_rounding" vs "kahan" (legacy)

Worktree: `.claude/worktrees/compact-kahan`, rama `feature/compact-kahan`, base commit `b427e0c`.
Python: `/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe`, `PYTHONPATH=src` relativo
al worktree. `kaon.__file__` verificado apuntando a
`.../worktrees/compact-kahan/src/kaon/__init__.py` (kaon 0.7.14) en cada corrida.

No se modificó nada bajo `src/`.

## Hardware / energía

- `(Get-CimInstance Win32_Battery).BatteryStatus` = **2 (AC)** en todas las verificaciones
  (18:36, 18:52, 19:00, 19:05 — antes de cada corrida real).
- GPU: NVIDIA RTX 3000 Ada Generation Laptop GPU (laptop, WDDM).
- `nvidia-smi --query-gpu=power.limit,power.max_limit,power.default_limit,clocks.max.sm,temperature.gpu,utilization.gpu`
  al inicio de la sesión (18:36:55):
  ```
  power.limit [W], power.max_limit [W], power.default_limit [W], clocks.max.sm [MHz], temperature.gpu, utilization.gpu [%]
  [N/A], 60.00 W, 35.00 W, 3105 MHz, 56, 0 %
  ```
  (`power.limit` no lo reporta este driver/GPU láptop en esta consulta; `power.draw`/`power.max_limit`
  sí. Cap efectivo observado en `nvidia-smi` durante carga: ~43 W.)
- Antes de CADA corrida real se confirmó GPU ociosa: `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw`
  en 0-3 % de utilización, sin procesos de cómputo activos salvo Postman (memoria ~100-700 MiB, `used_gpu_memory=N/A`,
  irrelevante — no es un proceso CUDA de benchmark).

## ANOMALÍA: contaminación por uso concurrente de la GPU (descartada)

El orquestador avisó que el revisor usó la GPU por error entre **~18:40 y 18:59** (en Windows,
`CUDA_VISIBLE_DEVICES=""` no oculta la GPU; hace falta `-1`). La primera corrida completa
(`writer` + shape `big` + `LoRA bag/no_momentum` parcial) arrancó a las **18:46:09** — dentro de
esa ventana. **Se descartó por completo** (nunca se persistió a JSON: el proceso fue matado con
`Stop-Process` antes de que `main()` escribiera resultados) y se repitió desde cero.

Antes de repetir se esperó hasta las **19:00:10**, más de un minuto después del cierre reportado
de la ventana (18:59), y se reconfirmó GPU ociosa (0 % util, 0 MiB, sin procesos de benchmark) y
AC. Los kernels Triton ya compilados en la corrida descartada quedaron cacheados en disco, así que
la repetición no tuvo que recompilar nada (por eso el shape "big" tardó ~2 s en vez de decenas de
segundos) — la ventana de la medición REAL (por debajo) no se solapa con la de compilación de la
corrida descartada.

## Ventanas de tiempo de cada medición (todas las que se reportan)

Todas dentro de **19:00:21 – 19:06:24**, es decir completamente FUERA de la ventana de
interferencia (18:40–18:59):

| medición | inicio | fin |
|---|---|---|
| writer isolated (2^22 elems) | 19:00:21 | 19:00:22 |
| step: big / no_momentum | 19:00:22 | 19:00:24 |
| step: big / momentum_4bit | 19:00:24 | 19:00:25 |
| step: LoRA bag / no_momentum | 19:00:25 | 19:02:10 |
| step: LoRA bag / momentum_4bit | 19:02:10 | 19:05:24 |
| step: UNet-ish / no_momentum | 19:05:24 | 19:05:29 |
| step: UNet-ish / momentum_4bit | 19:05:29 | 19:05:36 |
| memory (B/param, first-step peak) | 19:06:23 | 19:06:24 |

Wall time total del script de step+writer: 315.3 s (impreso por el propio script).

## Protocolo

- Brazos intercalados ABC ABC... dentro de cada grupo comparable (ver `bench_definitive.py:interleaved_bench`):
  para cada rep se recorren todos los métodos del grupo antes de pasar a la siguiente rep, con
  `torch.cuda.synchronize()` después de cada llamada individual y medición por eventos CUDA
  (`torch.cuda.Event(enable_timing=True)`).
- Warmup: 8 reps (writer) / 6 reps (step) por brazo antes de empezar a medir.
- Reps medidas: writer 40; step 40 (shape "big"), 32 (LoRA bag, UNet-ish) — todas ≥30.
- Mediana + IQR (Q1/Q3) reportados por brazo.
- "kahan" (legado) nunca llega a foreach/fused (`kaon._backend.per_param_only_bf16_method`
  fuerza `_group_foreach_eligible=False` para ese método) → sólo se midió en su camino real
  (per-param / "native", con `foreach=False, fused=False` explícito). SR y kahan8 se midieron
  TAMBIÉN en ese mismo camino native (para tener un 3-way ABC real apples-to-apples), además de
  sus propios grupos foreach-vs-foreach y fused-vs-fused (SR vs kahan8 solamente, 2 brazos AB).
- Momentum: "no_momentum" = `betas=(0.0, 0.999)`, `momentum_dtype="bfloat16"`; "momentum_4bit" =
  `betas=(0.9, 0.999)` (default), `momentum_dtype="4bit"`.
- Formas:
  - big: `[(1024,1200)]*2`.
  - LoRA bag: 512 tensores = 256 pares (A: `(16, dim)`, B: `(dim, 16)`), `dim` de 320 a 1280
    linealmente espaciado (256 valores, en su mayoría distintos) — ver `lora_bag()` en el script.
  - UNet-ish: `[(1024,1024)]*8 + [(4096,)]*16` (mismo bag orientativo del script del autor).

## Batería de control (`benchmarks/control/profiler.py`)

**Saltado.** `profiler.py --opt <name>` selecciona una entrada fija de `benchmarks/control/registry.py`
(lambda con parámetros hardcodeados); ni `registry.py` ni `profiler.py` exponen un flag
`bf16_method` — las tres entradas Adakaon del registry (`Adakaon-nomom`, `Adakaon-bf16`,
`Adakaon-bf16-fused`) no lo parametrizan. No se puede correr `profiler.py --opt Adakaon` con los
tres métodos sin tocar `src/` o el registry, que está fuera de alcance de esta tarea. No hay
números "ms/step full-FT/LoRA style" comparables con RANKINGS.md en este entregable.

## Archivos

- `bench_definitive.py` — writer + step benchmark (script propio, adaptado del orientativo
  `bench_write.py` del autor).
- `writer_results.json`, `writer_table.md` — writer aislado, 2^22 elementos.
- `step_results.json`, `step_table.md` — paso completo, 3 formas x 2 momentum x {native, foreach, fused}.
- `measure_memory.py` — copia del script orientativo del autor (sin cambios de lógica, sólo 2
  prints de timestamp añadidos) — reusado tal cual.
- `memory_table.md` — B/param de estado + compensación + pico de memoria del primer paso.

## Anomalías / notas

- El camino "foreach" en el bag LoRA (512 tensores, dims mayormente distintos) resultó MÁS LENTO
  que el camino "native" (per-param) en varios casos (p. ej. no_momentum: native SR 396 ms vs
  foreach SR 553 ms). Hipótesis: con 256 dims casi todos distintos, el foreach no logra apilar
  eficientemente en buckets grandes (a diferencia del bag orientativo del autor, que sólo usaba 2
  formas distintas) — la sobrecarga de construir/mantener muchos buckets pequeños supera el
  ahorro de lanzamientos. Esto es una característica real medida, no un artefacto del harness
  (confirmado reproducible entre la corrida descartada y la limpia, mismo orden de magnitud una
  vez fuera la compilación). No se investigó más a fondo por estar fuera de alcance (no se toca
  `src/`).
- `power.limit` no es reportado por `nvidia-smi` para esta GPU/driver en la consulta pedida;
  se registran `power.max_limit`/`power.default_limit`/`power.draw` como sustituto verificable.
