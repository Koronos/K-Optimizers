# retime-016 fix: foreach/fused compact-Kahan regression of 026621d

Metric: self-CUDA per optimizer step (profiler, sum of kernel self time), **min over 15 profiled
steps** per process (median was dominated by the laptop GPU's per-process clock modes: the same
SR code measured 4.3 or 5.5 ms depending on the process), then the **median over 6 processes per
side**, ABBA-interleaved (3 rounds x A B B A). A = 0.7.15 export (6201c1f), B = the fix on top of
32ae12b. AC power. Scripts: `selfcuda_fix.py`, `abba_fix.sh`, per-kernel `kern_fix.py`; raw
`fix_abba_raw.txt`. Shapes: `unet` = 8x(1024,1024) + 16x(4096,); `lora` = 400x(64,16)/(16,64) +
28x(64,). Adakaon: wd 0.1, cautious; `nomom` = betas (0, 0.999). Nekaon: defaults.

Before the fix (026621d, same metric, one ABBA-free pass): foreach UNet kahan8/16 1.30-1.82x.

| case | 0.7.15 (6201c1f) ms | fix ms | fix/0.7.15 |
|---|---|---|---|
| unet/ada-nomom/foreach/stochastic_rounding | 4.970 | 4.277 | 0.861 |
| unet/ada-4bit/foreach/stochastic_rounding | 6.745 | 6.429 | 0.953 |
| unet/nekaon/foreach/stochastic_rounding | 6.545 | 6.615 | 1.011 |
| unet/ada-nomom/foreach/kahan8 | 4.188 | 4.093 | 0.977 |
| unet/ada-4bit/foreach/kahan8 | 6.540 | 6.468 | 0.989 |
| unet/nekaon/foreach/kahan8 | 6.949 | 6.857 | 0.987 |
| unet/ada-nomom/foreach/kahan16 | 4.444 | 4.351 | 0.979 |
| unet/ada-4bit/foreach/kahan16 | 6.819 | 6.729 | 0.987 |
| unet/nekaon/foreach/kahan16 | 7.442 | 7.377 | 0.991 |
| unet/ada-nomom/fused/stochastic_rounding | 0.349 | 0.345 | 0.989 |
| unet/ada-4bit/fused/stochastic_rounding | 0.547 | 0.497 | 0.909 |
| unet/nekaon/fused/stochastic_rounding | 0.723 | 0.649 | 0.898 |
| unet/ada-nomom/fused/kahan8 | 0.532 | 0.493 | 0.927 |
| unet/ada-4bit/fused/kahan8 | 0.645 | 0.627 | 0.973 |
| unet/nekaon/fused/kahan8 | 1.111 | 1.062 | 0.957 |
| unet/ada-nomom/fused/kahan16 | 0.617 | 0.611 | 0.991 |
| unet/ada-4bit/fused/kahan16 | 0.679 | 0.675 | 0.994 |
| unet/nekaon/fused/kahan16 | 1.297 | 1.324 | 1.020 |
| lora/ada-nomom/foreach/stochastic_rounding | 0.372 | 0.372 | 0.999 |
| lora/ada-4bit/foreach/stochastic_rounding | 0.752 | 0.591 | 0.785 |
| lora/nekaon/foreach/stochastic_rounding | 1.377 | 1.486 | 1.079 |
| lora/ada-nomom/foreach/kahan8 | 1.209 | 1.321 | 1.092 |
| lora/ada-4bit/foreach/kahan8 | 2.530 | 2.732 | 1.080 |
| lora/nekaon/foreach/kahan8 | 3.678 | 3.625 | 0.986 |
| lora/ada-nomom/foreach/kahan16 | 2.258 | 2.081 | 0.922 |
| lora/ada-4bit/foreach/kahan16 | 3.529 | 3.352 | 0.950 |
| lora/nekaon/foreach/kahan16 | 3.670 | 3.495 | 0.952 |
| lora/ada-nomom/fused/stochastic_rounding | 0.163 | 0.162 | 0.997 |
| lora/ada-4bit/fused/stochastic_rounding | 0.221 | 0.222 | 1.007 |
| lora/nekaon/fused/stochastic_rounding | 0.332 | 0.332 | 1.000 |
| lora/ada-nomom/fused/kahan8 | 0.204 | 0.223 | 1.091 |
| lora/ada-4bit/fused/kahan8 | 0.309 | 0.304 | 0.984 |
| lora/nekaon/fused/kahan8 | 0.567 | 0.563 | 0.993 |
| lora/ada-nomom/fused/kahan16 | 0.131 | 0.133 | 1.019 |
| lora/ada-4bit/fused/kahan16 | 0.190 | 0.204 | 1.074 |
| lora/nekaon/fused/kahan16 | 0.342 | 0.335 | 0.980 |

SR rows run IDENTICAL kernels in A and B (checked kernel by kernel); their spread (0.79-1.08)
is the noise floor of this metric on this machine, LoRA foreach worst.

Kernel level (min over 15 steps, `kern_fix.py`):

| kernel (case) | 0.7.15 | 026621d | fix |
|---|---|---|---|
| `_adakaon_tile_kernel` (lora, 4bit, kahan8) | 45.9 us | 57.4 us | 41.7-42.8 us |
| `_adakaon_tile_kernel` (lora, nomom, kahan8) | 30.5 us | 41.5 us* | 32.4 us |
| `_adakaon_tile_kernel` (lora, nomom, kahan16) | 19.6 us | 23.3 us* | 19.8 us |
| `_adakaon_tile_kernel` (lora, 4bit, kahan16) | 27.9 us | 34.2 us* | 30.1 us |
| `_chunked_nomom_keep_batched_g` (unet, kahan8) | 80.4 us | 105 us | 92.1 us |

(* intermediate variants of this fix, not 026621d itself.) What changed:
1. foreach: the decay's decoded value is ONE Triton launch (`_ck_decode_kernel`) instead of
   ~10 ATen integer kernels, and the (weights, residuals) stack it decodes from is handed to the
   weight write instead of being stacked a second time.
2. fused keep passes: decode only the lanes whose cautious sign can depend on the residual
   (`wd_keep_flat`), with a block-uniform skip — bit-identical mask. Residual cost: +15% on
   that kernel (the extra compare/reduction), about +2% of the fused step.
3. fused one-block tile: kahan8 decodes up front (loads overlap the momentum math); kahan16
   decodes late (up front measured +19-23% for it). Residual cost: LoRA tile +6% (kahan8
   no-momentum) / +8% (kahan16 4-bit) on that kernel — the price of reading the full value.
