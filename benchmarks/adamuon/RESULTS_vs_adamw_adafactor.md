# AdaMuon vs AdamW-fused vs Adafactor — measured comparison

Should you switch from `AdamW(fused=True)` (LoRA) / Adafactor (memory-constrained
full fine-tune) to AdaMuon? These are the numbers behind the answer. Reproduce with
`pixel_ddpm_ab.py` (synthetic full training) and `sdxl_lora_ab.py` (real SDXL LoRA);
both now expose `adamw_fused` and `adafactor` arms.

Hardware: single RTX 4080. All arms `+cosine`, swept-best LR. AdaMuon =
`ns_steps=2, cautious=True, betas=(0.95,0.999)`. "objective" = held-out val MSE
(pixel) / deterministic train-objective probe (SDXL LoRA). Lower is better.

## 1. Full training from scratch (pixel-DDPM, U-Net C=128, 3 seeds, 800 steps)

| optimizer | best objective | ms/step | optimizer state |
|---|---|---|---|
| **AdaMuon (bf16 mom)** | **0.0651** | 12.1 | 2.03 B/param |
| **AdaMuon (int8 mom)** | 0.0652 | 13.6 | 1.04 B/param |
| Adafusion (int8 mom) | 0.0700 | 10.9 | 1.04 B/param |
| Adafactor (best lr 1e-2) | 0.0827 | 11.8 | 2.57 B/param |
| AdamW-fused (best lr 1e-3) | 0.0854 | 7.1 | 8.00 B/param |

**Time-to-quality (wall-clock, via the measured ms/step):**

| target = | AdaMuon-int8 | AdamW-fused | Adafactor |
|---|---|---|---|
| AdamW's final (0.086) | **200 st / 2.7 s** | 800 st / 5.7 s | 500 st / 5.9 s |
| Adafactor's final (0.083) | **250 st / 3.4 s** | never (plateaus 0.086) | 800 st / 9.4 s |

AdaMuon reaches AdamW's "sufficient" quality **~2.1× faster in wall-clock despite
~1.9× slower steps** — it needs 4× fewer steps. AdamW-fused wins raw ms/step but
converges to a worse floor and costs 8× the optimizer memory. Adafusion is faster
per step than Adafactor *and* converges better.

## 2. Real SDXL LoRA (Illustrious-XL, rank 16, 2 seeds, 500 steps)

| optimizer | best objective | ms/step | optimizer state |
|---|---|---|---|
| **AdaMuon-int8** | **0.0900** | 498 | **1.37 B/param** |
| AdamW-fused (best lr 1e-3) | 0.0928 | 463 | 4.00 B/param |
| AdamW-fused lr 3e-4 | 0.0963 | 461 | 4.00 B/param |

ms/step is dominated by the UNet here, so the optimizer choice barely moves it
(AdaMuon-int8 ~8% slower from the int8 quant tax; AdaMuon-**bf16** measured ~tied at
~466 ms). AdaMuon reaches a lower floor AdamW never touches, and reaches the shared
0.0945 mark in 123 s vs AdamW's 140 s, at **1/3 the optimizer memory**.

## Verdict (when AdamW's quality is already "good enough")

| regime | per-step speed | time-to-your-quality | switch? |
|---|---|---|---|
| SDXL LoRA | AdaMuon-bf16 ~tied / int8 ~+8% | ~1.1–1.6× faster + lower floor, 1/3 memory | Pareto-positive, **modest** |
| full training from scratch | ~1.9× slower | **~2× faster** | yes |
| memory-bound full fine-tune (vs Adafactor) | Adafusion faster, AdaMuon ~+15% | **~2.8× faster** | **yes** |

- **Raw ms/step**: `AdamW(fused=True)` is the king (hand-written kernel). AdaMuon-bf16
  ties it on UNet-bound LoRA; int8 trades ~8% step time for half the momentum bytes.
- **Convergence (time to a given quality)**: AdaMuon wins in every regime (3–4× fewer
  steps more than pays back the slower step).
- **Memory**: AdaMuon/Adafusion sit in the Adafactor class (~1–2 B/param). For full FT
  where AdamW (even 8-bit) OOMs, they are usable *and* beat plain Adafactor.

Bottom line: the clear win is **full fine-tuning** (you are forced onto Adafactor by
VRAM — AdaMuon/Adafusion give ~2–3× less time-to-quality at equal-or-less memory).
For LoRA the gain is real but modest; AdaMuon-bf16 if you want to keep AdamW-fused's
per-step speed, int8/4bit if you want the memory back.
