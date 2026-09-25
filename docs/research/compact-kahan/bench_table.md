| what | median ms | q1 | q3 |
|---|---|---|---|
| writer none | 0.266 | 0.260 | 0.270 |
| writer stochastic_rounding (Triton) | 0.070 | 0.070 | 0.070 |
| writer stochastic_rounding (torch) | 1.170 | 1.151 | 1.187 |
| writer kahan (bf16 shift, torch) | 0.386 | 0.385 | 0.388 |
| writer kahan8 (Triton axpy) | 0.074 | 0.074 | 0.074 |
| writer kahan8 (torch reference) | 5.803 | 5.689 | 6.040 |
| step LoRA bag 200x(256,256)+100x(512,) stochastic_rounding fused=True | 2.895 | 2.773 | 3.539 |
| step LoRA bag 200x(256,256)+100x(512,) stochastic_rounding fused=False | 16.748 | 16.148 | 21.238 |
| step LoRA bag 200x(256,256)+100x(512,) kahan8 fused=True | 3.625 | 2.979 | 4.464 |
| step LoRA bag 200x(256,256)+100x(512,) kahan8 fused=False | 17.665 | 16.839 | 20.973 |
| step big 2x(1024,1200) stochastic_rounding fused=True | 0.295 | 0.294 | 0.297 |
| step big 2x(1024,1200) stochastic_rounding fused=False | 2.891 | 2.729 | 3.840 |
| step big 2x(1024,1200) kahan8 fused=True | 0.318 | 0.313 | 0.321 |
| step big 2x(1024,1200) kahan8 fused=False | 4.094 | 3.944 | 4.973 |
| step UNet-ish 8x(1024,1024)+16x(4096,) stochastic_rounding fused=True | 2.611 | 2.200 | 2.700 |
| step UNet-ish 8x(1024,1024)+16x(4096,) stochastic_rounding fused=False | 13.407 | 12.747 | 14.982 |
| step UNet-ish 8x(1024,1024)+16x(4096,) kahan8 fused=True | 2.031 | 1.968 | 2.852 |
| step UNet-ish 8x(1024,1024)+16x(4096,) kahan8 fused=False | 14.321 | 13.609 | 18.210 |
