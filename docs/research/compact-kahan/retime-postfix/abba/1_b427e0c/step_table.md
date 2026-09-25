
## big 2x(1024,1200)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 1.3235 | 1.2524 | 1.7121 |
| native | kahan | 1.4602 | 1.3527 | 1.8330 |
| native | kahan8 | 1.4244 | 1.2411 | 1.8790 |
| foreach | stochastic_rounding | 1.0962 | 1.0383 | 1.2872 |
| foreach | kahan8 | 1.1284 | 1.0916 | 1.3527 |
| fused | stochastic_rounding | 0.3891 | 0.3031 | 0.4608 |
| fused | kahan8 | 0.3881 | 0.3062 | 0.4956 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 2.1734 | 2.0316 | 2.5426 |
| native | kahan | 2.2344 | 2.0869 | 2.6911 |
| native | kahan8 | 2.1028 | 2.0531 | 2.9123 |
| foreach | stochastic_rounding | 1.5084 | 1.4766 | 1.5708 |
| foreach | kahan8 | 1.5452 | 1.5063 | 1.6056 |
| fused | stochastic_rounding | 0.3062 | 0.2970 | 0.5304 |
| fused | kahan8 | 0.3164 | 0.3011 | 0.5612 |

## LoRA bag 512x r=16 dims 320-1280


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 421.8153 | 354.2477 | 484.8108 |
| native | kahan | 436.8671 | 365.5322 | 493.6428 |
| native | kahan8 | 420.3013 | 346.5226 | 507.3347 |
| foreach | stochastic_rounding | 604.2163 | 553.9778 | 673.9866 |
| foreach | kahan8 | 645.6310 | 564.7268 | 713.4874 |
| fused | stochastic_rounding | 124.3315 | 114.0460 | 142.2623 |
| fused | kahan8 | 126.7405 | 117.4569 | 140.6300 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 943.4716 | 804.7575 | 1003.9900 |
| native | kahan | 899.2681 | 788.8169 | 1044.0796 |
| native | kahan8 | 910.5987 | 756.7012 | 1017.6962 |
| foreach | stochastic_rounding | 960.7808 | 836.9644 | 1200.7045 |
| foreach | kahan8 | 987.6188 | 925.1420 | 1195.7177 |
| fused | stochastic_rounding | 158.6038 | 131.7222 | 186.0936 |
| fused | kahan8 | 156.1339 | 130.5477 | 185.7341 |

## UNet-ish 8x(1024,1024)+16x(4096,)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 13.2634 | 11.4319 | 17.0639 |
| native | kahan | 15.6882 | 13.1758 | 19.4970 |
| native | kahan8 | 15.8152 | 12.9935 | 20.1902 |
| foreach | stochastic_rounding | 4.7800 | 4.3950 | 5.9720 |
| foreach | kahan8 | 5.6714 | 4.7104 | 6.8209 |
| fused | stochastic_rounding | 0.5402 | 0.5171 | 0.6134 |
| fused | kahan8 | 0.6205 | 0.6021 | 0.6953 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 31.6052 | 27.0909 | 40.8074 |
| native | kahan | 34.9348 | 28.5338 | 46.1394 |
| native | kahan8 | 31.9657 | 28.0924 | 36.6295 |
| foreach | stochastic_rounding | 7.5607 | 6.9908 | 9.1320 |
| foreach | kahan8 | 7.5223 | 7.2028 | 9.4710 |
| fused | stochastic_rounding | 0.6902 | 0.5980 | 0.8233 |
| fused | kahan8 | 0.7434 | 0.6554 | 0.8776 |
