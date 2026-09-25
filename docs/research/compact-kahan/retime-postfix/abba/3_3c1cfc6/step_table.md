
## big 2x(1024,1200)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 1.4259 | 1.2605 | 1.7541 |
| native | kahan | 1.4909 | 1.3537 | 1.8596 |
| native | kahan8 | 1.4157 | 1.2790 | 1.8248 |
| foreach | stochastic_rounding | 1.1510 | 1.0537 | 1.2360 |
| foreach | kahan8 | 1.1960 | 1.0670 | 1.2544 |
| fused | stochastic_rounding | 0.2954 | 0.2621 | 0.3779 |
| fused | kahan8 | 0.2903 | 0.2703 | 0.3246 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 5.1640 | 4.4165 | 7.0943 |
| native | kahan | 5.2065 | 4.7831 | 7.4107 |
| native | kahan8 | 4.8553 | 4.6490 | 6.1686 |
| foreach | stochastic_rounding | 2.8160 | 2.6092 | 3.4662 |
| foreach | kahan8 | 2.7694 | 2.5641 | 3.7632 |
| fused | stochastic_rounding | 0.4270 | 0.4076 | 0.5294 |
| fused | kahan8 | 0.4485 | 0.4260 | 0.6031 |

## LoRA bag 512x r=16 dims 320-1280


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 380.7949 | 336.7158 | 443.0100 |
| native | kahan | 418.1576 | 364.1313 | 478.1619 |
| native | kahan8 | 384.7982 | 339.3352 | 456.3876 |
| foreach | stochastic_rounding | 524.6459 | 482.1688 | 624.9564 |
| foreach | kahan8 | 578.5477 | 485.7600 | 642.2845 |
| fused | stochastic_rounding | 110.9443 | 101.6463 | 128.0184 |
| fused | kahan8 | 107.8303 | 98.1535 | 134.1051 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 836.8886 | 749.3560 | 967.0144 |
| native | kahan | 858.8820 | 801.8391 | 985.4280 |
| native | kahan8 | 844.1969 | 756.2896 | 942.0093 |
| foreach | stochastic_rounding | 994.1217 | 849.3077 | 1118.0739 |
| foreach | kahan8 | 1115.2579 | 948.7667 | 1424.8223 |
| fused | stochastic_rounding | 165.8240 | 156.2859 | 195.7632 |
| fused | kahan8 | 176.0236 | 161.0383 | 226.7945 |

## UNet-ish 8x(1024,1024)+16x(4096,)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 18.9583 | 17.9425 | 19.9578 |
| native | kahan | 20.2373 | 17.9968 | 24.7419 |
| native | kahan8 | 18.6962 | 17.9507 | 21.5716 |
| foreach | stochastic_rounding | 4.7683 | 4.4954 | 6.6058 |
| foreach | kahan8 | 5.1420 | 4.7647 | 6.4123 |
| fused | stochastic_rounding | 0.6047 | 0.5806 | 0.7936 |
| fused | kahan8 | 0.6753 | 0.6595 | 0.7291 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 35.1217 | 29.0079 | 41.7403 |
| native | kahan | 36.9654 | 29.8639 | 41.1689 |
| native | kahan8 | 35.1447 | 31.4808 | 40.2156 |
| foreach | stochastic_rounding | 7.2166 | 7.0697 | 7.5295 |
| foreach | kahan8 | 7.4993 | 7.3298 | 7.7363 |
| fused | stochastic_rounding | 0.7214 | 0.6298 | 0.7752 |
| fused | kahan8 | 0.7762 | 0.6840 | 0.8141 |
