
## big 2x(1024,1200)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 1.7531 | 1.6681 | 2.1494 |
| native | kahan | 1.9241 | 1.7500 | 2.3552 |
| native | kahan8 | 1.7987 | 1.6671 | 2.0040 |
| foreach | stochastic_rounding | 1.5375 | 1.4418 | 1.9415 |
| foreach | kahan8 | 1.6051 | 1.5288 | 1.8893 |
| fused | stochastic_rounding | 0.3441 | 0.3256 | 0.3912 |
| fused | kahan8 | 0.3517 | 0.3420 | 0.4321 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 4.3013 | 3.8216 | 5.1753 |
| native | kahan | 4.1620 | 3.8267 | 5.4641 |
| native | kahan8 | 4.4959 | 3.9096 | 5.7508 |
| foreach | stochastic_rounding | 2.7105 | 2.4033 | 3.2215 |
| foreach | kahan8 | 2.7771 | 2.4392 | 3.0689 |
| fused | stochastic_rounding | 0.4772 | 0.4372 | 0.6400 |
| fused | kahan8 | 0.4741 | 0.4383 | 0.6318 |

## LoRA bag 512x r=16 dims 320-1280


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 396.4037 | 322.0531 | 501.9208 |
| native | kahan | 427.8313 | 311.0810 | 605.9213 |
| native | kahan8 | 398.6452 | 293.1743 | 507.4453 |
| foreach | stochastic_rounding | 552.9180 | 436.2773 | 594.3921 |
| foreach | kahan8 | 582.4876 | 508.9782 | 663.9554 |
| fused | stochastic_rounding | 119.7962 | 91.8794 | 132.9848 |
| fused | kahan8 | 109.4630 | 91.9992 | 121.0675 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 741.3663 | 704.8765 | 803.4427 |
| native | kahan | 759.3144 | 738.4832 | 827.4299 |
| native | kahan8 | 748.6070 | 713.3420 | 803.2205 |
| foreach | stochastic_rounding | 875.3387 | 850.5344 | 921.5037 |
| foreach | kahan8 | 905.0952 | 883.1703 | 953.9789 |
| fused | stochastic_rounding | 155.5420 | 140.8082 | 179.8144 |
| fused | kahan8 | 155.6884 | 141.7677 | 177.3599 |

## UNet-ish 8x(1024,1024)+16x(4096,)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 16.7762 | 11.4555 | 23.9718 |
| native | kahan | 16.7347 | 13.1277 | 21.7743 |
| native | kahan8 | 16.0794 | 12.3443 | 19.6721 |
| foreach | stochastic_rounding | 4.6679 | 4.6305 | 5.5439 |
| foreach | kahan8 | 5.0616 | 4.9490 | 5.9105 |
| fused | stochastic_rounding | 2.6563 | 1.2923 | 2.7105 |
| fused | kahan8 | 2.9696 | 1.0373 | 3.0106 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 29.4292 | 26.4028 | 32.5847 |
| native | kahan | 29.8368 | 27.0684 | 33.5063 |
| native | kahan8 | 28.9132 | 26.2124 | 32.3072 |
| foreach | stochastic_rounding | 8.1347 | 8.0906 | 8.2227 |
| foreach | kahan8 | 8.4357 | 8.3866 | 8.5873 |
| fused | stochastic_rounding | 3.3628 | 1.1643 | 3.4038 |
| fused | kahan8 | 3.5702 | 1.2083 | 3.6413 |
