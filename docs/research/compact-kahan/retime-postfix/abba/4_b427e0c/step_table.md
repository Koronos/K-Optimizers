
## big 2x(1024,1200)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 1.9139 | 1.4715 | 2.4525 |
| native | kahan | 1.7234 | 1.4817 | 2.1565 |
| native | kahan8 | 1.7070 | 1.3865 | 2.2856 |
| foreach | stochastic_rounding | 1.4730 | 1.2186 | 2.3542 |
| foreach | kahan8 | 1.6707 | 1.3711 | 2.2241 |
| fused | stochastic_rounding | 0.6467 | 0.4966 | 0.9144 |
| fused | kahan8 | 0.6789 | 0.4772 | 0.9789 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 2.9030 | 2.4433 | 3.6188 |
| native | kahan | 2.9624 | 2.5088 | 3.6106 |
| native | kahan8 | 2.8780 | 2.3634 | 3.5953 |
| foreach | stochastic_rounding | 1.5611 | 1.4213 | 1.7039 |
| foreach | kahan8 | 1.5539 | 1.4459 | 1.7172 |
| fused | stochastic_rounding | 0.2473 | 0.2437 | 0.2560 |
| fused | kahan8 | 0.2529 | 0.2499 | 0.2642 |

## LoRA bag 512x r=16 dims 320-1280


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 434.7254 | 389.0197 | 490.6875 |
| native | kahan | 440.2657 | 364.5686 | 485.7303 |
| native | kahan8 | 408.9492 | 350.5510 | 472.3487 |
| foreach | stochastic_rounding | 525.4758 | 435.1959 | 634.1960 |
| foreach | kahan8 | 594.3475 | 467.7816 | 667.7801 |
| fused | stochastic_rounding | 107.7868 | 92.2317 | 121.5764 |
| fused | kahan8 | 99.6603 | 92.0003 | 118.9110 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 761.9656 | 707.1867 | 871.4312 |
| native | kahan | 874.8221 | 704.9912 | 967.5059 |
| native | kahan8 | 776.8545 | 711.8459 | 904.3477 |
| foreach | stochastic_rounding | 1051.3392 | 973.8301 | 1239.3278 |
| foreach | kahan8 | 1151.6498 | 996.7811 | 1384.7133 |
| fused | stochastic_rounding | 138.9512 | 128.1464 | 152.0384 |
| fused | kahan8 | 132.0781 | 122.4509 | 146.1156 |

## UNet-ish 8x(1024,1024)+16x(4096,)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 11.5835 | 10.6004 | 15.2945 |
| native | kahan | 12.2778 | 10.5226 | 14.7569 |
| native | kahan8 | 11.9352 | 10.1478 | 15.6201 |
| foreach | stochastic_rounding | 4.0750 | 3.7530 | 4.1503 |
| foreach | kahan8 | 4.3628 | 3.9578 | 4.4442 |
| fused | stochastic_rounding | 0.5468 | 0.5274 | 0.5601 |
| fused | kahan8 | 0.6052 | 0.5888 | 0.6154 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 25.5252 | 22.3826 | 28.0064 |
| native | kahan | 27.0336 | 23.5684 | 33.1233 |
| native | kahan8 | 25.2938 | 23.9063 | 27.3367 |
| foreach | stochastic_rounding | 6.5475 | 6.3969 | 6.9396 |
| foreach | kahan8 | 6.8137 | 6.7103 | 7.3871 |
| fused | stochastic_rounding | 0.6129 | 0.6042 | 0.6451 |
| fused | kahan8 | 0.6605 | 0.6533 | 0.6758 |
