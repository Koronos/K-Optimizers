
## big 2x(1024,1200)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 1.3604 | 1.2462 | 1.5780 |
| native | kahan | 1.4223 | 1.3404 | 1.7357 |
| native | kahan8 | 1.3972 | 1.2728 | 1.6148 |
| foreach | stochastic_rounding | 0.8550 | 0.8141 | 1.0721 |
| foreach | kahan8 | 0.8760 | 0.8479 | 1.0752 |
| fused | stochastic_rounding | 0.4623 | 0.3236 | 0.5622 |
| fused | kahan8 | 0.4572 | 0.3205 | 0.5519 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 2.7884 | 2.3398 | 3.7990 |
| native | kahan | 2.8964 | 2.3511 | 3.7816 |
| native | kahan8 | 2.7397 | 2.3951 | 3.3126 |
| foreach | stochastic_rounding | 1.6517 | 1.4377 | 2.2477 |
| foreach | kahan8 | 1.6527 | 1.4602 | 2.3306 |
| fused | stochastic_rounding | 0.3087 | 0.2744 | 0.5304 |
| fused | kahan8 | 0.3241 | 0.2867 | 0.5683 |

## LoRA bag 512x r=16 dims 320-1280


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 412.0950 | 342.1962 | 485.1763 |
| native | kahan | 459.2753 | 379.3357 | 489.7126 |
| native | kahan8 | 434.9686 | 374.6028 | 466.9798 |
| foreach | stochastic_rounding | 538.7540 | 491.3408 | 593.1571 |
| foreach | kahan8 | 559.5095 | 520.0118 | 642.9205 |
| fused | stochastic_rounding | 88.8305 | 77.6940 | 96.9943 |
| fused | kahan8 | 90.2477 | 81.0363 | 106.4632 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 852.3177 | 783.0354 | 911.1317 |
| native | kahan | 886.7896 | 768.3502 | 975.2965 |
| native | kahan8 | 876.5532 | 742.6601 | 903.9268 |
| foreach | stochastic_rounding | 1096.4936 | 1010.1268 | 1160.1531 |
| foreach | kahan8 | 1160.6661 | 1085.6407 | 1259.4545 |
| fused | stochastic_rounding | 155.8477 | 130.1289 | 181.6791 |
| fused | kahan8 | 161.2605 | 132.7636 | 175.0006 |

## UNet-ish 8x(1024,1024)+16x(4096,)


### no_momentum (beta1=0)

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 11.8328 | 11.1821 | 14.2397 |
| native | kahan | 13.7718 | 12.6505 | 16.7096 |
| native | kahan8 | 12.0003 | 10.9496 | 13.5741 |
| foreach | stochastic_rounding | 4.1713 | 4.1093 | 4.2578 |
| foreach | kahan8 | 4.4785 | 4.4042 | 4.6162 |
| fused | stochastic_rounding | 0.5868 | 0.5396 | 0.6298 |
| fused | kahan8 | 0.6733 | 0.6042 | 0.6943 |

### momentum_4bit

| mode | method | median ms | q1 ms | q3 ms |
|---|---|---|---|---|
| native | stochastic_rounding | 28.7555 | 25.4730 | 40.2289 |
| native | kahan | 31.9657 | 26.3137 | 39.7292 |
| native | kahan8 | 30.7722 | 25.2836 | 43.3265 |
| foreach | stochastic_rounding | 9.4254 | 8.7859 | 11.8682 |
| foreach | kahan8 | 10.1299 | 8.4818 | 11.0868 |
| fused | stochastic_rounding | 0.7148 | 0.6656 | 0.8212 |
| fused | kahan8 | 0.7516 | 0.7168 | 0.8847 |
