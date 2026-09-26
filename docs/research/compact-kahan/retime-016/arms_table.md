| arm (shape/opt/mom/mode_method) | A median ms [IQR] | B median ms [IQR] | ratio B/A | pair ratios | flag |
|---|---|---|---|---|---|
| unet/adakaon/no_momentum/fused_stochastic_rounding | 0.8348 [0.7066,1.0291] | 0.7816 [0.7178,0.8899] | 0.936 | 0.988,0.891 |  |
| unet/adakaon/no_momentum/fused_kahan8 | 0.7895 [0.7127,0.9062] | 0.8120 [0.7578,0.8970] | 1.029 | 1.055,1.004 |  |
| unet/adakaon/no_momentum/fused_kahan16 | 0.8381 [0.7537,0.9155] | 0.9324 [0.8714,1.0383] | 1.112 | 1.140,1.087 | **REGRESSION** |
| unet/adakaon/no_momentum/foreach_stochastic_rounding | 4.9403 [4.6643,5.2613] | 4.9526 [4.6520,5.1384] | 1.002 | 1.035,0.973 |  |
| unet/adakaon/no_momentum/foreach_kahan8 | 5.1889 [4.9224,5.5532] | 8.4769 [7.8561,8.5934] | 1.634 | 1.702,1.571 | **REGRESSION** |
| unet/adakaon/no_momentum/foreach_kahan16 | 5.4559 [5.1487,5.8163] | 8.8707 [8.2637,9.0214] | 1.626 | 1.693,1.564 | **REGRESSION** |
| unet/adakaon/momentum_4bit/fused_stochastic_rounding | 0.9088 [0.8243,0.9871] | 0.9060 [0.7936,1.0885] | 0.997 | 1.072,0.922 |  |
| unet/adakaon/momentum_4bit/fused_kahan8 | 0.8801 [0.8151,0.9800] | 0.9454 [0.8182,1.0701] | 1.074 | 1.153,0.997 | watch |
| unet/adakaon/momentum_4bit/fused_kahan16 | 0.9162 [0.8356,1.0199] | 0.9958 [0.9155,1.1162] | 1.087 | 1.092,1.082 | **REGRESSION** |
| unet/adakaon/momentum_4bit/foreach_stochastic_rounding | 7.6416 [7.2755,8.3743] | 7.4143 [7.2438,7.7025] | 0.970 | 0.966,0.974 |  |
| unet/adakaon/momentum_4bit/foreach_kahan8 | 7.8582 [7.5663,9.5416] | 10.6911 [10.4919,10.9896] | 1.361 | 1.361,1.360 | **REGRESSION** |
| unet/adakaon/momentum_4bit/foreach_kahan16 | 8.1728 [7.8428,8.6620] | 11.1240 [10.8923,11.4115] | 1.361 | 1.337,1.387 | **REGRESSION** |
| unet/nekaon/-/fused_stochastic_rounding | 1.1510 [1.0066,1.4541] | 1.4126 [1.2472,1.7981] | 1.227 | 1.106,1.370 | **REGRESSION** |
| unet/nekaon/-/fused_kahan8 | 1.3071 [1.2073,1.6200] | 1.5731 [1.4797,1.9323] | 1.203 | 1.182,1.226 | **REGRESSION** |
| unet/nekaon/-/fused_kahan16 | 1.4541 [1.3885,1.7050] | 1.6914 [1.5729,2.1391] | 1.163 | 1.113,1.216 | **REGRESSION** |
| unet/nekaon/-/foreach_stochastic_rounding | 7.8088 [7.6268,8.8146] | 8.8404 [8.0599,11.1534] | 1.132 | 1.069,1.197 | **REGRESSION** |
| unet/nekaon/-/foreach_kahan8 | 8.2470 [8.0916,8.8105] | 11.8628 [11.5507,13.1502] | 1.438 | 1.453,1.424 | **REGRESSION** |
| unet/nekaon/-/foreach_kahan16 | 8.5914 [8.3978,9.2877] | 12.4851 [12.1006,13.2086] | 1.453 | 1.464,1.443 | **REGRESSION** |
| unet/adapnm/-/fused_stochastic_rounding | 2.0332 [1.2032,3.6137] | 1.6704 [1.4572,2.0541] | 0.822 | 0.616,1.225 |  |
| big/adakaon/no_momentum/fused_stochastic_rounding | 0.4767 [0.2959,0.6810] | 0.4460 [0.4045,0.5540] | 0.936 | 0.817,1.112 |  |
| big/adakaon/no_momentum/fused_kahan8 | 0.4623 [0.2857,0.7066] | 0.4163 [0.3768,0.5028] | 0.900 | 0.754,1.140 |  |
| big/adakaon/no_momentum/fused_kahan16 | 0.4116 [0.2652,0.5980] | 0.4024 [0.3686,0.4905] | 0.978 | 0.877,1.115 |  |
| big/adakaon/no_momentum/foreach_stochastic_rounding | 1.4510 [1.0086,1.9558] | 1.4098 [1.2247,1.7869] | 0.972 | 0.875,1.113 |  |
| big/adakaon/no_momentum/foreach_kahan8 | 1.4415 [1.0373,2.0449] | 1.7541 [1.5718,2.1944] | 1.217 | 1.082,1.412 | **REGRESSION** |
| big/adakaon/no_momentum/foreach_kahan16 | 1.4021 [1.0117,1.9220] | 1.7482 [1.5770,2.1668] | 1.247 | 1.117,1.426 | **REGRESSION** |
| big/adakaon/momentum_4bit/fused_stochastic_rounding | 0.4654 [0.3133,0.6339] | 0.6600 [0.3574,1.0496] | 1.418 | 0.772,2.264 | watch |
| big/adakaon/momentum_4bit/fused_kahan8 | 0.4342 [0.3082,0.5540] | 0.5509 [0.3553,0.9380] | 1.269 | 0.847,1.785 | watch |
| big/adakaon/momentum_4bit/fused_kahan16 | 0.4173 [0.2898,0.5837] | 0.5053 [0.3471,1.0035] | 1.211 | 0.836,1.663 | watch |
| big/adakaon/momentum_4bit/foreach_stochastic_rounding | 2.1798 [1.4100,2.8160] | 2.8989 [1.6148,5.3371] | 1.330 | 0.772,2.025 | watch |
| big/adakaon/momentum_4bit/foreach_kahan8 | 2.1417 [1.4500,2.8662] | 3.0372 [1.9517,5.7661] | 1.418 | 0.831,2.274 | watch |
| big/adakaon/momentum_4bit/foreach_kahan16 | 2.2226 [1.5176,2.6798] | 3.0822 [1.9640,4.8742] | 1.387 | 0.868,2.018 | watch |
| big/nekaon/-/fused_stochastic_rounding | 0.5169 [0.3942,0.7516] | 0.7683 [0.5530,1.0138] | 1.486 | 1.066,2.063 | **REGRESSION** |
| big/nekaon/-/fused_kahan8 | 0.5123 [0.4004,0.8100] | 0.6894 [0.5202,0.9800] | 1.346 | 0.969,1.880 | watch |
| big/nekaon/-/fused_kahan16 | 0.4785 [0.3799,0.7127] | 0.6392 [0.4864,0.8919] | 1.336 | 1.028,1.738 | watch |
| big/nekaon/-/foreach_stochastic_rounding | 2.0047 [1.4797,2.8242] | 2.9471 [1.8668,4.0663] | 1.470 | 0.991,2.179 | watch |
| big/nekaon/-/foreach_kahan8 | 2.1030 [1.5227,2.8621] | 3.2653 [2.2641,4.4032] | 1.553 | 1.094,2.163 | **REGRESSION** |
| big/nekaon/-/foreach_kahan16 | 2.0180 [1.5155,3.0044] | 3.1741 [2.2170,4.0643] | 1.573 | 1.166,2.082 | **REGRESSION** |
| big/adapnm/-/fused_stochastic_rounding | 0.8868 [0.7209,1.0516] | 1.0862 [0.7608,1.2841] | 1.225 | 1.002,1.519 | watch |
| lora/adakaon/no_momentum/fused_stochastic_rounding | 102.3672 [92.0791,126.7179] | 109.5634 [97.9855,122.3301] | 1.070 | 1.016,1.128 | watch |
| lora/adakaon/no_momentum/fused_kahan8 | 107.1849 [87.1895,131.9957] | 105.5488 [94.1056,136.8535] | 0.985 | 0.913,1.065 |  |
| lora/adakaon/no_momentum/fused_kahan16 | 103.9004 [88.6139,127.8464] | 107.6513 [95.0282,135.5039] | 1.036 | 0.987,1.086 |  |
| lora/adakaon/no_momentum/foreach_stochastic_rounding | 561.7498 [498.3316,621.3099] | 581.3161 [536.2166,660.5568] | 1.035 | 0.982,1.089 |  |
| lora/adakaon/no_momentum/foreach_kahan8 | 606.2154 [551.2079,685.8803] | 800.7396 [692.8885,959.4798] | 1.321 | 1.242,1.403 | **REGRESSION** |
| lora/adakaon/no_momentum/foreach_kahan16 | 578.2715 [524.2859,638.5255] | 787.9140 [678.6847,990.8859] | 1.363 | 1.334,1.390 | **REGRESSION** |
| lora/adakaon/momentum_4bit/fused_stochastic_rounding | 158.0321 [142.5941,179.6076] | 154.7835 [138.0792,197.4753] | 0.979 | 0.984,0.975 |  |
| lora/adakaon/momentum_4bit/fused_kahan8 | 152.1464 [136.2688,188.3607] | 171.2466 [153.4781,195.2676] | 1.126 | 1.138,1.113 | **REGRESSION** |
| lora/adakaon/momentum_4bit/fused_kahan16 | 159.8787 [143.6631,181.8470] | 168.7380 [150.0457,198.1379] | 1.055 | 1.038,1.072 | watch |
| lora/adakaon/momentum_4bit/foreach_stochastic_rounding | 993.5895 [876.7560,1075.2225] | 1037.4031 [946.5109,1197.8230] | 1.044 | 1.038,1.050 |  |
| lora/adakaon/momentum_4bit/foreach_kahan8 | 1033.2856 [944.2662,1165.5710] | 1212.4982 [1126.2146,1396.1176] | 1.173 | 1.165,1.182 | **REGRESSION** |
| lora/adakaon/momentum_4bit/foreach_kahan16 | 1036.6748 [938.6578,1156.9971] | 1210.6867 [1099.9603,1337.8314] | 1.168 | 1.185,1.151 | **REGRESSION** |
| lora/nekaon/-/fused_stochastic_rounding | 203.5023 [166.4256,237.2762] | 182.4922 [168.1900,211.7059] | 0.897 | 0.835,0.968 |  |
| lora/nekaon/-/fused_kahan8 | 197.1505 [178.0111,236.7601] | 204.7084 [165.1610,242.9471] | 1.038 | 0.912,1.178 |  |
| lora/nekaon/-/fused_kahan16 | 199.0863 [169.8755,240.1280] | 196.6075 [174.5295,256.5591] | 0.988 | 0.949,1.026 |  |
| lora/nekaon/-/foreach_stochastic_rounding | 1040.2209 [946.7587,1167.3210] | 1050.8544 [932.4534,1205.3801] | 1.010 | 0.982,1.038 |  |
| lora/nekaon/-/foreach_kahan8 | 1071.0827 [974.0872,1269.0432] | 1215.0126 [1108.4688,1431.8593] | 1.134 | 1.134,1.135 | **REGRESSION** |
| lora/nekaon/-/foreach_kahan16 | 1096.7434 [972.1415,1291.7330] | 1230.7264 [1049.2150,1429.0688] | 1.122 | 1.094,1.150 | **REGRESSION** |
| lora/adapnm/-/fused_stochastic_rounding | 326.8063 [267.6152,399.4511] | 333.2861 [281.1965,424.0353] | 1.020 | 1.003,1.034 |  |
