| bf16_method | fused | state B/param | compensation B/param | first-step peak B/param |
|---|---|---|---|---|
| stochastic_rounding | False | 2.039 | 0.000 | 20.387 |
| stochastic_rounding | True | 2.039 | 0.000 | 2.053 |
| kahan | False | 4.039 | 2.000 | 6.200 |
| kahan8 | False | 3.039 | 1.000 | 21.263 |
| kahan8 | True | 3.039 | 1.000 | 3.053 |
| none | False | 2.039 | 0.000 | 21.130 |
| none | True | 2.039 | 0.000 | 21.130 |
