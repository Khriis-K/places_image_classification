| GPUs | Nodes | Global batch | Peak lr | Steps/epoch | Throughput (img/s, epochs 2–10) | Training (min) | Validation (min) | Total (min) | Speedup | Efficiency | Val top-1 | Val top-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 256 | 0.04 | 6,692 | 1,103 | 259.3 | 6.2 | 265.5 | 1.00× | 100.0% | 56.70% | 86.65% |
| 2 | 1 | 512 | 0.08 | 3,346 | 2,109 | 136.0 | 3.0 | 139.1 | 1.91× | 95.5% | 56.81% | 86.59% |
| 4 | 2 | 1,024 | 0.16 | 1,673 | 4,122 | 70.0 | 1.7 | 71.6 | 3.71× | 92.6% | 56.73% | 86.73% |
| 8 | 4 | 2,048 | 0.32 | 836 | 8,070 | 36.0 | 0.9 | 37.0 | 7.18× | 89.8% | 56.75% | 86.67% |

1-GPU validation time is estimated (37 s per epoch; see plot_scaling.py). Speedup and efficiency use total minutes (training + validation).
