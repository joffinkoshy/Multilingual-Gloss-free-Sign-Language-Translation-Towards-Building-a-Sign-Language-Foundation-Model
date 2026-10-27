## test — baseline (Sign2(LID+Text)) vs. proposed (+LID self-conditioning), 3 seed(s)

| Metric | Baseline | Proposed |
|---|---|---|
| BLEU-4 | 3.04 ± 0.77 | 3.62 ± 0.28 |
| ROUGE-L | 16.16 ± 1.71 | 17.89 ± 0.15 |
| LID utterance acc. (%) | 91.11 ± 6.84 | 98.33 ± 0.92 |
| Learned gate g | – | 0.120 |
| Trainable params | 18,556,192 | 18,559,777 |

| Lang | Baseline BLEU | Proposed BLEU | Δ |
|---|---|---|---|
| bfi | 3.18 ± 0.56 | 4.00 ± 0.58 | +0.82 |
| bqn | 2.20 ± 0.44 | 2.29 ± 0.21 | +0.09 |
| csl | 3.05 ± 0.74 | 3.64 ± 0.72 | +0.60 |
| gsg | 2.59 ± 0.44 | 3.74 ± 0.82 | +1.15 |
| icl | 3.20 ± 1.65 | 3.58 ± 0.58 | +0.38 |
| ise | 2.45 ± 0.41 | 3.48 ± 0.62 | +1.03 |
| lls | 3.09 ± 1.12 | 4.15 ± 0.26 | +1.05 |
| rsl | 3.13 ± 0.64 | 3.25 ± 1.03 | +0.12 |
| swl | 3.46 ± 1.47 | 3.92 ± 0.90 | +0.46 |
| ukl | 3.25 ± 0.99 | 3.47 ± 0.39 | +0.22 |

Paired bootstrap (proposed − baseline BLEU):

- seed1: Δ=-0.34, 95% CI [-0.83, +0.13], p=0.918
- seed2: Δ=+0.68, 95% CI [+0.16, +1.20], p=0.005
- seed3: Δ=+1.44, 95% CI [+0.99, +1.93], p=0.000
