## dev — baseline (Sign2(LID+Text)) vs. proposed (+LID self-conditioning), 3 seed(s)

| Metric | Baseline | Proposed |
|---|---|---|
| BLEU-4 | 3.64 ± 1.18 | 4.95 ± 0.31 |
| ROUGE-L | 16.07 ± 2.34 | 19.36 ± 0.63 |
| LID utterance acc. (%) | 90.54 ± 7.19 | 98.54 ± 0.95 |
| Learned gate g | – | 0.120 |
| Trainable params | 18,556,192 | 18,559,777 |

| Lang | Baseline BLEU | Proposed BLEU | Δ |
|---|---|---|---|
| bfi | 4.19 ± 1.81 | 6.41 ± 1.19 | +2.22 |
| bqn | 1.97 ± 0.63 | 2.43 ± 0.80 | +0.46 |
| csl | 3.36 ± 1.92 | 4.01 ± 0.61 | +0.65 |
| gsg | 3.53 ± 1.07 | 4.50 ± 0.44 | +0.97 |
| icl | 3.91 ± 1.39 | 5.29 ± 1.23 | +1.39 |
| ise | 3.03 ± 0.49 | 4.19 ± 1.07 | +1.16 |
| lls | 4.22 ± 1.34 | 6.11 ± 0.99 | +1.89 |
| rsl | 3.75 ± 1.27 | 5.41 ± 0.55 | +1.66 |
| swl | 4.42 ± 1.54 | 5.47 ± 1.01 | +1.05 |
| ukl | 3.35 ± 0.62 | 4.81 ± 0.93 | +1.46 |

Paired bootstrap (proposed − baseline BLEU):

- seed1: Δ=-0.22, 95% CI [-1.17, +0.66], p=0.703
- seed2: Δ=+1.69, 95% CI [+0.93, +2.44], p=0.000
- seed3: Δ=+2.43, 95% CI [+1.58, +3.27], p=0.000
