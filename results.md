# EASE results

Dataset: Mohler et al. (2011), 2273 answers, 81 questions, grades 0-5.


## 5-fold CV (random)

| System | Pearson | Spearman | MAE | RMSE |
|---|---|---|---|---|
| Similarity only (baseline) | 0.494 | 0.480 | 0.726 | 0.954 |
| Linear fusion (Ridge, 5 signals) | 0.606 | 0.643 | 0.646 | 0.878 |
| EASE (monotonic boosted trees, 5 signals) | 0.666 | 0.669 | 0.580 | 0.818 |

## 5-fold CV (unseen questions)

| System | Pearson | Spearman | MAE | RMSE |
|---|---|---|---|---|
| Similarity only (baseline) | 0.491 | 0.480 | 0.726 | 0.955 |
| Linear fusion (Ridge, 5 signals) | 0.606 | 0.644 | 0.646 | 0.878 |
| EASE (monotonic boosted trees, 5 signals) | 0.652 | 0.659 | 0.594 | 0.832 |

## Is the gain real?

Pearson gain of EASE over similarity-only: **+0.173** (bootstrap 95% CI +0.148 to +0.198, 1000 resamples).

## Mean EASE score by human grade band (unseen-question CV, out of 10)

| Human grade /10 | Answers | Mean human | Mean EASE |
|---|---|---|---|
| 0-4 | 171 | 3.0 | 6.3 |
| 5-7 | 463 | 6.1 | 7.2 |
| 8-9 | 518 | 8.6 | 8.1 |
| 10 | 1121 | 10.0 | 9.2 |

The order is correct in every band, but scores are compressed toward the dataset average: only 7.5% of Mohler answers are graded 4/10 or lower, so the model has few wrong answers to learn from.

## Ablation (drop one signal, random 5-fold CV)

| Signal removed | Pearson | MAE |
|---|---|---|
| none (full EASE) | 0.666 | 0.580 |
| Semantic similarity (Sentence-BERT) | 0.636 | 0.593 |
| Concept coverage (spaCy noun phrases) | 0.665 | 0.580 |
| NLI entailment probability | 0.619 | 0.617 |
| NLI contradiction probability | 0.660 | 0.582 |
| Keyword overlap (question words removed) | 0.632 | 0.605 |

## Contradiction flag check

| Contradiction prob. above | Answers flagged | Mean human grade | Flagged but human >= 4/5 |
|---|---|---|---|
| 0.5 | 477 | 3.86 | 63% |
| 0.7 | 367 | 3.87 | 62% |
| 0.9 | 215 | 3.80 | 59% |
| 0.99 | 46 | 3.22 | 48% |

Many high-contradiction answers were still graded well by humans, so a hard score cap would be wrong. EASE flags these answers for teacher review instead.

## Global signal importance (mean |Shapley value|, points out of 10)

| Signal | Importance |
|---|---|
| Keyword overlap (question words removed) | 0.58 |
| NLI entailment probability | 0.55 |
| Semantic similarity (Sentence-BERT) | 0.45 |
| NLI contradiction probability | 0.18 |
| Concept coverage (spaCy noun phrases) | 0.02 |
