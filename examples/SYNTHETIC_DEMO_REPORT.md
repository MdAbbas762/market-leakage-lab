# Experiment report

Dataset: **SYNTHETIC simulated OHLCV, seed 42**; SHA-256: `e91e23a3f7bfe56f8c0e52b6481edaeb95fafe71f762b448f4ad0339b5cfd394`.
Task: predict next daily close direction at end of day t. No trading claims.

## Chronological windows (with 1-bar embargo)

- train: 2018-01-29 to 2021-09-13 (946 rows)
- validation: 2021-09-15 to 2022-11-29 (315 rows)
- test: 2022-12-01 to 2024-02-15 (316 rows)

## Model selection (pre-test validation only)

| Model | Walk-forward CV balanced accuracy | Validation balanced accuracy |
| --- | ---: | ---: |
| majority_baseline | 0.500 | 0.500 |
| logistic_regression | 0.534 | 0.572 |
| random_forest | 0.538 | 0.586 |

Selected: **random_forest** (highest validation balanced accuracy).

## Untouched chronological test

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected | 0.538 | 0.540 | 0.515 | 0.576 | 0.544 |
| Majority baseline | 0.522 | 0.500 | 0.000 | 0.000 | 0.000 |

## What went wrong / what remains uncertain

- Weak out-of-sample prediction is entirely possible and is an acceptable result.
- Selection on validation can overfit when many experiments are tried.
- Features based on the completed daily bar cannot be used before that close.
- This is not order-book modeling or evidence of a profitable trading strategy.
- Before expanding: test on different symbols, regimes, and truly point-in-time market data.

## Reproducibility

- SHA-256 dataset: `e91e23a3f7bfe56f8c0e52b6481edaeb95fafe71f762b448f4ad0339b5cfd394`
- Seed: 42; sklearn: 1.8.0; horizon: 1; embargo: 1
- Run: `python -m src.experiment --data <CSV> --source <description>`
- Run tests: `python -m pytest -q`
