# Market Leakage Lab — Reproducible Next-Day Direction Classification

A small **Python machine-learning experiment**, designed to demonstrate **point-in-time features, chronological train/validation/test splits, embargoed walk-forward validation, automated tests, honest baselines and reproducible reporting**.

**Scope:** Predict whether the **next daily closing price** will be above today's close, using only information known **after today's market close**. This is **not** high-frequency trading, order-book forecasting, an automated trading bot, or evidence of a profitable strategy.

## What it demonstrates

- Input validation of `Date, Open, High, Low, Close, Volume` and sorted, unique dates
- Nine features from current and historical bars only (returns, rolling volatility, range, volume)
- Next-session binary target (`Close[t+1] > Close[t]`) with missing final label excluded
- 60% train / 20% validation / 20% untouched test, by date, with a one-bar label embargo at each boundary
- Expanding-window **TimeSeriesSplit(gap=1)** on the training period
- Majority-class **DummyClassifier** baseline plus Logistic Regression and Random Forest
- Model selection from validation balanced accuracy **only**, followed by a single final test evaluation
- Accuracy, balanced accuracy, precision, recall, F1, optional ROC AUC, confusion matrices
- Dataset SHA-256, seed, versions, periods, fold scores and limitation reporting
- `pytest` tests and GitHub Actions CI

### Architecture

```text
Real market CSV (optional) ─────┐
                                 ├─> OHLCV validation ─> point-in-time features
Explicitly synthetic CSV ───────┘                             │
                                                             ▼
                                             next-close label, drop unknown end
                                                             │
                                                             ▼
                                             train ┆ embargo ┆ val ┆ embargo ┆ test
                                                │                │                │
                                                ├─ expanding CV ─┤                │
                                                └─ fit candidates┘                │
                                                        │                          │
                                            choose using validation ONLY           │
                                                        │                          │
                                           refit before test, embargo ───────────> measure once
                                                                                │
                                                                          JSON + REPORT
```

## Quick start: offline synthetic **demonstration**

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m src.data --demo data/synthetic_demo.csv
python -m pytest -q
python -m src.experiment --data data/synthetic_demo.csv --source "SYNTHETIC simulated OHLCV, seed 42" --out outputs/demo
```

On Linux/macOS replace activation with `source .venv/bin/activate`.

Inspect `outputs/demo/REPORT.md` and `outputs/demo/results.json`.

**Important:** Synthetic results are for validating the code and explaining splits. They do **not** demonstrate a market forecasting result. Never show them as actual financial-data performance.

## Real market dataset (recommended before sharing with an employer)

Option A: Put your own legitimately obtained daily OHLCV CSV at `data/spy_daily.csv`. Its headers must be:

```csv
Date,Open,High,Low,Close,Volume
```

Option B: Download with the optional `yfinance` package (requires an internet connection and working upstream service):

```powershell
python -m pip install yfinance
python -m src.download --symbol SPY --start 2015-01-01 --end 2026-10-30 --out data/spy_daily.csv
python -m src.experiment --data data/spy_daily.csv --source "Yahoo Finance daily SPY prices via yfinance (unadjusted)" --out outputs/spy
```

**Data disclosure:** SPY is an ETF; this is publicly available *daily bar data*, not tick-by-tick bid/ask order-book data. No downloaded market data is included in this repository. Downloaded data is gitignored. Check your data provider's terms and retain a local copy when reproducing a specific run. Calendar year range and provider adjustments are fixed in the command for repeatability, but corporate actions and upstream historical corrections can still affect the dataset; `results.json` includes its hash.

## Why random cross-validation is inappropriate

Random splits can train on chronologically later price observations when predicting earlier observations. In a nonstationary market, that creates an unrealistic evaluation of information available at prediction time. A more subtle issue is **label overlap**: at day *t*, our target uses the close at *t+1*. The final training label cannot use a price observed in the first validation/test sample. We therefore remove a **one-bar gap** at period boundaries and use a `TimeSeriesSplit(gap=1)` expanding-window procedure inside training.

The baseline helps show if model complexity adds real value. Since finance labels can be imbalanced, we use **balanced accuracy** to choose models instead of relying on raw accuracy alone. Even a validation winner can fail on the untouched test period. That is a useful result, not a failure of the project.

## Explicitly checked leakage risks

| Risk | Control |
| --- | --- |
| Train on future observations | Date-sorted chronological partitions |
| Feature at day *t* uses future price | Only current / lagged rolling features; perturbation test |
| Last observation's missing label becomes "down" | Explicit unknown target, removed |
| Train's next-day label overlaps first validation/test day | One-bar embargo + assertions on target end timestamps |
| StandardScaler fitted on all rows | Scaler lives in `sklearn.pipeline`; fitted within each fold |
| Multiple models chosen using test outcome | Model selected by validation only; test checked once |
| Big accuracy result without baseline | Majority baseline and balanced accuracy |

## Understanding and explaining the experiment

Be ready to answer: Why do we make the prediction **after** each daily close? Why is today's volume available at that time? Why is tomorrow's close unavailable? How does the one-bar embargo work? Why does a random K-fold split provide misleading results? Why did validation choose this model? Did it beat the baseline on unseen data? What changed across CV folds? What might fail if applied to real order-book events?

**Limitations:** Daily bars conceal intraday bid/ask spreads, market depth, event times, execution costs, queue priority, latency and fills. Historical records may have survivorship/adjustment issues. A single ticker and split cannot establish stable predictive performance. Repeated tuning based on the final test set would invalidate the holdout. No trading or profitability claims.

## Repository map

```text
src/data.py         # Validation + explicitly synthetic offline demo
src/download.py     # Optional real OHLCV download
src/features.py     # Past/current features + next-day target
src/experiment.py   # CV, holdouts, model selection, metrics, results
src/__init__.py
tests/test_pipeline.py
.github/workflows/tests.yml
requirements.txt
requirements-dev.txt
```

## Provenance and credit

This is a compact reproducible experiment template prepared with AI assistance. To present it as evidence of personal engineering skill, **run it on real data, inspect the source, understand the leakage controls, record what happened and make your own meaningful improvements or analysis**. Do not imply you independently authored or validated code that you have not personally worked through. Report failures and unpromising results rather than hiding them.

## Next extension (not required)

Move to an appropriately licensed **limit-order-book events dataset**, reconstruct mid-price from best bid/ask updates, define an event-time prediction horizon, implement purged/embargoed walk-forward splits accounting for overlapping labels, and evaluate out-of-sample calibration and transaction costs. Do not call the daily-bar baseline HFT.
