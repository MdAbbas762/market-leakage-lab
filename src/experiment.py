"""Leakage-safe supervised daily direction experiment with untouched chronological holdout."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import load_csv
from .features import FEATURES, HORIZON, make_supervised

SEED = 42
GAP = HORIZON


def chronological_splits(df: pd.DataFrame, *, train_fraction: float = .6,
                         validation_fraction: float = .2):
    """Disjoint ordered splits with an embargo row before validation and test."""
    n = len(df)
    tr_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    train = np.arange(0, tr_end - GAP)
    val = np.arange(tr_end, val_end - GAP)
    test = np.arange(val_end, n)
    if min(len(train), len(val), len(test)) < 50:
        raise ValueError('Each chronological split needs at least 50 samples')
    assert max(train) + GAP < min(val)
    assert max(val) + GAP < min(test)
    return train, val, test


def models():
    return {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        'random_forest': RandomForestClassifier(n_estimators=120, max_depth=5,
                                                min_samples_leaf=15, random_state=SEED, n_jobs=1),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    out = {
        'accuracy': round(float(accuracy_score(y, pred)), 5),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, pred)), 5),
        'precision': round(float(precision_score(y, pred, zero_division=0)), 5),
        'recall': round(float(recall_score(y, pred, zero_division=0)), 5),
        'f1': round(float(f1_score(y, pred, zero_division=0)), 5),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'observations': int(len(y)),
        'positive_rate': round(float(np.mean(y)), 5),
    }
    if len(np.unique(y)) > 1 and hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            out['roc_auc'] = round(float(roc_auc_score(y, probs[:, 1])), 5)
    return out


def validate_no_target_overlap(df, train_idx, next_idx) -> None:
    """No training label may use a price observed on or after the first next-split timestamp."""
    last_target_observation = df.iloc[train_idx]['target_end_date'].max()
    first_prediction_timestamp = df.iloc[next_idx]['Date'].min()
    if not (last_target_observation < first_prediction_timestamp):
        raise ValueError('Future-data leakage at split boundary: label reaches next split')


def walk_forward_cv(model, X, y, dates, target_end_dates, folds: int = 4) -> list:
    tscv = TimeSeriesSplit(n_splits=folds, gap=GAP)
    scores = []
    for i, (train, valid) in enumerate(tscv.split(X), start=1):
        if not (target_end_dates.iloc[train].max() < dates.iloc[valid].min()):
            raise ValueError('Future target contamination in walk-forward split')
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        m = evaluate(fitted, X.iloc[valid], y.iloc[valid])
        scores.append({'fold': i, 'train_end': str(dates.iloc[train[-1]].date()),
                       'validation_start': str(dates.iloc[valid[0]].date()),
                       'balanced_accuracy': m['balanced_accuracy'],
                       'accuracy': m['accuracy']})
    return scores


def run(csv_path: str, output_dir: str, *, source: str) -> dict:
    bars = load_csv(csv_path)
    data = make_supervised(bars)
    tr, va, te = chronological_splits(data)
    validate_no_target_overlap(data, tr, va)
    validate_no_target_overlap(data, va, te)
    X = data.loc[:, FEATURES]
    y = data['target']
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xv, yv = X.iloc[va], y.iloc[va]
    specs = models()
    comparison = {}
    for name, spec in specs.items():
        cv = walk_forward_cv(spec, Xtr, ytr, data['Date'].iloc[tr],
                             data['target_end_date'].iloc[tr])
        fitted = clone(spec).fit(Xtr, ytr)
        comparison[name] = {
            'walk_forward_cv': cv,
            'cv_balanced_accuracy_mean': round(float(np.mean([x['balanced_accuracy'] for x in cv])), 5),
            'validation': evaluate(fitted, Xv, yv),
        }
    # Choose strictly from the pre-test validation set; no test-driven tuning.
    # Baseline is eligible: it is legitimate for no ML model to outperform it.
    names = list(specs)
    chosen = max(names, key=lambda name: (comparison[name]['validation']['balanced_accuracy'], -names.index(name)))
    # Refit on all available pretest history, removing the last row whose label
    # looks ahead into the first held-out test date.
    final_idx = np.arange(0, te[0] - GAP)
    validate_no_target_overlap(data, final_idx, te)
    final_model = clone(specs[chosen]).fit(X.iloc[final_idx], y.iloc[final_idx])
    final_test = evaluate(final_model, X.iloc[te], y.iloc[te])
    benchmark = clone(specs['majority_baseline']).fit(X.iloc[final_idx], y.iloc[final_idx])
    baseline_test = evaluate(benchmark, X.iloc[te], y.iloc[te])
    dataset_digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    report = {
        'experiment': 'Next-day close direction from after-close daily OHLCV',
        'data_source': source,
        'data_file': str(Path(csv_path).name),
        'sha256': dataset_digest,
        'market_granularity': 'daily OHLCV; NOT tick/order-book/HFT data',
        'prediction_time': 'after day t close',
        'label': '1 if Close(t+1) > Close(t), else 0',
        'feature_columns': list(FEATURES),
        'seed': SEED,
        'label_horizon_bars': HORIZON,
        'embargo_bars': GAP,
        'versions': {'python_sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'rows': {'raw': len(bars), 'labeled': len(data), 'train': len(tr),
                 'validation': len(va), 'test': len(te)},
        'windows': {name: {'start': str(data.iloc[ix[0]]['Date'].date()),
                           'end': str(data.iloc[ix[-1]]['Date'].date())}
                    for name, ix in [('train', tr), ('validation', va), ('test', te)]},
        'models': comparison,
        'selected_by_validation_only': chosen,
        'held_out_test': final_test,
        'held_out_baseline': baseline_test,
        'limitations': [
            'Daily OHLCV data are not limit-order-book / HFT data.',
            'No transaction costs, slippage, spread, queue or fills modeled.',
            'Single symbol/time horizon; regime changes can break generalization.',
            'Results on synthetic data say nothing about real market predictability.',
            'Testing multiple variations against the final test set would invalidate it.',
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    markdown = [
        '# Experiment report', '',
        f"Dataset: **{source}**; SHA-256: `{dataset_digest}`.",
        f"Task: predict next daily close direction at end of day t. No trading claims.", '',
        '## Chronological windows (with 1-bar embargo)', '',
        *[f"- {name}: {v['start']} to {v['end']} ({report['rows'][name]} rows)" for name,v in report['windows'].items()],
        '', '## Model selection (pre-test validation only)', '',
        '| Model | Walk-forward CV balanced accuracy | Validation balanced accuracy |',
        '| --- | ---: | ---: |',
        *[f"| {name} | {v['cv_balanced_accuracy_mean']:.3f} | {v['validation']['balanced_accuracy']:.3f} |" for name, v in comparison.items()],
        '', f"Selected: **{chosen}** (highest validation balanced accuracy).", '',
        '## Untouched chronological test', '',
        '| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {nm} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |" for nm,m in [('Selected', final_test),('Majority baseline', baseline_test)]],
        '', '## What went wrong / what remains uncertain', '',
        '- Weak out-of-sample prediction is entirely possible and is an acceptable result.',
        '- Selection on validation can overfit when many experiments are tried.',
        '- Features based on the completed daily bar cannot be used before that close.',
        '- This is not order-book modeling or evidence of a profitable trading strategy.',
        '- Before expanding: test on different symbols, regimes, and truly point-in-time market data.',
        '', '## Reproducibility', '',
        f"- SHA-256 dataset: `{dataset_digest}`",
        f"- Seed: {SEED}; sklearn: {sklearn.__version__}; horizon: {HORIZON}; embargo: {GAP}",
        '- Run: `python -m src.experiment --data <CSV> --source <description>`',
        '- Run tests: `python -m pytest -q`',
        ''
    ]
    (out / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Input CSV with Date/Open/High/Low/Close/Volume')
    p.add_argument('--out', default='outputs')
    p.add_argument('--source', required=True, help='Must say real provider or explicitly synthetic')
    a = p.parse_args()
    result = run(a.data, a.out, source=a.source)
    print(f"Selected: {result['selected_by_validation_only']}")
    print(f"Test balanced accuracy: {result['held_out_test']['balanced_accuracy']:.3f}; "
          f"baseline: {result['held_out_baseline']['balanced_accuracy']:.3f}")
    print(f"Saved {a.out}/results.json and {a.out}/REPORT.md")

if __name__ == '__main__':
    main()
