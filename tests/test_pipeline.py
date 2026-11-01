"""Tests for point-in-time features, target construction, and chronological isolation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.data import generate_synthetic, load_csv, validate_bars
from src.features import FEATURES, make_supervised
from src.experiment import chronological_splits, validate_no_target_overlap, walk_forward_cv, models

@pytest.fixture
def bars(tmp_path):
    f = tmp_path / 'fake.csv'
    generate_synthetic(f, n=600)
    return load_csv(f)


def test_synthetic_data_has_valid_ohlcv(bars):
    assert len(bars) == 600
    assert bars.Date.is_monotonic_increasing
    assert np.all(bars.High >= bars[['Open', 'Close']].max(axis=1))


def test_no_future_prices_in_features(bars):
    baseline = make_supervised(bars)
    changed = bars.copy()
    # Rewriting a future bar must not alter historical day-t features.
    changed.loc[450:, 'Close'] *= 1.1
    changed.loc[450:, 'High'] *= 1.15
    changed.loc[450:, 'Low'] *= .90
    modified = make_supervised(changed)
    pd.testing.assert_frame_equal(baseline.loc[:400, FEATURES], modified.loc[:400, FEATURES])


def test_label_is_next_close_and_last_unknown(bars):
    supervised = make_supervised(bars)
    sample = supervised.iloc[20]
    at = bars.index[bars.Date.eq(sample.Date)][0]
    expected = int(bars.Close.iloc[at+1] > bars.Close.iloc[at])
    assert sample.target == expected
    assert supervised.Date.max() < bars.Date.max()  # last bar never falsely marked down


def test_gaps_prevent_target_horizon_leakage(bars):
    data = make_supervised(bars)
    tr, va, te = chronological_splits(data)
    validate_no_target_overlap(data, tr, va)
    validate_no_target_overlap(data, va, te)
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te)


def test_walk_forward_cv_never_uses_future_labels(bars):
    data = make_supervised(bars)
    tr, _, _ = chronological_splits(data)
    scores = walk_forward_cv(models()['logistic_regression'], data.loc[tr, FEATURES],
                             data.loc[tr, 'target'], data.loc[tr, 'Date'],
                             data.loc[tr, 'target_end_date'])
    assert len(scores) == 4
    assert all(s['train_end'] < s['validation_start'] for s in scores)


def test_bad_data_is_rejected(bars):
    duplicated = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        validate_bars(duplicated)
    bad_ohlc = bars.copy()
    bad_ohlc.loc[0, 'High'] = bad_ohlc.loc[0, 'Low']
    with pytest.raises(ValueError, match='consistency'):
        validate_bars(bad_ohlc)
