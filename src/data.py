"""Load and validate daily OHLCV data, or generate explicitly synthetic demo bars."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')


def validate_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if not set(REQUIRED).issubset(raw.columns):
        raise ValueError(f'Missing columns: {sorted(set(REQUIRED) - set(raw.columns))}')
    frame = raw.loc[:, REQUIRED].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='raise')
    if frame['Date'].isna().any() or frame['Date'].duplicated().any():
        raise ValueError('Missing or duplicate trading dates')
    for c in REQUIRED[1:]:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    if frame[list(REQUIRED[1:])].isna().any().any():
        raise ValueError('Missing price or volume values')
    frame = frame.sort_values('Date', kind='stable').reset_index(drop=True)
    p = frame[['Open', 'High', 'Low', 'Close']]
    if not np.isfinite(p.to_numpy()).all() or (p <= 0).any().any():
        raise ValueError('Prices must be finite and > 0')
    if (frame['Volume'] < 0).any() or not np.isfinite(frame['Volume']).all():
        raise ValueError('Volume must be finite and nonnegative')
    if ((frame['High'] < frame[['Open', 'Close', 'Low']].max(axis=1) - 1e-8) |
        (frame['Low'] > frame[['Open', 'Close', 'High']].min(axis=1) + 1e-8)).any():
        raise ValueError('OHLC high/low consistency violation')
    if len(frame) < 350:
        raise ValueError('At least 350 trading-day rows required')
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_bars(pd.read_csv(path))


def generate_synthetic(path: str | Path, *, n: int = 1600, seed: int = 42) -> None:
    """Create fictional daily bars for offline smoke tests; NOT real financial data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.0018 * np.tanh(55 * r[t-1]) + rng.normal(0, 0.010)
    close = 100.0 * np.exp(np.cumsum(r))
    prev_close = np.r_[100.0, close[:-1]]
    op = prev_close * np.exp(rng.normal(0, 0.0018, n))
    intraday = rng.uniform(0.001, 0.013, n)
    high = np.maximum(op, close) * (1 + intraday)
    low = np.minimum(op, close) * (1 - intraday)
    volume = rng.integers(800_000, 5_000_000, n)
    df = pd.DataFrame({'Date':dates, 'Open':op, 'High':high, 'Low':low, 'Close':close, 'Volume':volume})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, date_format='%Y-%m-%d')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', required=True, help='Path to write explicitly synthetic OHLCV CSV')
    parser.add_argument('--rows', type=int, default=1600)
    args = parser.parse_args()
    generate_synthetic(args.demo, n=args.rows)
    print(f'SYNTHETIC demo saved to {args.demo}; not actual market history.')

if __name__ == '__main__':
    main()
