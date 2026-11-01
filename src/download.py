"""Optional real-data fetch via yfinance. Requires internet and provider availability."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data import validate_bars


def main() -> None:
    p = argparse.ArgumentParser(description='Download historical daily bars for personal research only')
    p.add_argument('--symbol', default='SPY', help='Example: SPY')
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2026-10-30', help='Exclusive date; use the same date for reproducibility')
    p.add_argument('--out', default='data/spy_daily.csv')
    a = p.parse_args()
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit('Install optional downloader: pip install yfinance') from e
    data = yf.download(a.symbol, start=a.start, end=a.end, auto_adjust=False,
                       progress=False, group_by='column', threads=False)
    if data.empty:
        raise SystemExit('No data returned. Check provider/network/symbol/date range.')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index().rename(columns={'Datetime': 'Date'})
    data = validate_bars(data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False, date_format='%Y-%m-%d')
    print(f'Saved {len(data)} REAL provider-sourced rows to {out}. Cite Yahoo Finance/yfinance as data source.')

if __name__ == '__main__':
    main()
