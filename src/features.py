"""Point-in-time features available at the close of day t; target is close(t+1)>close(t)."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ('ret_1', 'ret_3', 'ret_5', 'volatility_5', 'volatility_20',
            'range_pct', 'close_to_open', 'volume_change', 'volume_rel_10')
HORIZON = 1


def make_supervised(bars: pd.DataFrame) -> pd.DataFrame:
    d = bars.copy()
    close = d['Close']
    pct = close.pct_change(fill_method=None)
    d['ret_1'] = pct
    d['ret_3'] = close.pct_change(3, fill_method=None)
    d['ret_5'] = close.pct_change(5, fill_method=None)
    d['volatility_5'] = pct.rolling(5).std()
    d['volatility_20'] = pct.rolling(20).std()
    d['range_pct'] = (d['High'] - d['Low']) / close
    d['close_to_open'] = (close - d['Open']) / d['Open']
    d['volume_change'] = d['Volume'].pct_change(fill_method=None)
    d['volume_rel_10'] = d['Volume'] / d['Volume'].rolling(10).mean() - 1
    # The final observation has no next-day label. Never turn its missing target into class 0.
    next_close = close.shift(-HORIZON)
    d['target'] = (next_close > close).where(next_close.notna(), np.nan)
    d['target_end_date'] = d['Date'].shift(-HORIZON)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, 'target', 'target_end_date'])
    d = d.reset_index(drop=True)
    d['target'] = d['target'].astype(int)
    return d.loc[:, ['Date', 'target_end_date', *FEATURES, 'target']]
