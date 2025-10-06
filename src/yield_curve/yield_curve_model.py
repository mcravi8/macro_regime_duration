import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from config import DATA_RAW, NS_TAU, VAR_LAGS

def _load_yields():
    y = pd.read_csv(DATA_RAW / "treasury_yields.csv", parse_dates=["date"], index_col="date")
    return y.dropna()

def _ns_loadings(maturities, tau):
    m = np.array(maturities, dtype=float)
    x = m / tau
    L1 = np.ones_like(x)
    L2 = (1 - np.exp(-x)) / x
    L3 = L2 - np.exp(-x)
    return np.vstack([L1, L2, L3]).T

def _fit_ns_for_date(yields_row, mats, tau=NS_TAU):
    y = yields_row.values.astype(float)
    X = _ns_loadings(mats, tau)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def fit_ns_and_forecast(regime_probs):
    y = _load_yields()
    mats = [24, 60, 120, 360]
    betas = []
    for _, row in y.iterrows():
        betas.append(_fit_ns_for_date(row, mats))
    ns = pd.DataFrame(betas, index=y.index, columns=["level","slope","curvature"]).dropna()

    aligned = ns.join(regime_probs, how="inner").dropna()
    exog_cols = [c for c in aligned.columns if c.startswith("state_")]
    ns_only = aligned[["level","slope","curvature"]]
    exog = aligned[exog_cols]

    var = VAR(ns_only)
    res = var.fit(VAR_LAGS, trend="c", exog=exog)

    steps = 6
    exog_future = exog.iloc[-1:].repeat(steps)
    forecasts = res.forecast(y=ns_only.values[-res.k_ar:], steps=steps, exog_future=exog_future.values)
    idx = pd.date_range(ns_only.index[-1], periods=steps+1, freq="M")[1:]
    f_df = pd.DataFrame(forecasts, index=idx, columns=ns_only.columns)

    # Expected returns proxy: carry - duration * delta_yield
    last_beta = ns_only.iloc[-1]
    X = _ns_loadings(mats, tau=NS_TAU)
    yld_last = X @ last_beta.values
    exp_rets = []
    for _, row in f_df.iterrows():
        yld_fore = X @ row.values
        dy = yld_fore - yld_last
        durations = np.array([2.0, 5.0, 10.0, 20.0])
        carry = np.maximum(yld_last, 0) / 12.0
        ret = carry - durations * dy
        exp_rets.append(ret)
    exp_rets = pd.DataFrame(exp_rets, index=f_df.index, columns=["UST_2Y","UST_5Y","UST_10Y","UST_30Y"])

    return ns, exp_rets