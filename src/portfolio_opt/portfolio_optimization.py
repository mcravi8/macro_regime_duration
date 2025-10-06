import numpy as np
import pandas as pd
from config import TARGET_DURATION_RANGE, MAX_WEIGHT, TURNOVER_PENALTY, UNIVERSE

def _covariance_stub(expected_returns):
    vols = pd.Series([0.03,0.05,0.08,0.12], index=["UST_2Y","UST_5Y","UST_10Y","UST_30Y"])
    C = np.diag(vols.values**2)
    return pd.DataFrame(C, index=vols.index, columns=vols.index)

def _optimize_er_sharpe(mu, cov, target_duration_range, max_weight):
    assets = mu.index.tolist()
    dur = np.array([UNIVERSE[a] for a in assets])
    n = len(assets)
    w = np.ones(n)/n

    def stats(w):
        ret = float(mu.values @ w)
        vol = float(np.sqrt(w @ cov.values @ w))
        dur_p = float(dur @ w)
        return ret, vol, dur_p

    lr = 0.01
    for _ in range(500):
        ret, vol, _ = stats(w)
        if vol <= 1e-8:
            grad = mu.values
        else:
            grad = (mu.values*vol - ret*(cov.values @ w)/vol) / (vol**2 + 1e-12)
        w += lr*grad
        w = np.maximum(w, 0); w = w / w.sum()
        w = np.minimum(w, max_weight); w = w / w.sum()

        lo, hi = target_duration_range
        dur_p = float(dur @ w)
        if dur_p < lo:
            # tilt longer
            idx = np.argsort(dur)
            for i in idx[::-1]:
                if dur_p >= lo: break
                add = min(0.01, max_weight - w[i])
                w[i] += add
                w = w / w.sum()
                dur_p = float(dur @ w)
        elif dur_p > hi:
            # tilt shorter
            idx = np.argsort(dur)
            for i in idx:
                if dur_p <= hi: break
                red = min(0.01, w[i])
                w[i] -= red
                w = np.maximum(w, 0); w = w / w.sum()
                dur_p = float(dur @ w)
    return pd.Series(w, index=assets)

def backtest_strategy(expected_returns):
    exp_rets = expected_returns.copy().sort_index()
    cov = _covariance_stub(exp_rets)
    weights = []
    prev_w = None
    realized = []

    for t in range(1, len(exp_rets)):
        mu = exp_rets.iloc[t-1]
        w = _optimize_er_sharpe(mu, cov, TARGET_DURATION_RANGE, MAX_WEIGHT)
        turn_cost = 0.0 if prev_w is None else float((w - prev_w).abs().sum()) * TURNOVER_PENALTY
        weights.append((exp_rets.index[t], w))
        realized_ret = float(exp_rets.iloc[t] @ w) - turn_cost
        realized.append({"date": exp_rets.index[t], "strategy": realized_ret})
        prev_w = w

    weights_df = pd.DataFrame({dt: w for dt, w in weights}).T
    weights_df.index.name = "date"
    bm_w = pd.Series([0.2,0.35,0.30,0.15], index=exp_rets.columns)
    backtest = pd.DataFrame(realized).set_index("date")
    backtest["benchmark"] = (exp_rets.iloc[1:] @ bm_w).values
    backtest["excess"] = backtest["strategy"] - backtest["benchmark"]
    return backtest, weights_df