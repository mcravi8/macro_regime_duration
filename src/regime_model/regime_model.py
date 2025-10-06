import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from config import DATA_RAW

def _load_macro():
    df = pd.read_csv(DATA_RAW / "macro_data.csv", parse_dates=["date"], index_col="date")
    df = df[["GDP","UNRATE","CPI_INFL","ISM"]].dropna()
    return df

def _fit_markov(df):
    X = (df - df.mean()) / df.std()
    pc = np.linalg.svd(X, full_matrices=False)[0][:,0]
    model = MarkovRegression(pc, k_regimes=3, trend="c", switching_variance=True)
    res = model.fit(method="em", disp=False, maxiter=200)
    return res

def _label_regimes(df, probs):
    labels = []
    for k in probs.columns:
        mask = probs[k] > 0.5
        g = df.loc[mask, "GDP"].mean()
        inf = df.loc[mask, "CPI_INFL"].mean()
        if g > 0 and inf < df["CPI_INFL"].mean():
            name = "Expansion"
        elif g <= 0 and inf > df["CPI_INFL"].mean():
            name = "Stagflation"
        else:
            name = "Recession"
        labels.append({"state": k, "label": name})
    return pd.DataFrame(labels)

def estimate_regimes():
    df = _load_macro()
    res = _fit_markov(df)
    smoothed = res.smoothed_marginal_probabilities
    smoothed.columns = [f"state_{i}" for i in smoothed.columns]
    probs = smoothed.copy()
    labels_df = _label_regimes(df, probs)

    # Simple regime duration stats
    regime_series = probs.idxmax(axis=1)
    durations = []
    cur_state, cur_len = None, 0
    for s in regime_series:
        if s != cur_state:
            if cur_state is not None:
                durations.append({"state": cur_state, "months": cur_len})
            cur_state, cur_len = s, 1
        else:
            cur_len += 1
    if cur_state is not None:
        durations.append({"state": cur_state, "months": cur_len})
    regime_stats = pd.DataFrame(durations).groupby("state")["months"].agg(["count","mean","median"]).reset_index()

    return probs, labels_df, regime_stats