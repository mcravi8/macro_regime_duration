import pandas as pd
import numpy as np

def sharpe(ret, rf=0.0):
    mu = ret.mean()*12
    sd = ret.std()* (12**0.5)
    return (mu - rf)/ (sd + 1e-12)

def max_drawdown(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum/peak - 1.0)
    return dd.min()

def summarize_performance(bt: pd.DataFrame) -> str:
    s = []
    for col in ["strategy","benchmark","excess"]:
        if col not in bt.columns: continue
        r = bt[col].dropna()
        if len(r) == 0: 
            continue
        cagr = ( (1+r).prod() ** (12/len(r)) ) - 1
        s.append(f"[{col}] CAGR: {cagr:.2%}, Sharpe: {sharpe(r):.2f}, MDD: {max_drawdown(r):.2%}, HitRate: {(r>0).mean():.1%}")
    return "\n".join(s)