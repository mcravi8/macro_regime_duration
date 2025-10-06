import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_FIGS

def _savefig(name, save_path=None):
    if save_path is None:
        save_path = OUTPUT_FIGS / name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_regime_probs(probs: pd.DataFrame, save_path=None):
    plt.figure()
    probs.plot()
    plt.title("Regime Smoothed Probabilities")
    plt.xlabel("Date"); plt.ylabel("Probability")
    _savefig("regime_probabilities.png", save_path)

def plot_ns_factors(ns: pd.DataFrame, save_path=None):
    plt.figure()
    ns.plot()
    plt.title("Nelson–Siegel Factors")
    plt.xlabel("Date"); plt.ylabel("Value")
    _savefig("ns_factors.png", save_path)

def plot_performance(bt: pd.DataFrame, save_path=None):
    plt.figure()
    eq = (1 + bt.fillna(0)).cumprod()
    eq.plot()
    plt.title("Strategy vs Benchmark (Cumulative)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    _savefig("portfolio_performance.png", save_path)