"""
Portfolio Optimization for Duration-Based Treasury Strategies
(cvxpy-free implementation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TARGET_DURATION = 6.0
DURATION_TOL = 2.0
MAX_WEIGHT = 0.5
TURNOVER_COST = 0.001

class PortfolioOptimizer:
    def __init__(self):
        self.universe = ["2Y", "5Y", "10Y", "30Y"]
        self.durations = {"2Y": 1.9, "5Y": 4.5, "10Y": 8.5, "30Y": 18.0}
        self.returns = None
        self.regime_probs = None
        self.weights_history = []
        self.backtest_results = None

    # ---------- Data ----------
    def load_data(self, returns_path="data/raw/treasury_returns.csv",
                  regime_path="data/processed/regime_probabilities.csv"):
        print("Loading data...")
        self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        self.returns = self.returns[self.universe].dropna()
        common = self.returns.index.intersection(self.regime_probs.index)
        self.returns, self.regime_probs = self.returns.loc[common], self.regime_probs.loc[common]
        print(f"  ✓ {len(self.returns)} monthly obs")
        return self.returns

    # ---------- Simple numeric optimizer ----------
    @staticmethod
    def _project_simplex(w, cap=0.5):
        """Project weights to [0,cap], sum=1"""
        w = np.clip(w, 0, cap)
        if w.sum() == 0:
            w[:] = 1/len(w)
        w /= w.sum()
        return w

    def _optimize(self, mu, cov, target_duration, prev_w=None, steps=500, lr=0.03):
        durs = np.array([self.durations[a] for a in mu.index])
        w = np.ones(len(mu)) / len(mu) if prev_w is None else prev_w.copy()

        for _ in range(steps):
            port_ret = mu.values @ w
            port_vol = np.sqrt(max(w @ cov.values @ w, 1e-10))
            grad = (mu.values * port_vol - port_ret * (cov.values @ w) / port_vol) / (port_vol**2 + 1e-10)

            # duration penalty
            dur = durs @ w
            if dur < target_duration - DURATION_TOL:
                grad += 0.5 * (target_duration - dur) * durs
            elif dur > target_duration + DURATION_TOL:
                grad -= 0.5 * (dur - target_duration) * durs

            # turnover penalty
            if prev_w is not None:
                grad -= TURNOVER_COST * np.sign(w - prev_w)

            w += lr * grad
            w = self._project_simplex(w, MAX_WEIGHT)
        return pd.Series(w, index=mu.index)

    # ---------- Regime handling ----------
    def calculate_regime_statistics(self):
        regime_cols = [c for c in self.regime_probs.columns if "Regime_" in c]
        most = self.regime_probs[regime_cols].idxmax(axis=1)
        self.regime_stats = {}
        for reg in regime_cols:
            mask = most == reg
            sub = self.returns.loc[mask]
            if len(sub) > 20:
                self.regime_stats[reg] = {"mean": sub.mean(), "cov": sub.cov()}
        return self.regime_stats

    def regime_conditional_allocation(self, probs, prev_w=None):
        regime_cols = [c for c in probs.index if "Regime_" in c]
        mean, cov = pd.Series(0, self.universe), pd.DataFrame(0, self.universe, self.universe)
        for r in regime_cols:
            if r in self.regime_stats:
                p = probs[r]
                mean += p * self.regime_stats[r]["mean"]
                cov += p * self.regime_stats[r]["cov"]

        recession = probs.get("Recession", 0)
        expansion = probs.get("Expansion", 0)
        target_dur = np.clip(TARGET_DURATION + 2 * (recession - expansion), 4, 10)

        w = self._optimize(mean, cov, target_dur, prev_w=prev_w)
        return w, target_dur

    # ---------- Backtest ----------
    def backtest(self, rebalance_freq=1):
        print("\nRunning backtest...")
        self.calculate_regime_statistics()
        weights, results, prev_w = [], [], None

        for i, date in enumerate(self.returns.index):
            if i % rebalance_freq == 0 or prev_w is None:
                probs = self.regime_probs.iloc[i]
                w, dur = self.regime_conditional_allocation(probs, prev_w)
                prev_w = w
                weights.append({"date": date, **w.to_dict(), "target_duration": dur})

            r = self.returns.iloc[i]
            ret = float((prev_w * r).sum())
            results.append({"date": date, "return": ret})

        self.weights_history = pd.DataFrame(weights).set_index("date")
        self.backtest_results = pd.DataFrame(results).set_index("date")
        self.backtest_results["cumret"] = (1 + self.backtest_results["return"]).cumprod()
        print("  ✓ Backtest complete")
        return self.backtest_results

    # ---------- Plot / Save ----------
    def plot_performance(self, path="output/figures/portfolio_perf.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.backtest_results.index, self.backtest_results["cumret"], label="Strategy", lw=2)
        plt.title("Regime-Conditional Duration Strategy")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved plot: {path}")

    def save_results(self, outdir="data/processed"):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        self.backtest_results.to_csv(f"{outdir}/backtest_results.csv")
        self.weights_history.to_csv(f"{outdir}/portfolio_weights.csv")
        print(f"✓ Results saved to {outdir}")

def main():
    print("="*60)
    print("PORTFOLIO OPTIMIZATION (NumPy version)")
    print("="*60)
    opt = PortfolioOptimizer()
    opt.load_data()
    opt.backtest()
    opt.plot_performance()
    opt.save_results()
    print("="*60)
    print("PORTFOLIO OPTIMIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()