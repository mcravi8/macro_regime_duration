"""
Portfolio Optimization for Duration-Based Treasury Strategies
(NumPy implementation; no cvxpy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Defaults (also mirrored in config.py)
TARGET_DURATION = 6.0
DURATION_TOL = 2.0
MAX_WEIGHT = 0.5
TURNOVER_COST = 0.001


class PortfolioOptimizer:
    def __init__(self):
        # Investable universe and durations
        self.universe = ["2Y", "5Y", "10Y", "30Y"]
        self.durations = {"2Y": 1.9, "5Y": 4.5, "10Y": 8.5, "30Y": 18.0}

        # Data containers
        self.returns = None                 # monthly returns for universe
        self.regime_probs = None            # historical regime probabilities
        self.forecast_next_probs = None     # optional next-step regime probs from VAR
        self.regime_label_map = None        # optional map: Regime_Index -> Label

        # Results
        self.weights_history = pd.DataFrame()
        self.backtest_results = pd.DataFrame()
        self.regime_stats = {}

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    def load_data(
        self,
        returns_path: str = "data/raw/treasury_returns.csv",
        regime_path: str = "data/processed/regime_probabilities.csv",
        forecast_path: str | None = None,
        label_map_path: str | None = None,
        regime_labels_path: str | None = None,  # backward-compat name
    ):
        """
        Load returns, regime probabilities, and (optionally) a VAR-based regime
        forecast CSV and a regime label map (produced by the regime model).

        Parameters
        ----------
        returns_path : str
            Path to monthly treasury returns CSV.
        regime_path : str
            Path to saved regime probabilities CSV.
        forecast_path : str | None
            Optional path to VAR-based forecast CSV (with columns including Regime_0/1/2).
        label_map_path : str | None
            Optional path to regime label map CSV with columns ['Regime_Index','Label'].
        regime_labels_path : str | None
            Backward-compatible alias for label_map_path.
        """
        # Backward-compat: if only regime_labels_path is provided, use it
        if regime_labels_path and not label_map_path:
            label_map_path = regime_labels_path

        print("Loading data...")

        # Core inputs
        self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)

        # Keep only the investable universe and drop missing rows
        self.returns = self.returns[self.universe].dropna()

        # Align dates with regime probabilities
        common = self.returns.index.intersection(self.regime_probs.index)
        self.returns = self.returns.loc[common]
        self.regime_probs = self.regime_probs.loc[common]
        print(f"  ✓ {len(self.returns)} monthly obs")

        # Optional: load VAR next-step regime forecast (if file exists & has Regime_* cols)
        self.forecast_next_probs = None
        if forecast_path:
            try:
                fc = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
                regime_cols = [c for c in fc.columns if c.startswith("Regime_")]
                if len(regime_cols) >= 2:
                    # Use the last available row as "next-step" probabilities
                    self.forecast_next_probs = fc.iloc[-1][regime_cols].astype(float)
                    print(f"  ✓ Loaded VAR forecast from {forecast_path} (as next-step probs)")
                else:
                    print(f"  ⚠️  No Regime_* columns in {forecast_path}; skipping forecast usage.")
            except Exception as e:
                print(f"  ⚠️  Could not load forecast file {forecast_path}: {e}")

        # Optional: load regime label map (Regime_Index -> Label) to help duration tilts
        self.regime_label_map = {}
        if label_map_path:
            try:
                lm = pd.read_csv(label_map_path)
                if {"Regime_Index", "Label"}.issubset(lm.columns):
                    self.regime_label_map = dict(zip(lm["Regime_Index"], lm["Label"]))
                    print("  ✓ Loaded regime label map for duration tilts.")
            except Exception as e:
                print(f"  ⚠️  Could not load label map {label_map_path}: {e}")

        return self.returns

    # ---------------------------------------------------------------------
    # Helper: projection & optimizer
    # ---------------------------------------------------------------------
    @staticmethod
    def _project_simplex(w: np.ndarray, cap: float = MAX_WEIGHT) -> np.ndarray:
        """Project weights into [0, cap] with sum=1."""
        w = np.clip(w, 0.0, cap)
        s = w.sum()
        if s <= 0:
            w[:] = 1.0 / len(w)
        else:
            w /= s
        return w

    def _optimize(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        target_duration: float,
        prev_w: np.ndarray | None = None,
        steps: int = 500,
        lr: float = 0.03,
    ) -> pd.Series:
        """
        Simple gradient ascent on Sharpe-like objective with duration & turnover penalties.
        """
        assert list(mu.index) == list(cov.index) == list(cov.columns), "mu/cov assets mismatch"

        durs = np.array([self.durations[a] for a in mu.index])
        w = np.ones(len(mu)) / len(mu) if prev_w is None else prev_w.copy()

        # Numerical safety
        covv = cov.values
        covv = covv + 1e-8 * np.eye(covv.shape[0])  # tiny ridge

        for _ in range(steps):
            port_ret = float(mu.values @ w)
            port_var = float(w @ covv @ w)
            port_vol = np.sqrt(max(port_var, 1e-12))

            # gradient of (port_ret / port_vol) wrt w (approximate)
            grad = (mu.values * port_vol - port_ret * (covv @ w) / port_vol) / (port_vol**2 + 1e-12)

            # duration penalty (push toward target within tolerance)
            dur = float(durs @ w)
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

    # ---------------------------------------------------------------------
    # Regime statistics & conditional allocation
    # ---------------------------------------------------------------------
    def calculate_regime_statistics(self):
        """
        Compute per-regime mean and covariance of returns using the historical
        most-likely regime (argmax over Regime_* columns).
        """
        regime_cols = [c for c in self.regime_probs.columns if c.startswith("Regime_")]
        if not regime_cols:
            raise ValueError("No Regime_* columns found in regime_probabilities.")

        most = self.regime_probs[regime_cols].idxmax(axis=1)
        self.regime_stats = {}
        for reg in regime_cols:
            mask = most == reg
            sub = self.returns.loc[mask, self.universe]
            if len(sub) > 20:  # ensure some minimum history per regime
                self.regime_stats[reg] = {"mean": sub.mean(), "cov": sub.cov()}
        return self.regime_stats

    def _label_probability(self, probs: pd.Series, label: str) -> float:
        """
        Map labeled probability (e.g., 'Recession') to the correct Regime_i column
        using self.regime_label_map if available; else fall back to direct key lookup.
        """
        # If the probs already contain the label as a column, use it
        if label in probs.index:
            return float(probs[label])

        # Otherwise, map label -> Regime_i
        if self.regime_label_map:
            # Find which regime index corresponds to the label
            for k, v in self.regime_label_map.items():
                if v == label:
                    col = f"Regime_{k}"
                    if col in probs.index:
                        return float(probs[col])
        return 0.0

    def regime_conditional_allocation(self, probs: pd.Series, prev_w: np.ndarray | None = None):
        """
        Mix per-regime mean/cov using current probabilities; tilt duration
        using Expansion/Recession labels if available.
        """
        regime_cols = [c for c in probs.index if c.startswith("Regime_")]
        mean = pd.Series(0.0, index=self.universe)
        cov = pd.DataFrame(0.0, index=self.universe, columns=self.universe)

        for r in regime_cols:
            if r in self.regime_stats:
                p = float(probs[r])
                mean += p * self.regime_stats[r]["mean"]
                cov += p * self.regime_stats[r]["cov"]

        # Duration tilt by labeled probabilities (if available)
        recession_p = self._label_probability(probs, "Recession")
        expansion_p = self._label_probability(probs, "Expansion")
        target_dur = float(np.clip(TARGET_DURATION + 2.0 * (recession_p - expansion_p), 4.0, 10.0))

        w = self._optimize(mean, cov, target_dur, prev_w=prev_w)
        return w.values, target_dur

    # ---------------------------------------------------------------------
    # Backtest
    # ---------------------------------------------------------------------
    def backtest(self, rebalance_freq: int = 1):
        """
        Simple monthly rebalanced backtest. Uses historical regime probabilities.
        If a VAR forecast was loaded, also writes the *next* allocation suggested
        by the last observed weights and the forecast probabilities.
        """
        print("\nRunning backtest...")
        if self.returns is None or self.regime_probs is None:
            raise ValueError("Call load_data() first.")

        self.calculate_regime_statistics()

        weights_records = []
        results = []
        prev_w = None

        for i, date in enumerate(self.returns.index):
            if i % rebalance_freq == 0 or prev_w is None:
                probs = self.regime_probs.iloc[i]
                w, dur = self.regime_conditional_allocation(probs, prev_w)
                prev_w = w
                rec = {"date": date, "target_duration": dur}
                rec.update({a: wj for a, wj in zip(self.universe, w)})
                weights_records.append(rec)

            r = self.returns.iloc[i].values
            ret = float(np.dot(prev_w, r))
            results.append({"date": date, "return": ret})

        self.weights_history = pd.DataFrame(weights_records).set_index("date")
        self.backtest_results = pd.DataFrame(results).set_index("date")
        self.backtest_results["cumret"] = (1.0 + self.backtest_results["return"]).cumprod()
        print("  ✓ Backtest complete")

        # Optional: compute & save the NEXT allocation using forecast probs
        if self.forecast_next_probs is not None:
            try:
                last_w = self.weights_history.iloc[-1][self.universe].values
                next_w, next_dur = self.regime_conditional_allocation(self.forecast_next_probs, last_w)
                next_alloc = pd.Series(next_w, index=self.universe)
                out = pd.DataFrame([{"target_duration": next_dur, **next_alloc.to_dict()}])
                Path("data/processed").mkdir(parents=True, exist_ok=True)
                out.to_csv("data/processed/next_allocation.csv", index=False)
                print("✓ Saved VAR-based next allocation to data/processed/next_allocation.csv")
            except Exception as e:
                print(f"⚠️ Could not compute next allocation from forecast: {e}")

        return self.backtest_results

    # ---------------------------------------------------------------------
    # Performance tables
    # ---------------------------------------------------------------------
    @staticmethod
    def _annualize_mean(series: pd.Series) -> float:
        return float(series.mean() * 12.0)

    @staticmethod
    def _annualize_vol(series: pd.Series) -> float:
        return float(series.std(ddof=0) * np.sqrt(12.0))

    @staticmethod
    def _sharpe(series: pd.Series) -> float:
        vol = series.std(ddof=0)
        if vol <= 0 or np.isnan(vol):
            return np.nan
        return float(np.sqrt(12.0) * series.mean() / vol)

    @staticmethod
    def _max_drawdown_from_returns(rets: pd.Series) -> float:
        curve = (1.0 + rets).cumprod()
        dd = curve / curve.cummax() - 1.0
        return float(dd.min())

    def compute_and_save_performance(
        self,
        returns_outdir: str = "output/returns",
        tables_outdir: str = "output/tables",
    ):
        """Compute overall & by-regime performance; save CSVs for README."""
        if self.backtest_results.empty:
            print("No backtest results — run backtest() first.")
            return

        Path(returns_outdir).mkdir(parents=True, exist_ok=True)
        Path(tables_outdir).mkdir(parents=True, exist_ok=True)

        # Dynamic (strategy) returns
        dyn = self.backtest_results["return"].copy()
        dyn.name = "return"
        dyn_cum = (1.0 + dyn).cumprod()
        pd.DataFrame({"return": dyn, "cumret": dyn_cum}).to_csv(
            f"{returns_outdir}/dynamic_portfolio.csv", index_label="date"
        )

        # Simple benchmark: equal-weight across universe
        eq_w = np.ones(len(self.universe)) / len(self.universe)
        bench = (self.returns[self.universe] @ eq_w).reindex(dyn.index).fillna(0.0)
        bench.name = "return"
        bench_cum = (1.0 + bench).cumprod()
        pd.DataFrame({"return": bench, "cumret": bench_cum}).to_csv(
            f"{returns_outdir}/benchmark_portfolio.csv", index_label="date"
        )

        # Overall stats
        summary = pd.DataFrame(
            [
                {
                    "Series": "Strategy",
                    "Mean Return (ann)": self._annualize_mean(dyn),
                    "Volatility (ann)": self._annualize_vol(dyn),
                    "Sharpe": self._sharpe(dyn),
                    "Max Drawdown": self._max_drawdown_from_returns(dyn),
                },
                {
                    "Series": "Equal-Weight Benchmark",
                    "Mean Return (ann)": self._annualize_mean(bench),
                    "Volatility (ann)": self._annualize_vol(bench),
                    "Sharpe": self._sharpe(bench),
                    "Max Drawdown": self._max_drawdown_from_returns(bench),
                },
            ]
        )
        summary["Sharpe Improvement"] = summary.loc[0, "Sharpe"] - summary.loc[1, "Sharpe"]
        summary.to_csv(f"{tables_outdir}/portfolio_performance.csv", index=False)
        print(f"✓ Saved overall performance → {tables_outdir}/portfolio_performance.csv")
        print(summary.round(3).to_string(index=False))

        # By-regime stats (using dominant regime)
        regime_cols = [c for c in self.regime_probs.columns if c.startswith("Regime_")]
        if regime_cols:
            dominant = self.regime_probs[regime_cols].idxmax(axis=1).reindex(dyn.index)
            # Map to readable labels if available
            if self.regime_label_map:
                name_map = {f"Regime_{k}": v for k, v in self.regime_label_map.items()}
                dominant = dominant.map(name_map).fillna(dominant)
        else:
            # Fallback if labeled columns exist
            lab_cols = [c for c in self.regime_probs.columns if c in ("Recession", "Moderate Growth", "Expansion")]
            if lab_cols:
                dominant = self.regime_probs[lab_cols].idxmax(axis=1).reindex(dyn.index)
            else:
                dominant = pd.Series(index=dyn.index, data="Unknown")

        rows = []
        for label, grp in dyn.groupby(dominant):
            if grp.empty:
                continue
            rows.append(
                {
                    "Regime": str(label),
                    "Obs": int(len(grp)),
                    "Mean Return (ann)": self._annualize_mean(grp),
                    "Volatility (ann)": self._annualize_vol(grp),
                    "Sharpe": self._sharpe(grp),
                    "Max Drawdown": self._max_drawdown_from_returns(grp),
                }
            )
        by_regime = pd.DataFrame(rows).sort_values("Regime")
        by_regime.to_csv(f"{tables_outdir}/performance_by_regime.csv", index=False)
        print(f"✓ Saved by-regime performance → {tables_outdir}/performance_by_regime.csv")

    # ---------------------------------------------------------------------
    # Plot & Save
    # ---------------------------------------------------------------------
    def plot_performance(self, path: str = "output/figures/portfolio_perf.png"):
        if self.backtest_results.empty:
            print("No backtest results to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.backtest_results.index,
            self.backtest_results["cumret"],
            label="Strategy",
            lw=2,
        )
        plt.title("Regime-Conditional Duration Strategy")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved plot: {path}")

    def save_results(self, outdir: str = "data/processed"):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if not self.backtest_results.empty:
            self.backtest_results.to_csv(f"{outdir}/backtest_results.csv")
        if not self.weights_history.empty:
            self.weights_history.to_csv(f"{outdir}/portfolio_weights.csv")
        print(f"✓ Results saved to {outdir}")


def main():
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION (NumPy version)")
    print("=" * 60)
    opt = PortfolioOptimizer()
    opt.load_data()
    opt.backtest()
    opt.plot_performance()
    opt.compute_and_save_performance()  # NEW: save tables/returns for README
    opt.save_results()
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
