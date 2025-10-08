"""
Yield Curve Modeling
- Extracts Nelson–Siegel factors (Level, Slope, Curvature)
- Fits VAR model on factors + regime probabilities
- Forecasts next N months of yield curve evolution
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")


class YieldCurveModel:
    """
    Orchestrates:
      1) Nelson–Siegel factor extraction from Treasury yields
      2) VAR(l) on [Level, Slope, Curvature] + regime probabilities
      3) Multi-step forecast
    """

    def __init__(self, yield_data: pd.DataFrame, regime_probs: pd.DataFrame):
        # Expect yield_data with columns that include: '2Y','5Y','10Y','30Y'
        # Expect regime_probs with columns 'Regime_0', 'Regime_1', ...
        self.yield_data = yield_data.copy()
        self.regime_probs = regime_probs.copy()

        # Artifacts
        self.ns_factors: pd.DataFrame | None = None
        self.var_results = None
        self.var_forecast: pd.DataFrame | None = None

    # -------------------------------------------------------------------------
    # 1) Nelson–Siegel extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def _nelson_siegel(maturity, level, slope, curvature, tau=0.0609):
        """
        Nelson–Siegel functional form. 'maturity' is in years.
        """
        m = np.asarray(maturity, dtype=float)
        term = (1.0 - np.exp(-m * tau)) / (m * tau)
        return level + slope * (term - 1.0) + curvature * ((term - 1.0) * (m * tau))

    def extract_ns_factors(self, maturities: list[int] | None = None, tau: float = 0.0609):
        """
        Fit the NS curve per date on selected maturities (default: 2,5,10,30Y).
        Saves:
          - data/processed/ns_factors.csv
          - output/figures/ns_factors.png
        """
        print("\n" + "=" * 60)
        print("EXTRACTING NELSON–SIEGEL FACTORS")
        print("=" * 60)

        if maturities is None:
            maturities = [2, 5, 10, 30]

        cols_needed = [f"{m}Y" for m in maturities]
        missing = [c for c in cols_needed if c not in self.yield_data.columns]
        if missing:
            raise ValueError(
                f"Yield data is missing required columns: {missing}. "
                f"Available: {list(self.yield_data.columns)}"
            )

        yld = (
            self.yield_data
            .copy()
            .sort_index()
            .loc[:, cols_needed]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="any")
        )

        maturities_arr = np.array(maturities, dtype=float)
        rows = []

        # Reasonable parameter bounds (in percent space)
        bounds = ([-10.0, -15.0, -15.0], [15.0, 15.0, 15.0])

        for dt, row in yld.iterrows():
            yobs = row.values.astype(float)
            # Initial guess: level ~ avg, slope negative, curvature small
            p0 = [np.nanmean(yobs), -2.0, 1.0]
            try:
                popt, _ = curve_fit(
                    lambda m, L, S, C: self._nelson_siegel(m, L, S, C, tau=tau),
                    maturities_arr,
                    yobs,
                    p0=p0,
                    bounds=bounds,
                    maxfev=20000,
                )
                rows.append((dt, *popt))
            except Exception:
                # Skip date if fit fails
                continue

        if not rows:
            raise RuntimeError("Nelson–Siegel fitting failed for all dates.")

        ns = pd.DataFrame(rows, columns=["Date", "Level", "Slope", "Curvature"]).set_index("Date").sort_index()
        self.ns_factors = ns

        # Save CSV
        out_csv = Path("data/processed/ns_factors.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        ns.to_csv(out_csv)

        # Plot -> output (so it’s easy to view from VSCode / browser)
        fig_path = Path("output/figures/ns_factors.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(14, 5))
        for col in ["Level", "Slope", "Curvature"]:
            plt.plot(ns.index, ns[col], label=col)
        plt.title("Nelson–Siegel Yield Curve Factors")
        plt.xlabel("Date")
        plt.ylabel("Factor Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Extracted factors for {len(ns):,} dates")
        print(f"✓ Saved to {out_csv}")
        print(f"✓ Saved factor plot: {fig_path}")

        return ns

    # -------------------------------------------------------------------------
    # 2) VAR estimation
    # -------------------------------------------------------------------------
    def estimate_var(self, lags: int = 2):
        """
        Fit VAR(lags) on [Level, Slope, Curvature] + monthly regime probabilities.
        Writes a summary to data/processed/var_summary.txt (fallback if needed).
        """
        print("\n" + "=" * 60)
        print("ESTIMATING VAR MODEL")
        print("=" * 60)

        if self.ns_factors is None or self.ns_factors.empty:
            raise RuntimeError("Call extract_ns_factors() before estimate_var().")

        # Monthly averaging and strict alignment
        fac_m = self.ns_factors.asfreq("D").resample("M").mean()
        reg_m = (
            self.regime_probs
            .copy()
            .sort_index()
            .asfreq("D")
            .resample("M")
            .mean()
        )

        # Only keep Regime_* columns
        regime_cols = [c for c in reg_m.columns if c.startswith("Regime_")]
        reg_m = reg_m[regime_cols]

        start = max(fac_m.index.min(), reg_m.index.min())
        end = min(fac_m.index.max(), reg_m.index.max())

        df = fac_m.loc[start:end].join(reg_m.loc[start:end], how="inner").dropna(how="any")
        print(f"  Factors span: {fac_m.index.min().date()} -> {fac_m.index.max().date()}")
        print(f"  Regimes span: {reg_m.index.min().date()} -> {reg_m.index.max().date()}")
        print(f"  Overlapping months (before lags): {len(df)}")

        if len(df) <= lags + 5:
            raise ValueError("Not enough observations for VAR estimation after alignment.")

        print(f"✓ VAR dataset shape: {df.shape}")

        try:
            model = VAR(df)
            res = model.fit(maxlags=lags, ic=None)
            self.var_results = res
            print(f"✓ VAR({res.k_ar}) fitted successfully")

            # Try to Cholesky the covariance; if it fails, log & continue.
            try:
                _ = np.linalg.cholesky(np.asarray(res.sigma_u))
            except np.linalg.LinAlgError:
                print("⚠️ Residual covariance not positive-definite; info criteria may fail.")

            # Save a summary (with robust fallback)
            out_txt = Path("data/processed/var_summary.txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(out_txt, "w") as f:
                    f.write(str(res.summary()))
                print(f"✓ Saved VAR summary: {out_txt}")
            except Exception as e:
                with open(out_txt, "w") as f:
                    f.write("Fallback VAR summary\n")
                    f.write(f"Variables: {list(df.columns)}\n")
                    f.write(f"Observations: {len(df)}\n")
                    f.write(f"Lags: {res.k_ar}\n")
                print(f"⚠️ Full summary generation failed ({e}); wrote fallback instead.")

        except Exception as e:
            print(f"✗ VAR model fitting failed: {e}")
            raise

        return self.var_results

    # -------------------------------------------------------------------------
    # 3) Forecast next N months
    # -------------------------------------------------------------------------
    def forecast_next(self, steps: int = 6):
        """
        Forecast next N months for all variables in the VAR.
        Writes data/processed/var_forecast.csv and returns the DataFrame.
        """
        print("\n" + "=" * 60)
        print("FORECASTING FACTORS")
        print("=" * 60)

        if self.var_results is None:
            raise RuntimeError("Call estimate_var() before forecast_next().")

        res = self.var_results
        # Determine the correct attribute for endog history
        if hasattr(res, "endog"):
            yhist = np.asarray(res.endog)
        elif hasattr(res, "y"):
            yhist = np.asarray(res.y)
        else:
            raise AttributeError("VARResults has neither .endog nor .y")

        k = res.k_ar
        last = yhist[-k:]

        fc = res.forecast(y=last, steps=steps)
        cols = res.names
        # Forecast dates start one month after last available factor date
        start_date = (self.ns_factors.index[-1] + pd.offsets.MonthEnd(1)).normalize()
        idx = pd.date_range(start=start_date, periods=steps, freq="M")

        out = pd.DataFrame(fc, columns=cols, index=idx)
        out_csv = Path("data/processed/var_forecast.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv)
        self.var_forecast = out

        print(f"✓ Saved {steps}-month VAR forecast to {out_csv}")
        return out

    # -------------------------------------------------------------------------
    # (Optional) Re-plot factors later without re-fitting
    # -------------------------------------------------------------------------
    def plot_factors(self, save_path: str = "output/figures/ns_factors.png"):
        """Re-plot factors if they’re already computed."""
        if self.ns_factors is None or self.ns_factors.empty:
            raise RuntimeError("No factors to plot. Run extract_ns_factors() first.")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(14, 5))
        for col in ["Level", "Slope", "Curvature"]:
            plt.plot(self.ns_factors.index, self.ns_factors[col], label=col)
        plt.title("Nelson–Siegel Yield Curve Factors")
        plt.xlabel("Date")
        plt.ylabel("Factor Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved NS factors plot: {save_path}")
