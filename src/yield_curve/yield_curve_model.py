"""
Yield Curve Modeling using Nelson–Siegel Factors and VAR
Part of the Macro Regime Duration Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
from numpy.linalg import LinAlgError

warnings.filterwarnings("ignore")


class YieldCurveModel:
    def __init__(self, yield_data=None, regime_probs=None):
        self.yield_data = yield_data
        self.regime_probs = regime_probs
        self.ns_factors = None
        self.var_model = None
        self.var_results = None
        self.var_data = None

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    def load_data(self, yields_path, regime_path):
        print("\n============================================================")
        print("LOADING INPUT DATA")
        print("============================================================")

        self.yield_data = pd.read_csv(yields_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)

        print(f"✓ Loaded yields: {self.yield_data.shape}")
        print(f"✓ Loaded regimes: {self.regime_probs.shape}")

    # ============================================================
    # STEP 2: EXTRACT NELSON–SIEGEL FACTORS
    # ============================================================
    def extract_ns_factors(self):
        print("\n============================================================")
        print("EXTRACTING NELSON–SIEGEL FACTORS")
        print("============================================================")

        yields = self.yield_data.dropna(how="any").copy()
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # in years
        tau = 1.5

        # Construct NS loadings
        def ns_loadings(mats, tau):
            x = mats / tau
            L1 = (1 - np.exp(-x)) / x
            L2 = L1 - np.exp(-x)
            return np.column_stack([np.ones(len(mats)), L1, L2])

        X = ns_loadings(maturities, tau)

        betas = []
        for date, row in yields.iterrows():
            y = row.values
            if np.any(np.isnan(y)):
                betas.append([np.nan, np.nan, np.nan])
                continue

            try:
                model = LinearRegression().fit(X, y)
                betas.append(model.coef_)
            except Exception:
                betas.append([np.nan, np.nan, np.nan])

        self.ns_factors = pd.DataFrame(
            betas, index=yields.index, columns=["Level", "Slope", "Curvature"]
        ).dropna()

        print(f"✓ Extracted factors for {len(self.ns_factors)} dates")

        # Save
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        self.ns_factors.to_csv("data/processed/ns_factors.csv")
        plt.figure(figsize=(10, 6))
        self.ns_factors.plot(ax=plt.gca(), title="Nelson–Siegel Factors")
        plt.tight_layout()
        plt.savefig("data/processed/ns_factors.png")
        plt.close()
        print("✓ Saved to data/processed/ns_factors.csv")
        print("✓ Saved factor plot: data/processed/ns_factors.png")

    # ============================================================
    # STEP 3: ESTIMATE VAR MODEL
    # ============================================================
    def estimate_var(self, lags=2):
        print("\n============================================================")
        print("ESTIMATING VAR MODEL")
        print("============================================================")

        if self.ns_factors is None or self.regime_probs is None:
            raise ValueError("NS factors and regime probabilities must be loaded first.")

        # Align to monthly end-of-month data
        f_factors = self.ns_factors.resample("M").last()
        f_regime = self.regime_probs.resample("M").last()

        common_idx = f_factors.index.intersection(f_regime.index)
        f_factors = f_factors.loc[common_idx]
        f_regime = f_regime.loc[common_idx]

        # Select top 3 regime columns (probabilities only)
        regime_cols = [c for c in f_regime.columns if "Regime" in c][:3]
        df = pd.concat([f_factors, f_regime[regime_cols]], axis=1).dropna()

        self.var_data = df
        print(
            f"  Factors span: {self.ns_factors.index.min().date()} -> {self.ns_factors.index.max().date()}"
        )
        print(
            f"  Regimes span: {self.regime_probs.index.min().date()} -> {self.regime_probs.index.max().date()}"
        )
        print(f"  Overlapping months: {len(common_idx)}")
        print(f"✓ VAR dataset shape: {df.shape}")

        if len(df) < 30:
            raise ValueError("Not enough observations for VAR estimation after alignment.")

        # Fit VAR
        self.var_model = VAR(df)
        self.var_results = self.var_model.fit(lags)
        print(f"✓ VAR({lags}) fitted successfully")

        # --- Stabilize covariance if needed
        cov = self.var_results.sigma_u_mle
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            jitter = abs(np.min(eigvals)) + 1e-6
            self.var_results.sigma_u_mle += np.eye(cov.shape[0]) * jitter
            print(f"⚙️ Added jitter {jitter:.2e} to stabilize covariance.")

        # --- Info criteria (AIC/BIC)
        try:
            aic = self.var_results.aic
            bic = self.var_results.bic
            print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
        except LinAlgError:
            print("⚠️ Covariance not positive-definite — skipping AIC/BIC display.")

        # --- Save summary safely
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        try:
            with open("data/processed/var_summary.txt", "w") as f:
                f.write(str(self.var_results.summary()))
            print("✓ VAR summary saved to data/processed/var_summary.txt")
        except Exception as e:
            print(f"⚠️ Full summary generation failed: {e}")
            # Fallback manual info
            with open("data/processed/var_summary.txt", "w") as f:
                f.write("VAR SUMMARY (manual fallback)\n")
                f.write("=" * 50 + "\n")
                f.write(f"Lags: {lags}\n")
                f.write(f"Observations: {len(df)}\n")
                f.write("Columns: " + ", ".join(df.columns) + "\n")
                f.write("\nResidual covariance matrix:\n")
                f.write(str(np.round(self.var_results.sigma_u_mle, 5)) + "\n")
            print("✓ Fallback VAR summary written instead.")

    # ============================================================
    # STEP 4: FORECAST + SAVE
    # ============================================================
    def forecast_factors(self, steps=6):
        print("\n============================================================")
        print("FORECASTING FACTORS")
        print("============================================================")
        if self.var_results is None:
            raise ValueError("VAR model must be fitted before forecasting.")
        forecast = self.var_results.forecast(self.var_data.values[-self.var_results.k_ar:], steps)
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.var_data.columns,
            index=pd.date_range(self.var_data.index[-1], periods=steps + 1, freq="M")[1:],
        )
        forecast_df.to_csv("data/processed/var_forecast.csv")
        print(f"✓ Saved {steps}-month VAR forecast to data/processed/var_forecast.csv")

    def calculate_expected_returns(self):
        print("\n============================================================")
        print("CALCULATING EXPECTED RETURNS (placeholder)")
        print("============================================================")
        # Future enhancement: link VAR forecasts to bond returns

    def plot_factors(self, save_path="data/processed/ns_factors.png"):
        plt.figure(figsize=(10, 6))
        self.ns_factors.plot(ax=plt.gca(), title="Nelson–Siegel Factors")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_results(self, output_dir="data/processed"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if self.ns_factors is not None:
            self.ns_factors.to_csv(f"{output_dir}/ns_factors.csv")
        if self.var_data is not None:
            self.var_data.to_csv(f"{output_dir}/var_dataset.csv")
        print(f"✓ Results saved to {output_dir}")
