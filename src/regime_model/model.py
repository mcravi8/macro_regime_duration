"""
Regime Identification Model
Implements Markov-switching regression for macro regime detection
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt


class RegimeModel:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.data = None
        self.data_standardized = None
        self.model = None
        self.results = None
        self.regime_probs = None
        self.regime_labels = {}

    # ============================================================
    # 1. DATA PREPARATION
    # ============================================================
    def prepare_data(self, data_path="data/raw/macro_data.csv"):
        """Load and standardize macroeconomic data"""
        print("Loading macro data...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df = df.dropna()

        # Keep core variables
        self.data = df.copy()

        # Standardize
        self.data_standardized = (df - df.mean()) / df.std()
        print(f"  ✓ Loaded data: {self.data.shape}")
        print(f"  ✓ Date range: {self.data.index.min()} to {self.data.index.max()}")

    # ============================================================
    # 2. MODEL ESTIMATION
    # ============================================================
    def estimate_model(self, dependent_var='gdp_growth', exog_vars=['inflation', 'unemployment']):
        """Estimate Markov-switching regression model"""
        print(f"\nEstimating {self.n_regimes}-regime Markov-switching model...")

        y = self.data_standardized[dependent_var]
        X = self.data_standardized[exog_vars] if exog_vars else None

        self.model = MarkovRegression(
            endog=y,
            k_regimes=self.n_regimes,
            exog=X,
            switching_variance=True
        )

        print("\n  Fitting model (this may take a few minutes)...")
        self.results = self.model.fit(maxiter=1000, disp=False)
        print("  ✓ Model estimation complete!")

        # --- Safe extraction of probabilities ---
        probs = np.asarray(self.results.smoothed_marginal_probabilities)

        n_obs = len(self.data)
        if probs.shape == (self.n_regimes, n_obs):
            probs = probs.T
        elif probs.shape == (n_obs, self.n_regimes):
            pass
        else:
            raise ValueError(f"Unexpected shape {probs.shape} for regime probabilities")

        self.regime_probs = pd.DataFrame(
            probs,
            index=self.data.index,
            columns=[f"Regime_{i}" for i in range(self.n_regimes)]
        )

        self._identify_regimes()

        # Add readable labels
        for i, label in self.regime_labels.items():
            self.regime_probs[label] = self.regime_probs[f"Regime_{i}"]

        print("\nHead of regime probabilities (in memory):")
        print(self.regime_probs.head())

        return self.results

    # ============================================================
    # 3. IDENTIFY & LABEL REGIMES
    # ============================================================
    def _identify_regimes(self):
        """Assign regime labels based on mean GDP growth"""
        regime_means = self.results.params[[f"const[{i}]" for i in range(self.n_regimes)]]
        sorted_idx = np.argsort(regime_means)
        self.regime_labels = {
            sorted_idx[0]: "Recession",
            sorted_idx[1]: "Moderate Growth",
            sorted_idx[2]: "Expansion"
        }

    # ============================================================
    # 4. PLOTS
    # ============================================================
    def plot_regime_probabilities(self, save_path="output/figures/regime_probabilities.png"):
        """Plot smoothed regime probabilities"""
        plt.figure(figsize=(12, 6))
        for i in range(self.n_regimes):
            plt.plot(
                self.regime_probs.index,
                self.regime_probs[f"Regime_{i}"],
                label=f"Regime {i} - {self.regime_labels[i]}"
            )
        plt.title("Smoothed Regime Probabilities")
        plt.xlabel("Date")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Saved regime probability plot: {save_path}")

    def plot_regime_scatter(self, save_path="output/figures/regime_scatter.png"):
        """Scatter plot of GDP growth colored by dominant regime"""
        dominant = self.regime_probs[[f"Regime_{i}" for i in range(self.n_regimes)]].idxmax(axis=1)
        colors = {
            "Regime_0": "red",
            "Regime_1": "orange",
            "Regime_2": "green"
        }

        plt.figure(figsize=(12, 6))
        plt.scatter(
            self.data.index,
            self.data['gdp_growth'],
            c=dominant.map(colors),
            alpha=0.6
        )
        plt.title("Dominant Regime Over Time")
        plt.xlabel("Date")
        plt.ylabel("GDP Growth (standardized)")
        plt.tight_layout()
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Saved regime scatter plot: {save_path}")

    # ============================================================
    # 5. SAVE RESULTS
    # ============================================================
    def save_results(self, output_dir="data/processed"):
        """Save regime probabilities and labels"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.regime_probs is None or self.regime_probs.isna().all().all():
            raise RuntimeError("Cannot save regime probabilities: all values are NaN.")

        to_save = self.regime_probs.copy()
        ordered_cols = [f"Regime_{i}" for i in range(self.n_regimes)]
        ordered_cols += [self.regime_labels[i] for i in range(self.n_regimes)]
        to_save = to_save[ordered_cols]

        out_path = os.path.join(output_dir, "regime_probabilities.csv")
        to_save.to_csv(out_path)
        print(f"✓ Saved regime probabilities: {out_path}")

        labels_df = pd.DataFrame(
            [(i, self.regime_labels[i]) for i in range(self.n_regimes)],
            columns=['Regime_Index', 'Label']
        )
        labels_df.to_csv(os.path.join(output_dir, "regime_labels.csv"), index=False)
        print(f"✓ Saved regime labels: {output_dir}/regime_labels.csv")

        # Quick sanity validation
        check = pd.read_csv(out_path, index_col=0)
        if check.isna().all().all():
            raise RuntimeError("Saved regime_probabilities.csv is all NaN — check model output.")

        return self.regime_probs

    # ============================================================
    # 6. SUMMARY
    # ============================================================
    def print_summary(self):
        """Print model summary and regime transition info"""
        print("\n" + "=" * 60)
        print("REGIME MODEL SUMMARY")
        print("=" * 60)
        print(self.results.summary())

        # Transition matrix
        print("\nRegime Transition Matrix:")
        print("-" * 60)
        print(np.round(self.results.regime_transition, 3))

        # Expected duration
        print("\nExpected Regime Duration (months):")
        print("-" * 60)
        for i in range(self.n_regimes):
            p_stay = float(self.results.regime_transition[i, i])  # convert to scalar
            duration = 1 / (1 - p_stay)
            print(f"{self.regime_labels[i]}: {duration:.1f} months")
