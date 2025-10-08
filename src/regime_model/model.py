"""
Regime Identification Model
Implements Markov-switching regression for macro regime detection
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class RegimeModel:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.data: pd.DataFrame | None = None
        self.data_standardized: pd.DataFrame | None = None
        self.model: MarkovRegression | None = None
        self.results = None
        self.regime_probs: pd.DataFrame | None = None
        # e.g. {0: "Recession", 1: "Moderate Growth", 2: "Expansion"}
        self.regime_labels: dict[int, str] = {}

    # ============================================================
    # 1) DATA PREPARATION
    # ============================================================
    def prepare_data(self, data_path: str = "data/raw/macro_data.csv"):
        """Load and z-score macro series needed by the regime model."""
        print("Loading macro data...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Drop any all-NaN rows
        df = df.dropna(how="all")
        # Keep simple, consistent set of columns if present
        expected = ["gdp_growth", "inflation", "unemployment"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in macro_data.csv: {missing}")

        df = df[expected].dropna()
        self.data = df
        self.data_standardized = (df - df.mean()) / df.std(ddof=0)

        print(f"  ✓ Loaded data: {self.data.shape}")
        print(f"  ✓ Date range: {self.data.index.min()} to {self.data.index.max()}")

    # ============================================================
    # 2) MODEL ESTIMATION
    # ============================================================
    def estimate_model(
        self,
        dependent_var: str = "gdp_growth",
        exog_vars: list[str] = ["inflation", "unemployment"],
    ):
        """Estimate a Markov-switching regression and build probability table."""
        if self.data_standardized is None:
            raise RuntimeError("Call prepare_data() first.")

        print(f"\nEstimating {self.n_regimes}-regime Markov-switching model...")

        y = self.data_standardized[dependent_var]
        X = self.data_standardized[exog_vars] if exog_vars else None

        self.model = MarkovRegression(
            endog=y,
            k_regimes=self.n_regimes,
            exog=X,
            switching_variance=True,
        )

        print("\n  Fitting model (this may take a few minutes)...")
        self.results = self.model.fit(maxiter=1000, disp=False)
        print("  ✓ Model estimation complete!")

        # --- Extract smoothed probabilities robustly ---
        probs = np.asarray(self.results.smoothed_marginal_probabilities)
        n_obs = len(y)

        # Shape can be (k, T) or (T, k) depending on statsmodels version
        if probs.shape == (self.n_regimes, n_obs):
            probs = probs.T
        elif probs.shape != (n_obs, self.n_regimes):
            raise ValueError(f"Unexpected smoothed probabilities shape: {probs.shape}")

        rp = pd.DataFrame(
            probs,
            index=self.data.index,
            columns=[f"Regime_{i}" for i in range(self.n_regimes)],
        )
        self.regime_probs = rp

        # Identify & attach readable labels
        self._identify_regimes()
        for i, label in self.regime_labels.items():
            self.regime_probs[label] = self.regime_probs[f"Regime_{i}"]

        print("\nHead of regime probabilities (in memory):")
        print(self.regime_probs.head())

        return self.results

    # ============================================================
    # 3) IDENTIFY & LABEL REGIMES
    # ============================================================
    def _identify_regimes(self):
        """Assign regime labels by ordering regime intercepts (const) on growth."""
        # Pull per-regime intercepts for dependent variable
        means = [self.results.params[f"const[{i}]"] for i in range(self.n_regimes)]
        order = np.argsort(means)  # low -> mid -> high growth

        # Map the lowest growth to Recession, mid to Moderate, highest to Expansion
        labels_in_order = ["Recession", "Moderate Growth", "Expansion"]
        self.regime_labels = dict(zip(order, labels_in_order))

    # ============================================================
    # 4) PLOTS
    # ============================================================
    def _clean_probabilities(self) -> pd.DataFrame | None:
        """Return a numeric, date-indexed copy of regime probabilities."""
        if self.regime_probs is None:
            return None

        probs_raw = self.regime_probs.copy()
        probs_raw.index = pd.to_datetime(probs_raw.index, errors="coerce")
        probs_raw = probs_raw.loc[probs_raw.index.notna()].sort_index()

        numeric_cols = {}
        for col in probs_raw.columns:
            numeric_cols[col] = pd.to_numeric(probs_raw[col], errors="coerce")

        probs = pd.DataFrame(numeric_cols, index=probs_raw.index)
        probs = probs.loc[:, probs.notna().any(axis=0)]
        return probs

    def plot_regime_probabilities(self, save_path: str = "output/figures/regime_probabilities.png"):
        """
        Line and area plot of the smoothed probabilities with visibility boosts for flat regimes.
        """
        probs = self._clean_probabilities()
        if probs is None or probs.empty:
            print("Regime probabilities are empty after cleaning/parsing.")
            return

        regime_cols = [c for c in probs.columns if c.startswith("Regime_")] or list(probs.columns)

        fig, ax = plt.subplots(figsize=(14, 5))
        global_bounds = []

        for col in regime_cols:
            series = probs[col].astype(float)
            print(f"{col} describe:\n{series.describe()}")

            display_series = series.copy()
            std = float(display_series.std(skipna=True))
            if not np.isnan(std) and std < 1e-3:
                denom = std if std > 1e-12 else 1e-12
                display_series = (display_series - display_series.mean()) / denom
                values = display_series.to_numpy(dtype=float)
                finite_vals = values[np.isfinite(values)]
                if finite_vals.size:
                    min_val, max_val = np.nanmin(finite_vals), np.nanmax(finite_vals)
                    rng = max_val - min_val
                    if rng > 0:
                        display_series = (display_series - min_val) / rng
                    else:
                        display_series = display_series - min_val + 0.5

            line, = ax.plot(display_series.index, display_series, label=col, lw=1.8)
            mask = display_series.to_numpy(dtype=float) > 0.001
            ax.fill_between(
                display_series.index,
                0,
                display_series,
                where=mask,
                color=line.get_color(),
                alpha=0.15,
            )

            values = display_series.to_numpy(dtype=float)
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size:
                global_bounds.extend([np.nanmin(finite_vals), np.nanmax(finite_vals)])

        if global_bounds:
            ymin, ymax = min(global_bounds) - 0.02, max(global_bounds) + 0.02
        else:
            ymin, ymax = 0.0, 1.0

        ax.set_ylim(max(0.0, ymin), min(1.05, ymax))
        ax.set_title("Regime Probabilities Over Time")
        ax.set_ylabel("Probability")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Saved regime probability plot: {save_path}")

    def plot_regime_scatter(self, save_path: str = "output/figures/regime_scatter.png"):
        """
        Scatter of standardized GDP growth colored by the dominant regime.
        Robust to index misalignment between macro data and regime probabilities.
        """
        if self.data_standardized is None:
            print("Standardized macro data unavailable; cannot plot scatter.")
            return

        probs = self._clean_probabilities()
        if probs is None or probs.empty:
            print("No regime probabilities available for scatter.")
            return

        gdp = self.data_standardized["gdp_growth"].copy()
        gdp.index = pd.to_datetime(gdp.index, errors="coerce")
        gdp = gdp.loc[gdp.index.notna()].sort_index()

        common_idx = probs.index.intersection(gdp.index)
        if common_idx.empty:
            print("No overlapping dates between regime probabilities and GDP growth for scatter.")
            return

        probs = probs.loc[common_idx].fillna(0.0)
        gdp = gdp.loc[common_idx]

        regime_cols = [c for c in probs.columns if c.startswith("Regime_")]
        if not regime_cols:
            regime_cols = list(probs.columns)

        nonzero_counts = (probs[regime_cols].abs() > 1e-8).sum()
        print("Scatter data counts (non-zero per regime column):")
        for col, count in nonzero_counts.items():
            print(f"  {col}: {int(count)}")

        dominant = probs[regime_cols].idxmax(axis=1)
        color_cycle = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
        color_map = {regime: color_cycle[i % len(color_cycle)] for i, regime in enumerate(regime_cols)}

        fig, ax = plt.subplots(figsize=(14, 5))
        for regime in regime_cols:
            mask = dominant == regime
            if mask.sum() == 0:
                continue
            ax.scatter(gdp.index[mask], gdp[mask], s=18, alpha=0.7, color=color_map[regime], label=regime)

        ax.set_title("Dominant Regime Over Time (Standardized GDP Growth)")
        ax.set_ylabel("Std. GDP Growth")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Saved regime scatter plot: {save_path}")

    def plot_regime_panels(self, save_path: str = "output/figures/regime_panels.png"):
        """
        Four-panel diagnostic: GDP growth plus each labeled regime probability with visibility boosts.
        """
        if self.data is None:
            print("Macro data unavailable; cannot draw regime panels.")
            return

        probs = self._clean_probabilities()
        if probs is None or probs.empty:
            print("No regime probabilities available for regime panels.")
            return

        data = self.data.copy()
        data.index = pd.to_datetime(data.index, errors="coerce")
        data = data.loc[data.index.notna()].sort_index()

        common_idx = probs.index.intersection(data.index)
        if common_idx.empty:
            print("No overlapping dates between regime probabilities and macro data; reindexing GDP.")
            data = data.reindex(probs.index).interpolate(method="time").ffill().bfill()
            aligned_idx = probs.index
            probs = probs.reindex(aligned_idx)
        else:
            aligned_idx = common_idx
            data = data.loc[aligned_idx]
            probs = probs.loc[aligned_idx]

        named_labels = ["Expansion", "Moderate Growth", "Recession"]
        panel_probs = pd.DataFrame(index=probs.index)

        for label in named_labels:
            if label in probs:
                panel_probs[label] = probs[label]
                continue
            for regime_idx, regime_label in self.regime_labels.items():
                if regime_label == label and f"Regime_{regime_idx}" in probs.columns:
                    panel_probs[label] = probs[f"Regime_{regime_idx}"]
                    break

        panel_probs = panel_probs.fillna(0.0)

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1.3, 1, 1, 1]},
        )

        axes[0].plot(aligned_idx, data["gdp_growth"], color="black", lw=1.5)
        axes[0].axhline(0, color="red", lw=1, ls="--", alpha=0.4)
        axes[0].set_title("Real GDP Growth and Regime Probabilities")
        axes[0].set_ylabel("GDP Growth (%)")
        axes[0].grid(True, alpha=0.3)

        colors = {"Expansion": "#2ca02c", "Moderate Growth": "#ff7f0e", "Recession": "#d62728"}
        for ax, label in zip(axes[1:], named_labels):
            if label not in panel_probs:
                ax.text(0.5, 0.5, f"{label} (missing)", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            series = panel_probs[label].astype(float)
            print(f"{label} describe:\n{series.describe()}")

            display_series = series.copy()
            std = float(display_series.std(skipna=True))
            if not np.isnan(std) and std < 1e-3:
                denom = std if std > 1e-12 else 1e-12
                display_series = (display_series - display_series.mean()) / denom
                values = display_series.to_numpy(dtype=float)
                finite_vals = values[np.isfinite(values)]
                if finite_vals.size:
                    min_val, max_val = np.nanmin(finite_vals), np.nanmax(finite_vals)
                    rng = max_val - min_val
                    if rng > 0:
                        display_series = (display_series - min_val) / rng
                    else:
                        display_series = display_series - min_val + 0.5

            ax.plot(aligned_idx, display_series, color=colors[label], lw=1.8, label=label)
            mask = display_series.to_numpy(dtype=float) > 0.001
            ax.fill_between(
                aligned_idx,
                0,
                display_series,
                where=mask,
                color=colors[label],
                alpha=0.2,
            )

            values = display_series.to_numpy(dtype=float)
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size:
                ymin, ymax = np.nanmin(finite_vals) - 0.02, np.nanmax(finite_vals) + 0.02
            else:
                ymin, ymax = 0.0, 1.0
            ax.set_ylim(max(0.0, ymin), min(1.05, ymax))
            ax.set_ylabel("Probability")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Saved 4-panel regime figure: {save_path}")

    # ============================================================
    # 5) SAVE RESULTS
    # ============================================================
    def save_results(self, output_dir: str = "data/processed"):
        """Save regime probabilities and labels; sanity-check the output."""
        if self.regime_probs is None:
            raise RuntimeError("No regime probabilities to save.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ensure both numeric and named columns are present and ordered
        to_save = self.regime_probs.copy()
        ordered_cols = [f"Regime_{i}" for i in range(self.n_regimes)]
        ordered_cols += [self.regime_labels[i] for i in range(self.n_regimes)]
        to_save = to_save[ordered_cols]

        probs_path = os.path.join(output_dir, "regime_probabilities.csv")
        to_save.to_csv(probs_path)
        print(f"✓ Saved regime probabilities: {probs_path}")

        labels_df = pd.DataFrame(
            [(i, self.regime_labels[i]) for i in range(self.n_regimes)],
            columns=["Regime_Index", "Label"],
        )
        labels_path = os.path.join(output_dir, "regime_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"✓ Saved regime labels: {labels_path}")

        # Sanity check: file should not be all NaN
        check = pd.read_csv(probs_path, index_col=0)
        if check.isna().all().all():
            raise RuntimeError("Saved regime_probabilities.csv is all NaN — check model output.")

        return self.regime_probs

    # ============================================================
    # 6) SUMMARY
    # ============================================================
    def print_summary(self):
        """Print model summary, transition matrix and expected durations."""
        if self.results is None:
            print("No fitted model.")
            return

        print("\n" + "=" * 60)
        print("REGIME MODEL SUMMARY")
        print("=" * 60)
        print(self.results.summary())

        # Transition matrix (statsmodels exposes .regime_transition)
        print("\nRegime Transition Matrix:")
        print("-" * 60)
        tm = np.array(self.results.regime_transition, dtype=float)
        print(np.round(tm, 3))

        # Expected duration by regime
        print("\nExpected Regime Duration (months):")
        print("-" * 60)
        for i in range(self.n_regimes):
            p_stay = float(tm[i, i])
            dur = 1.0 / max(1 - p_stay, 1e-12)
            print(f"{self.regime_labels[i]}: {dur:.1f} months")

        # Also save a brief text summary (helps confirm on disk)
        out = Path("output/results/model_summary.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(str(self.results.summary()))
            f.write("\n\nTransition matrix:\n")
            f.write(str(np.round(tm, 6)))
        print(f"✓ Saved regime summary: {out}")
