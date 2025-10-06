"""
Macro Regime Identification using Markov-Switching Models
Identifies expansion, stagflation, and recession regimes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RegimeModel:
    def __init__(self, n_regimes=3):
        """
        Initialize regime model
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify (default: 3)
            3 regimes: Expansion, Stagflation, Recession
        """
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        self.regime_probs = None
        
    def prepare_data(self, data_path='data/raw/macro_data.csv'):
        """Load and prepare macro data for regime estimation"""
        print("Loading macro data...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Select key indicators for regime identification
        # We'll use GDP growth and inflation as primary indicators
        self.data = df[['gdp_growth', 'inflation', 'unemployment']].copy()
        
        # Drop missing values
        self.data = self.data.dropna()
        
        # Standardize for better convergence
        self.data_standardized = (self.data - self.data.mean()) / self.data.std()
        
        print(f"  ✓ Loaded data: {self.data.shape}")
        print(f"  ✓ Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def estimate_model(self, dependent_var='gdp_growth', exog_vars=['inflation', 'unemployment']):
        """
        Estimate Markov-switching regression model
        
        Parameters:
        -----------
        dependent_var : str
            Variable to model with regime-switching mean
        exog_vars : list
            Additional variables (non-switching)
        """
        print(f"\nEstimating {self.n_regimes}-regime Markov-switching model...")
        print(f"  Dependent variable: {dependent_var}")
        print(f"  Exogenous variables: {exog_vars}")
        
        # Prepare data
        y = self.data_standardized[dependent_var]
        X = self.data_standardized[exog_vars] if exog_vars else None
        
        # Estimate model
        # This estimates regime-switching intercepts with exogenous variables
        self.model = MarkovRegression(
            endog=y,
            k_regimes=self.n_regimes,
            exog=X,
            switching_variance=True  # Allow variance to differ by regime
        )
        
        print("\n  Fitting model (this may take a few minutes)...")
        self.results = self.model.fit(
            maxiter=1000,
            disp=False
        )
        
        print("  ✓ Model estimation complete!")
        
        # Extract regime probabilities
        self.regime_probs = pd.DataFrame(
            self.results.smoothed_marginal_probabilities,
            index=self.data.index,
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )
        
        # Identify regime characteristics
        self._identify_regimes()
        
        return self.results
    
    def _identify_regimes(self):
        """Identify what each regime represents based on parameters"""
        print("\nIdentifying regime characteristics...")
        
        # Get regime-specific parameters
        regime_means = self.results.params[0:self.n_regimes]
        
        # Sort regimes by mean (growth rate)
        regime_order = np.argsort(regime_means)
        
        # Label regimes
        self.regime_labels = {}
        if self.n_regimes == 3:
            self.regime_labels[regime_order[0]] = 'Recession'
            self.regime_labels[regime_order[1]] = 'Moderate Growth'
            self.regime_labels[regime_order[2]] = 'Expansion'
        elif self.n_regimes == 2:
            self.regime_labels[regime_order[0]] = 'Recession'
            self.regime_labels[regime_order[1]] = 'Expansion'
        
        # Add labeled probabilities
        for regime_idx, label in self.regime_labels.items():
            self.regime_probs[label] = self.regime_probs[f'Regime_{regime_idx}']
        
        # Print regime statistics
        print("\nRegime Characteristics:")
        print("-" * 60)
        for regime_idx, label in self.regime_labels.items():
            mask = self.regime_probs[f'Regime_{regime_idx}'] > 0.5
            if mask.sum() > 0:
                avg_gdp = self.data.loc[mask, 'gdp_growth'].mean()
                avg_inf = self.data.loc[mask, 'inflation'].mean()
                avg_unemp = self.data.loc[mask, 'unemployment'].mean()
                
                print(f"{label}:")
                print(f"  Average GDP growth: {avg_gdp:.2f}%")
                print(f"  Average inflation: {avg_inf:.2f}%")
                print(f"  Average unemployment: {avg_unemp:.2f}%")
                print(f"  Observations: {mask.sum()}")
                print()
    
    def print_summary(self):
        """Print model summary statistics"""
        if self.results is None:
            print("Model not estimated yet!")
            return
        
        print("\n" + "="*60)
        print("REGIME MODEL SUMMARY")
        print("="*60)
        print(self.results.summary())
        
        # Transition matrix
        print("\nRegime Transition Matrix:")
        print("-" * 60)
        trans_matrix = pd.DataFrame(
            self.results.regime_transition,
            index=[self.regime_labels.get(i, f'Regime_{i}') for i in range(self.n_regimes)],
            columns=[self.regime_labels.get(i, f'Regime_{i}') for i in range(self.n_regimes)]
        )
        print(trans_matrix.round(3))
        
        # Expected duration in each regime
        print("\nExpected Regime Duration (months):")
        print("-" * 60)
        for i in range(self.n_regimes):
            duration = 1 / (1 - self.results.regime_transition[i, i])
            label = self.regime_labels.get(i, f'Regime_{i}')
            print(f"{label}: {duration:.1f} months")
    
    def plot_regime_probabilities(self, save_path='output/figures/regime_probabilities.png'):
        """Plot regime probabilities over time"""
        fig, axes = plt.subplots(self.n_regimes + 1, 1, figsize=(14, 10))
        
        # Plot GDP growth
        axes[0].plot(self.data.index, self.data['gdp_growth'], color='black', linewidth=1.5)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('GDP Growth (%)')
        axes[0].set_title('Real GDP Growth and Regime Probabilities')
        axes[0].grid(True, alpha=0.3)
        
        # Plot regime probabilities
        colors = ['red', 'orange', 'green']
        for i in range(self.n_regimes):
            regime_label = self.regime_labels.get(i, f'Regime_{i}')
            axes[i+1].fill_between(
                self.regime_probs.index,
                0,
                self.regime_probs[f'Regime_{i}'],
                alpha=0.6,
                color=colors[i],
                label=regime_label
            )
            axes[i+1].set_ylabel('Probability')
            axes[i+1].set_ylim([0, 1])
            axes[i+1].legend(loc='upper right')
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved regime probability plot: {save_path}")
        
        return fig
    
    def plot_regime_scatter(self, save_path='output/figures/regime_scatter.png'):
        """Plot regimes in growth-inflation space"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine most likely regime for each observation
        most_likely_regime = self.regime_probs[[f'Regime_{i}' for i in range(self.n_regimes)]].idxmax(axis=1)
        most_likely_regime = most_likely_regime.map({f'Regime_{i}': i for i in range(self.n_regimes)})
        
        colors = ['red', 'orange', 'green']
        for i in range(self.n_regimes):
            mask = most_likely_regime == i
            label = self.regime_labels.get(i, f'Regime_{i}')
            
            ax.scatter(
                self.data.loc[mask, 'gdp_growth'],
                self.data.loc[mask, 'inflation'],
                c=colors[i],
                label=label,
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('GDP Growth (%)', fontsize=12)
        ax.set_ylabel('Inflation (%)', fontsize=12)
        ax.set_title('Macro Regimes: Growth vs Inflation', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved regime scatter plot: {save_path}")
        
        return fig
    
    def save_results(self, output_dir='data/processed'):
        """Save regime probabilities and model results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save regime probabilities
        self.regime_probs.to_csv(f'{output_dir}/regime_probabilities.csv')
        print(f"\n✓ Saved regime probabilities: {output_dir}/regime_probabilities.csv")
        
        # Save regime labels
        labels_df = pd.DataFrame(list(self.regime_labels.items()), columns=['Regime_Index', 'Label'])
        labels_df.to_csv(f'{output_dir}/regime_labels.csv', index=False)
        print(f"✓ Saved regime labels: {output_dir}/regime_labels.csv")
        
        return self.regime_probs

def main():
    """Main execution"""
    print("="*60)
    print("MACRO REGIME IDENTIFICATION")
    print("="*60)
    
    # Initialize and estimate model
    regime_model = RegimeModel(n_regimes=3)
    regime_model.prepare_data()
    regime_model.estimate_model()
    
    # Print results
    regime_model.print_summary()
    
    # Create visualizations
    regime_model.plot_regime_probabilities()
    regime_model.plot_regime_scatter()
    
    # Save results
    regime_model.save_results()
    
    print("\n" + "="*60)
    print("REGIME MODEL COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
