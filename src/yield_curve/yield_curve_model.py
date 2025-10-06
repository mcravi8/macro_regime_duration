"""
Yield Curve Modeling using Nelson-Siegel and VAR Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class YieldCurveModel:
    def __init__(self):
        """Initialize yield curve model"""
        self.yields = None
        self.factors = None
        self.var_model = None
        self.forecasts = None
        
        # Standard maturities in years for Nelson-Siegel
        self.maturities_map = {
            '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0,
            '3Y': 3.0, '5Y': 5.0, '7Y': 7.0, '10Y': 10.0,
            '20Y': 20.0, '30Y': 30.0
        }
        
    def load_data(self, yields_path='data/raw/treasury_yields.csv', 
                  regime_path='data/processed/regime_probabilities.csv'):
        """Load yield curve and regime data"""
        print("Loading yield curve data...")
        
        self.yields = pd.read_csv(yields_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        
        # Align dates
        common_idx = self.yields.index.intersection(self.regime_probs.index)
        self.yields = self.yields.loc[common_idx]
        self.regime_probs = self.regime_probs.loc[common_idx]
        
        print(f"  ✓ Loaded {len(self.yields)} observations")
        print(f"  ✓ Maturities: {list(self.yields.columns)}")
        
        return self.yields
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda_param=0.0609):
        """
        Nelson-Siegel yield curve model
        
        y(tau) = beta0 + beta1*((1-exp(-lambda*tau))/(lambda*tau)) 
                      + beta2*((1-exp(-lambda*tau))/(lambda*tau) - exp(-lambda*tau))
        
        Parameters:
        -----------
        tau : array
            Maturities in years
        beta0 : float
            Level factor (long-term rate)
        beta1 : float
            Slope factor (short-term component)
        beta2 : float
            Curvature factor (medium-term component)
        lambda_param : float
            Decay parameter (typically around 0.0609 for monthly data)
        """
        tau = np.array(tau)
        exp_term = np.exp(-lambda_param * tau)
        
        # Avoid division by zero
        factor1 = np.where(tau > 0, (1 - exp_term) / (lambda_param * tau), 1.0)
        factor2 = factor1 - exp_term
        
        return beta0 + beta1 * factor1 + beta2 * factor2
    
    def fit_nelson_siegel(self, yields_row, maturities):
        """
        Fit Nelson-Siegel model to a single yield curve observation
        
        Returns:
        --------
        dict with beta0, beta1, beta2, lambda_param
        """
        # Initial guesses
        beta0_init = yields_row[-1]  # Long-term yield
        beta1_init = yields_row[0] - yields_row[-1]  # Short-long spread
        beta2_init = 2 * yields_row[len(yields_row)//2] - yields_row[0] - yields_row[-1]
        lambda_init = 0.0609
        
        # Objective function: minimize squared errors
        def objective(params):
            beta0, beta1, beta2, lambda_param = params
            fitted = self.nelson_siegel(maturities, beta0, beta1, beta2, lambda_param)
            return np.sum((yields_row - fitted)**2)
        
        # Optimize
        result = minimize(
            objective,
            x0=[beta0_init, beta1_init, beta2_init, lambda_init],
            method='L-BFGS-B',
            bounds=[
                (None, None),  # beta0
                (None, None),  # beta1
                (None, None),  # beta2
                (0.001, 1.0)   # lambda (must be positive)
            ]
        )
        
        if result.success:
            return {
                'beta0': result.x[0],
                'beta1': result.x[1],
                'beta2': result.x[2],
                'lambda': result.x[3],
                'fit_error': result.fun
            }
        else:
            return None
    
    def extract_factors(self):
        """Extract Nelson-Siegel factors for all dates"""
        print("\nExtracting Nelson-Siegel factors...")
        
        # Get maturities in years
        maturities = np.array([self.maturities_map[col] for col in self.yields.columns 
                               if col in self.maturities_map])
        
        # Fit NS model for each date
        factors_list = []
        for date, row in self.yields.iterrows():
            # Get yields that have maturity info
            yields_arr = np.array([row[col] for col in self.yields.columns 
                                  if col in self.maturities_map])
            
            # Remove NaN values
            valid_idx = ~np.isnan(yields_arr)
            if valid_idx.sum() < 4:  # Need at least 4 points
                factors_list.append({'date': date, 'beta0': np.nan, 'beta1': np.nan, 
                                   'beta2': np.nan, 'lambda': np.nan})
                continue
            
            fitted = self.fit_nelson_siegel(
                yields_arr[valid_idx],
                maturities[valid_idx]
            )
            
            if fitted:
                factors_list.append({
                    'date': date,
                    'beta0': fitted['beta0'],  # Level
                    'beta1': fitted['beta1'],  # Slope
                    'beta2': fitted['beta2'],  # Curvature
                    'lambda': fitted['lambda']
                })
        
        self.factors = pd.DataFrame(factors_list).set_index('date')
        self.factors = self.factors.dropna()
        
        # Rename for clarity
        self.factors.columns = ['Level', 'Slope', 'Curvature', 'Lambda']
        
        print(f"  ✓ Extracted factors for {len(self.factors)} dates")
        print(f"  ✓ Factor means: Level={self.factors['Level'].mean():.2f}, "
              f"Slope={self.factors['Slope'].mean():.2f}, "
              f"Curvature={self.factors['Curvature'].mean():.2f}")
        
        return self.factors
    
    def estimate_var(self, lags=2):
        """
        Estimate VAR model linking factors and regime probabilities
        
        Parameters:
        -----------
        lags : int
            Number of lags in VAR model
        """
        print(f"\nEstimating VAR({lags}) model...")
        
        # Combine factors with regime probabilities
        regime_cols = [col for col in self.regime_probs.columns if 'Regime_' in col]
        
        # Align data
        common_idx = self.factors.index.intersection(self.regime_probs.index)
        
        var_data = pd.concat([
            self.factors.loc[common_idx, ['Level', 'Slope', 'Curvature']],
            self.regime_probs.loc[common_idx, regime_cols]
        ], axis=1).dropna()
        
        print(f"  Variables: {list(var_data.columns)}")
        print(f"  Observations: {len(var_data)}")
        
        # Estimate VAR
        self.var_model = VAR(var_data)
        self.var_results = self.var_model.fit(lags)
        
        print("  ✓ VAR model estimated")
        print(f"  ✓ AIC: {self.var_results.aic:.2f}")
        print(f"  ✓ BIC: {self.var_results.bic:.2f}")
        
        return self.var_results
    
    def forecast_factors(self, steps=6):
        """
        Forecast yield curve factors
        
        Parameters:
        -----------
        steps : int
            Forecast horizon in months
        """
        print(f"\nForecasting {steps} months ahead...")
        
        forecast = self.var_results.forecast(
            self.var_results.endog[-self.var_results.k_ar:],
            steps=steps
        )
        
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.var_results.names,
            index=pd.date_range(
                start=self.var_results.endog_names[0],
                periods=steps,
                freq='M'
            )
        )
        
        self.forecasts = forecast_df
        
        print("  ✓ Forecasts generated")
        print(f"\n  Forecast summary (6-month ahead):")
        print(forecast_df[['Level', 'Slope', 'Curvature']].iloc[-1].round(2))
        
        return forecast_df
    
    def calculate_expected_returns(self):
        """
        Calculate expected bond returns from yield forecasts
        Uses: Return = Carry + Roll-down + Price change from yield movement
        """
        print("\nCalculating expected returns...")
        
        # Duration approximations
        durations = {'2Y': 1.9, '5Y': 4.5, '10Y': 8.5, '30Y': 18.0}
        
        # Current yields
        current_yields = self.yields.iloc[-1]
        
        # Forecasted factors
        forecasted_level = self.forecasts['Level'].iloc[-1]
        forecasted_slope = self.forecasts['Slope'].iloc[-1]
        
        expected_returns = {}
        
        for maturity, duration in durations.items():
            if maturity in current_yields:
                current_yield = current_yields[maturity]
                
                # Carry (current yield over holding period)
                carry = current_yield / 12  # Monthly
                
                # Estimate yield change using forecasted factors
                tau = self.maturities_map[maturity]
                current_fitted = self.nelson_siegel(
                    tau,
                    self.factors['Level'].iloc[-1],
                    self.factors['Slope'].iloc[-1],
                    self.factors['Curvature'].iloc[-1]
                )
                forecast_fitted = self.nelson_siegel(
                    tau,
                    forecasted_level,
                    forecasted_slope,
                    self.factors['Curvature'].iloc[-1]  # Assume curvature stable
                )
                
                yield_change = forecast_fitted - current_fitted
                
                # Price change from yield movement
                price_change = -duration * (yield_change / 100)
                
                # Total expected return
                expected_returns[maturity] = {
                    'carry': carry,
                    'price_change': price_change,
                    'total_return': carry + price_change,
                    'current_yield': current_yield,
                    'forecast_yield': forecast_fitted
                }
        
        self.expected_returns = pd.DataFrame(expected_returns).T
        
        print("  ✓ Expected returns calculated")
        print(f"\n  Expected monthly returns:")
        print(self.expected_returns['total_return'].round(4))
        
        return self.expected_returns
    
    def plot_factors(self, save_path='output/figures/ns_factors.png'):
        """Plot Nelson-Siegel factors over time"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        factor_names = ['Level', 'Slope', 'Curvature']
        colors = ['blue', 'green', 'red']
        
        for ax, factor, color in zip(axes, factor_names, colors):
            ax.plot(self.factors.index, self.factors[factor], 
                   color=color, linewidth=1.5, label=factor)
            ax.set_ylabel(factor, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Add recession shading if regime data available
            if hasattr(self, 'regime_probs'):
                recession_col = [col for col in self.regime_probs.columns 
                               if 'Recession' in col]
                if recession_col:
                    recession_idx = self.regime_probs[recession_col[0]] > 0.5
                    for start, end in self._get_periods(recession_idx):
                        ax.axvspan(start, end, alpha=0.2, color='gray')
        
        axes[0].set_title('Nelson-Siegel Factors: Level, Slope, Curvature', 
                         fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=11)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved factor plot: {save_path}")
        
        return fig
    
    @staticmethod
    def _get_periods(boolean_series):
        """Helper to get start/end dates of True periods"""
        periods = []
        in_period = False
        for i, (date, val) in enumerate(boolean_series.items()):
            if val and not in_period:
                start = date
                in_period = True
            elif not val and in_period:
                periods.append((start, boolean_series.index[i-1]))
                in_period = False
        if in_period:
            periods.append((start, boolean_series.index[-1]))
        return periods
    
    def save_results(self, output_dir='data/processed'):
        """Save factors and forecasts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.factors.to_csv(f'{output_dir}/ns_factors.csv')
        print(f"\n✓ Saved Nelson-Siegel factors: {output_dir}/ns_factors.csv")
        
        if self.expected_returns is not None:
            self.expected_returns.to_csv(f'{output_dir}/expected_returns.csv')
            print(f"✓ Saved expected returns: {output_dir}/expected_returns.csv")
        
        return self.factors

def main():
    """Main execution"""
    print("="*60)
    print("YIELD CURVE MODELING")
    print("="*60)
    
    # Initialize model
    yc_model = YieldCurveModel()
    
    # Load data
    yc_model.load_data()
    
    # Extract factors
    yc_model.extract_factors()
    
    # Estimate VAR
    yc_model.estimate_var(lags=2)
    
    # Generate forecasts
    yc_model.forecast_factors(steps=6)
    
    # Calculate expected returns
    yc_model.calculate_expected_returns()
    
    # Visualize
    yc_model.plot_factors()
    
    # Save results
    yc_model.save_results()
    
    print("\n" + "="*60)
    print("YIELD CURVE MODELING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()"""
Yield Curve Modeling using Nelson-Siegel and VAR Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class YieldCurveModel:
    def __init__(self):
        """Initialize yield curve model"""
        self.yields = None
        self.factors = None
        self.var_model = None
        self.forecasts = None
        
        # Standard maturities in years for Nelson-Siegel
        self.maturities_map = {
            '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0,
            '3Y': 3.0, '5Y': 5.0, '7Y': 7.0, '10Y': 10.0,
            '20Y': 20.0, '30Y': 30.0
        }
        
    def load_data(self, yields_path='data/raw/treasury_yields.csv', 
                  regime_path='data/processed/regime_probabilities.csv'):
        """Load yield curve and regime data"""
        print("Loading yield curve data...")
        
        self.yields = pd.read_csv(yields_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        
        # Align dates
        common_idx = self.yields.index.intersection(self.regime_probs.index)
        self.yields = self.yields.loc[common_idx]
        self.regime_probs = self.regime_probs.loc[common_idx]
        
        print(f"  ✓ Loaded {len(self.yields)} observations")
        print(f"  ✓ Maturities: {list(self.yields.columns)}")
        
        return self.yields
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda_param=0.0609):
        """
        Nelson-Siegel yield curve model
        
        y(tau) = beta0 + beta1*((1-exp(-lambda*tau))/(lambda*tau)) 
                      + beta2*((1-exp(-lambda*tau))/(lambda*tau) - exp(-lambda*tau))
        
        Parameters:
        -----------
        tau : array
            Maturities in years
        beta0 : float
            Level factor (long-term rate)
        beta1 : float
            Slope factor (short-term component)
        beta2 : float
            Curvature factor (medium-term component)
        lambda_param : float
            Decay parameter (typically around 0.0609 for monthly data)
        """
        tau = np.array(tau)
        exp_term = np.exp(-lambda_param * tau)
        
        # Avoid division by zero
        factor1 = np.where(tau > 0, (1 - exp_term) / (lambda_param * tau), 1.0)
        factor2 = factor1 - exp_term
        
        return beta0 + beta1 * factor1 + beta2 * factor2
    
    def fit_nelson_siegel(self, yields_row, maturities):
        """
        Fit Nelson-Siegel model to a single yield curve observation
        
        Returns:
        --------
        dict with beta0, beta1, beta2, lambda_param
        """
        # Initial guesses
        beta0_init = yields_row[-1]  # Long-term yield
        beta1_init = yields_row[0] - yields_row[-1]  # Short-long spread
        beta2_init = 2 * yields_row[len(yields_row)//2] - yields_row[0] - yields_row[-1]
        lambda_init = 0.0609
        
        # Objective function: minimize squared errors
        def objective(params):
            beta0, beta1, beta2, lambda_param = params
            fitted = self.nelson_siegel(maturities, beta0, beta1, beta2, lambda_param)
            return np.sum((yields_row - fitted)**2)
        
        # Optimize
        result = minimize(
            objective,
            x0=[beta0_init, beta1_init, beta2_init, lambda_init],
            method='L-BFGS-B',
            bounds=[
                (None, None),  # beta0
                (None, None),  # beta1
                (None, None),  # beta2
                (0.001, 1.0)   # lambda (must be positive)
            ]
        )
        
        if result.success:
            return {
                'beta0': result.x[0],
                'beta1': result.x[1],
                'beta2': result.x[2],
                'lambda': result.x[3],
                'fit_error': result.fun
            }
        else:
            return None
    
    def extract_factors(self):
        """Extract Nelson-Siegel factors for all dates"""
        print("\nExtracting Nelson-Siegel factors...")
        
        # Get maturities in years
        maturities = np.array([self.maturities_map[col] for col in self.yields.columns 
                               if col in self.maturities_map])
        
        # Fit NS model for each date
        factors_list = []
        for date, row in self.yields.iterrows():
            # Get yields that have maturity info
            yields_arr = np.array([row[col] for col in self.yields.columns 
                                  if col in self.maturities_map])
            
            # Remove NaN values
            valid_idx = ~np.isnan(yields_arr)
            if valid_idx.sum() < 4:  # Need at least 4 points
                factors_list.append({'date': date, 'beta0': np.nan, 'beta1': np.nan, 
                                   'beta2': np.nan, 'lambda': np.nan})
                continue
            
            fitted = self.fit_nelson_siegel(
                yields_arr[valid_idx],
                maturities[valid_idx]
            )
            
            if fitted:
                factors_list.append({
                    'date': date,
                    'beta0': fitted['beta0'],  # Level
                    'beta1': fitted['beta1'],  # Slope
                    'beta2': fitted['beta2'],  # Curvature
                    'lambda': fitted['lambda']
                })
        
        self.factors = pd.DataFrame(factors_list).set_index('date')
        self.factors = self.factors.dropna()
        
        # Rename for clarity
        self.factors.columns = ['Level', 'Slope', 'Curvature', 'Lambda']
        
        print(f"  ✓ Extracted factors for {len(self.factors)} dates")
        print(f"  ✓ Factor means: Level={self.factors['Level'].mean():.2f}, "
              f"Slope={self.factors['Slope'].mean():.2f}, "
              f"Curvature={self.factors['Curvature'].mean():.2f}")
        
        return self.factors
    
    def estimate_var(self, lags=2):
        """
        Estimate VAR model linking factors and regime probabilities
        
        Parameters:
        -----------
        lags : int
            Number of lags in VAR model
        """
        print(f"\nEstimating VAR({lags}) model...")
        
        # Combine factors with regime probabilities
        regime_cols = [col for col in self.regime_probs.columns if 'Regime_' in col]
        
        # Align data
        common_idx = self.factors.index.intersection(self.regime_probs.index)
        
        var_data = pd.concat([
            self.factors.loc[common_idx, ['Level', 'Slope', 'Curvature']],
            self.regime_probs.loc[common_idx, regime_cols]
        ], axis=1).dropna()
        
        print(f"  Variables: {list(var_data.columns)}")
        print(f"  Observations: {len(var_data)}")
        
        # Estimate VAR
        self.var_model = VAR(var_data)
        self.var_results = self.var_model.fit(lags)
        
        print("  ✓ VAR model estimated")
        print(f"  ✓ AIC: {self.var_results.aic:.2f}")
        print(f"  ✓ BIC: {self.var_results.bic:.2f}")
        
        return self.var_results
    
    def forecast_factors(self, steps=6):
        """
        Forecast yield curve factors
        
        Parameters:
        -----------
        steps : int
            Forecast horizon in months
        """
        print(f"\nForecasting {steps} months ahead...")
        
        forecast = self.var_results.forecast(
            self.var_results.endog[-self.var_results.k_ar:],
            steps=steps
        )
        
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.var_results.names,
            index=pd.date_range(
                start=self.var_results.endog_names[0],
                periods=steps,
                freq='M'
            )
        )
        
        self.forecasts = forecast_df
        
        print("  ✓ Forecasts generated")
        print(f"\n  Forecast summary (6-month ahead):")
        print(forecast_df[['Level', 'Slope', 'Curvature']].iloc[-1].round(2))
        
        return forecast_df
    
    def calculate_expected_returns(self):
        """
        Calculate expected bond returns from yield forecasts
        Uses: Return = Carry + Roll-down + Price change from yield movement
        """
        print("\nCalculating expected returns...")
        
        # Duration approximations
        durations = {'2Y': 1.9, '5Y': 4.5, '10Y': 8.5, '30Y': 18.0}
        
        # Current yields
        current_yields = self.yields.iloc[-1]
        
        # Forecasted factors
        forecasted_level = self.forecasts['Level'].iloc[-1]
        forecasted_slope = self.forecasts['Slope'].iloc[-1]
        
        expected_returns = {}
        
        for maturity, duration in durations.items():
            if maturity in current_yields:
                current_yield = current_yields[maturity]
                
                # Carry (current yield over holding period)
                carry = current_yield / 12  # Monthly
                
                # Estimate yield change using forecasted factors
                tau = self.maturities_map[maturity]
                current_fitted = self.nelson_siegel(
                    tau,
                    self.factors['Level'].iloc[-1],
                    self.factors['Slope'].iloc[-1],
                    self.factors['Curvature'].iloc[-1]
                )
                forecast_fitted = self.nelson_siegel(
                    tau,
                    forecasted_level,
                    forecasted_slope,
                    self.factors['Curvature'].iloc[-1]  # Assume curvature stable
                )
                
                yield_change = forecast_fitted - current_fitted
                
                # Price change from yield movement
                price_change = -duration * (yield_change / 100)
                
                # Total expected return
                expected_returns[maturity] = {
                    'carry': carry,
                    'price_change': price_change,
                    'total_return': carry + price_change,
                    'current_yield': current_yield,
                    'forecast_yield': forecast_fitted
                }
        
        self.expected_returns = pd.DataFrame(expected_returns).T
        
        print("  ✓ Expected returns calculated")
        print(f"\n  Expected monthly returns:")
        print(self.expected_returns['total_return'].round(4))
        
        return self.expected_returns
    
    def plot_factors(self, save_path='output/figures/ns_factors.png'):
        """Plot Nelson-Siegel factors over time"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        factor_names = ['Level', 'Slope', 'Curvature']
        colors = ['blue', 'green', 'red']
        
        for ax, factor, color in zip(axes, factor_names, colors):
            ax.plot(self.factors.index, self.factors[factor], 
                   color=color, linewidth=1.5, label=factor)
            ax.set_ylabel(factor, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Add recession shading if regime data available
            if hasattr(self, 'regime_probs'):
                recession_col = [col for col in self.regime_probs.columns 
                               if 'Recession' in col]
                if recession_col:
                    recession_idx = self.regime_probs[recession_col[0]] > 0.5
                    for start, end in self._get_periods(recession_idx):
                        ax.axvspan(start, end, alpha=0.2, color='gray')
        
        axes[0].set_title('Nelson-Siegel Factors: Level, Slope, Curvature', 
                         fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=11)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved factor plot: {save_path}")
        
        return fig
    
    @staticmethod
    def _get_periods(boolean_series):
        """Helper to get start/end dates of True periods"""
        periods = []
        in_period = False
        for i, (date, val) in enumerate(boolean_series.items()):
            if val and not in_period:
                start = date
                in_period = True
            elif not val and in_period:
                periods.append((start, boolean_series.index[i-1]))
                in_period = False
        if in_period:
            periods.append((start, boolean_series.index[-1]))
        return periods
    
    def save_results(self, output_dir='data/processed'):
        """Save factors and forecasts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.factors.to_csv(f'{output_dir}/ns_factors.csv')
        print(f"\n✓ Saved Nelson-Siegel factors: {output_dir}/ns_factors.csv")
        
        if self.expected_returns is not None:
            self.expected_returns.to_csv(f'{output_dir}/expected_returns.csv')
            print(f"✓ Saved expected returns: {output_dir}/expected_returns.csv")
        
        return self.factors

def main():
    """Main execution"""
    print("="*60)
    print("YIELD CURVE MODELING")
    print("="*60)
    
    # Initialize model
    yc_model = YieldCurveModel()
    
    # Load data
    yc_model.load_data()
    
    # Extract factors
    yc_model.extract_factors()
    
    # Estimate VAR
    yc_model.estimate_var(lags=2)
    
    # Generate forecasts
    yc_model.forecast_factors(steps=6)
    
    # Calculate expected returns
    yc_model.calculate_expected_returns()
    
    # Visualize
    yc_model.plot_factors()
    
    # Save results
    yc_model.save_results()
    
    print("\n" + "="*60)
    print("YIELD CURVE MODELING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
