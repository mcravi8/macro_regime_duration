"""
Portfolio Optimization for Duration-Based Treasury Strategies
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self):
        """Initialize portfolio optimizer"""
        self.returns = None
        self.regime_probs = None
        self.weights_history = []
        
        # Portfolio universe: Treasury maturities
        self.universe = ['2Y', '5Y', '10Y', '30Y']
        self.durations = {'2Y': 1.9, '5Y': 4.5, '10Y': 8.5, '30Y': 18.0}
        
    def load_data(self, 
                  returns_path='data/raw/treasury_returns.csv',
                  regime_path='data/processed/regime_probabilities.csv'):
        """Load returns and regime data"""
        print("Loading data for portfolio optimization...")
        
        self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        self.regime_probs = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        
        # Filter to our universe
        self.returns = self.returns[self.universe]
        
        # Align dates
        common_idx = self.returns.index.intersection(self.regime_probs.index)
        self.returns = self.returns.loc[common_idx]
        self.regime_probs = self.regime_probs.loc[common_idx]
        
        # Drop NaN
        self.returns = self.returns.dropna()
        self.regime_probs = self.regime_probs.loc[self.returns.index]
        
        print(f"  ✓ Loaded {len(self.returns)} observations")
        print(f"  ✓ Universe: {self.universe}")
        
        return self.returns
    
    def calculate_regime_statistics(self, lookback=120):
        """
        Calculate regime-conditional return statistics
        
        Parameters:
        -----------
        lookback : int
            Months of historical data to use
        """
        print("\nCalculating regime-conditional statistics...")
        
        # Identify most likely regime for each period
        regime_cols = [col for col in self.regime_probs.columns if 'Regime_' in col]
        most_likely = self.regime_probs[regime_cols].idxmax(axis=1)
        
        self.regime_stats = {}
        
        for regime in regime_cols:
            mask = (most_likely == regime) & (self.returns.notna().all(axis=1))
            
            if mask.sum() > 20:  # Need sufficient observations
                regime_returns = self.returns[mask]
                
                self.regime_stats[regime] = {
                    'mean': regime_returns.mean(),
                    'cov': regime_returns.cov(),
                    'observations': mask.sum()
                }
                
                print(f"  {regime}: {mask.sum()} observations")
                print(f"    Mean returns: {regime_returns.mean().round(4).to_dict()}")
        
        return self.regime_stats
    
    def optimize_portfolio(self, 
                          expected_returns, 
                          cov_matrix,
                          target_duration=6.0,
                          duration_tolerance=2.0,
                          max_weight=0.5,
                          risk_aversion=2.0):
        """
        Optimize portfolio using mean-variance framework with constraints
        
        Parameters:
        -----------
        expected_returns : Series
            Expected returns for each asset
        cov_matrix : DataFrame
            Covariance matrix
        target_duration : float
            Target portfolio duration
        duration_tolerance : float
            +/- tolerance for duration constraint
        max_weight : float
            Maximum weight in any single security
        risk_aversion : float
            Risk aversion parameter (lambda)
        
        Returns:
        --------
        Series with optimal weights
        """
        n = len(self.universe)
        
        # Decision variables
        w = cp.Variable(n)
        
        # Portfolio statistics
        mu = expected_returns.values
        sigma = cov_matrix.values
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, sigma)
        
        # Portfolio duration
        durations_array = np.array([self.durations[asset] for asset in self.universe])
        portfolio_duration = durations_array @ w
        
        # Objective: maximize return - risk_aversion * variance
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0,  # Long-only
            w <= max_weight,  # Diversification
            portfolio_duration >= target_duration - duration_tolerance,
            portfolio_duration <= target_duration + duration_tolerance
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status == 'optimal':
            weights = pd.Series(w.value, index=self.universe)
            
            # Calculate portfolio stats
            port_return = weights @ expected_returns
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            port_duration = weights @ durations_array
            
            return {
                'weights': weights,
                'return': port_return,
                'volatility': port_vol,
                'duration': port_duration,
                'sharpe': port_return / port_vol if port_vol > 0 else 0
            }
        else:
            print(f"  ⚠ Optimization failed: {problem.status}")
            # Return equal-weight fallback
            weights = pd.Series(1/n, index=self.universe)
            return {
                'weights': weights,
                'return': weights @ expected_returns,
                'volatility': np.sqrt(weights @ cov_matrix @ weights),
                'duration': weights @ durations_array,
                'sharpe': 0
            }
    
    def regime_conditional_allocation(self, current_regime_probs, lookback=120):
        """
        Determine allocation based on current regime probabilities
        
        Parameters:
        -----------
        current_regime_probs : Series
            Current regime probabilities
        lookback : int
            Historical window for statistics
        """
        # Get regime-weighted expected returns and covariance
        regime_cols = [col for col in current_regime_probs.index if 'Regime_' in col]
        
        weighted_mean = pd.Series(0.0, index=self.universe)
        weighted_cov = pd.DataFrame(0.0, index=self.universe, columns=self.universe)
        
        for regime in regime_cols:
            if regime in self.regime_stats:
                prob = current_regime_probs[regime]
                weighted_mean += prob * self.regime_stats[regime]['mean']
                weighted_cov += prob * self.regime_stats[regime]['cov']
        
        # Adjust target duration based on regime
        # If recession probability high → extend duration
        # If expansion probability high → normal duration
        recession_prob = current_regime_probs.get('Recession', 0)
        expansion_prob = current_regime_probs.get('Expansion', 0)
        
        base_duration = 6.0
        duration_adjustment = 2.0 * (recession_prob - expansion_prob)
        target_duration = base_duration + duration_adjustment
        target_duration = np.clip(target_duration, 4.0, 10.0)
        
        # Optimize
        portfolio = self.optimize_portfolio(
            weighted_mean,
            weighted_cov,
            target_duration=target_duration,
            duration_tolerance=2.0,
            risk_aversion=2.0
        )
        
        portfolio['target_duration'] = target_duration
        portfolio['regime_probs'] = current_regime_probs
        
        return portfolio
    
    def backtest(self, rebalance_freq=1, lookback=120, start_date=None):
        """
        Backtest regime-conditional portfolio strategy
        
        Parameters:
        -----------
        rebalance_freq : int
            Rebalance frequency in months
        lookback : int
            Historical window for statistics (months)
        start_date : str
            Start date for backtest (default: None uses first available)
        """
        print("\nRunning backtest...")
        
        # Calculate regime statistics
        self.calculate_regime_statistics(lookback=lookback)
        
        # Initialize results storage
        results = []
        weights_history = []
        
        # Start after sufficient lookback period
        if start_date:
            backtest_start = pd.to_datetime(start_date)
        else:
            backtest_start = self.returns.index[lookback]
        
        backtest_dates = self.returns.index[self.returns.index >= backtest_start]
        
        print(f"  Backtest period: {backtest_dates[0]} to {backtest_dates[-1]}")
        print(f"  Observations: {len(backtest_dates)}")
        
        current_weights = None
        
        for i, date in enumerate(backtest_dates):
            # Rebalance check
            if i % rebalance_freq == 0 or current_weights is None:
                # Get current regime probabilities
                current_regime = self.regime_probs.loc[date]
                
                # Optimize portfolio
                portfolio = self.regime_conditional_allocation(
                    current_regime,
                    lookback=lookback
                )
                
                current_weights = portfolio['weights']
                
                weights_history.append({
                    'date': date,
                    **current_weights.to_dict(),
                    'target_duration': portfolio['target_duration']
                })
            
            # Calculate return for this period
            if date in self.returns.index:
                period_returns = self.returns.loc[date]
                portfolio_return = (current_weights * period_returns).sum()
                
                results.append({
                    'date': date,
                    'return': portfolio_return,
                    **current_weights.to_dict()
                })
        
        # Convert to DataFrames
        self.backtest_results = pd.DataFrame(results).set_index('date')
        self.weights_history = pd.DataFrame(weights_history).set_index('date')
        
        # Calculate cumulative returns
        self.backtest_results['cumulative_return'] = \
            (1 + self.backtest_results['return']).cumprod()
        
        # Calculate benchmark (equal-weight)
        benchmark_weights = pd.Series(0.25, index=self.universe)
        benchmark_returns = (self.returns.loc[backtest_dates] * benchmark_weights).sum(axis=1)
        self.backtest_results['benchmark_return'] = benchmark_returns
        self.backtest_results['benchmark_cumulative'] = \
            (1 + benchmark_returns).cumprod()
        
        print("\n  ✓ Backtest complete")
        self._print_performance_summary()
        
        return self.backtest_results
    
    def _print_performance_summary(self):
        """Print backtest performance metrics"""
        strategy_ret = self.backtest_results['return']
        benchmark_ret = self.backtest_results['benchmark_return']
        
        def calc_metrics(returns):
            total_return = (1 + returns).prod() - 1
            ann_return = (1 + total_return) ** (12 / len(returns)) - 1
            ann_vol = returns.std() * np.sqrt(12)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            max_dd = (returns.cumsum() - returns.cumsum().expanding().max()).min()
            
            return {
                'Total Return': f"{total_return*100:.2f}%",
                'Annualized Return': f"{ann_return*100:.2f}%",
                'Annualized Vol': f"{ann_vol*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_dd*100:.2f}%"
            }
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print("\nStrategy:")
        for metric, value in calc_metrics(strategy_ret).items():
            print(f"  {metric}: {value}")
        
        print("\nBenchmark (Equal-Weight):")
        for metric, value in calc_metrics(benchmark_ret).items():
            print(f"  {metric}: {value}")
        
        # Excess return
        excess = strategy_ret.mean() - benchmark_ret.mean()
        print(f"\nAverage Excess Return: {excess*12*100:.2f}% annualized")
    
    def plot_performance(self, save_path='output/figures/portfolio_performance.png'):
        """Plot backtest performance"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative returns
        axes[0].plot(self.backtest_results.index, 
                    self.backtest_results['cumulative_return'],
                    label='Regime-Conditional Strategy', linewidth=2, color='blue')
        axes[0].plot(self.backtest_results.index,
                    self.backtest_results['benchmark_cumulative'],
                    label='Equal-Weight Benchmark', linewidth=2, 
                    color='gray', linestyle='--')
        axes[0].set_ylabel('Cumulative Return', fontsize=11)
        axes[0].set_title('Portfolio Performance: Regime-Conditional Duration Strategy',
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Weights over time
        for col in self.universe:
            axes[1].plot(self.weights_history.index,
                        self.weights_history[col],
                        label=col, linewidth=1.5)
        
        axes[1].set_ylabel('Portfolio Weight', fontsize=11)
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].set_title('Dynamic Portfolio Weights', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10, ncol=4)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved performance plot: {save_path}")
        
        return fig
    
    def save_results(self, output_dir='data/processed'):
        """Save backtest results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.backtest_results.to_csv(f'{output_dir}/backtest_results.csv')
        self.weights_history.to_csv(f'{output_dir}/portfolio_weights.csv')
        
        print(f"\n✓ Saved backtest results: {output_dir}/backtest_results.csv")
        print(f"✓ Saved portfolio weights: {output_dir}/portfolio_weights.csv")

def main():
    """Main execution"""
    print("="*60)
    print("PORTFOLIO OPTIMIZATION & BACKTESTING")
    print("="*60)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Load data
    optimizer.load_data()
    
    # Run backtest
    optimizer.backtest(
        rebalance_freq=1,  # Monthly rebalancing
        lookback=120,      # 10-year lookback
        start_date='2005-01-01'
    )
    
    # Plot results
    optimizer.plot_performance()
    
    # Save results
    optimizer.save_results()
    
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
