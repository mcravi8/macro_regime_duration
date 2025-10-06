"""
Example Interactive Analysis Script
Demonstrates how to use the framework components interactively
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('.')

from src.data_collection.collect_data import DataCollector
from src.regime_model.regime_model import RegimeModel
from src.yield_curve.yield_curve_model import YieldCurveModel
from src.portfolio_opt.portfolio_optimization import PortfolioOptimizer

def example_regime_analysis():
    """Example: Analyze current regime probabilities"""
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Current Regime Analysis")
    print("="*60)
    
    # Load regime probabilities
    regime_probs = pd.read_csv('data/processed/regime_probabilities.csv', 
                               index_col=0, parse_dates=True)
    
    # Get latest regime probabilities
    latest = regime_probs.iloc[-1]
    
    print("\nCurrent Regime Probabilities (as of {}):".format(
        regime_probs.index[-1].strftime('%Y-%m-%d')
    ))
    print("-" * 60)
    
    for col in regime_probs.columns:
        if 'Regime_' in col or any(label in col for label in ['Recession', 'Expansion', 'Moderate']):
            print(f"  {col}: {latest[col]:.1%}")
    
    # Identify most likely regime
    regime_cols = [col for col in regime_probs.columns if 'Regime_' in col]
    most_likely = latest[regime_cols].idxmax()
    print(f"\n  → Most likely regime: {most_likely}")
    
    # Historical regime summary
    print("\n\nHistorical Regime Statistics:")
    print("-" * 60)
    
    for col in regime_cols:
        pct_time = (regime_probs[col] > 0.5).mean()
        avg_prob = regime_probs[col].mean()
        print(f"  {col}:")
        print(f"    Time in regime: {pct_time:.1%}")
        print(f"    Average probability: {avg_prob:.1%}")

def example_yield_curve_forecast():
    """Example: Generate yield curve forecast"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Yield Curve Forecast")
    print("="*60)
    
    # Load current yields
    yields = pd.read_csv('data/raw/treasury_yields.csv', 
                        index_col=0, parse_dates=True)
    
    # Load factors
    factors = pd.read_csv('data/processed/ns_factors.csv',
                         index_col=0, parse_dates=True)
    
    # Display current curve
    latest_yields = yields.iloc[-1].dropna()
    latest_factors = factors.iloc[-1]
    
    print("\nCurrent Yield Curve (as of {}):".format(
        yields.index[-1].strftime('%Y-%m-%d')
    ))
    print("-" * 60)
    
    for maturity, yield_val in latest_yields.items():
        print(f"  {maturity}: {yield_val:.2f}%")
    
    print("\n\nNelson-Siegel Factors:")
    print("-" * 60)
    print(f"  Level (long-term rate): {latest_factors['Level']:.2f}%")
    print(f"  Slope (term spread): {latest_factors['Slope']:.2f}%")
    print(f"  Curvature (hump): {latest_factors['Curvature']:.2f}%")
    
    # Interpretation
    print("\n\nInterpretation:")
    print("-" * 60)
    
    if latest_factors['Slope'] < 0:
        print("  ⚠ Inverted yield curve detected (slope < 0)")
        print("    → Potential recession signal")
    elif latest_factors['Slope'] > 1:
        print("  ✓ Steep yield curve (slope > 1)")
        print("    → Expansionary environment")
    else:
        print("  ○ Flat yield curve")
        print("    → Transition period")

def example_portfolio_recommendation():
    """Example: Generate current portfolio recommendation"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Portfolio Recommendation")
    print("="*60)
    
    # Load regime probabilities
    regime_probs = pd.read_csv('data/processed/regime_probabilities.csv',
                               index_col=0, parse_dates=True)
    
    # Load expected returns
    expected_returns = pd.read_csv('data/processed/expected_returns.csv',
                                  index_col=0)
    
    # Get current regime
    latest_regime = regime_probs.iloc[-1]
    
    # Determine allocation based on regime
    recession_prob = latest_regime.get('Recession', 0)
    expansion_prob = latest_regime.get('Expansion', 0)
    
    print("\nCurrent Market Environment:")
    print("-" * 60)
    print(f"  Recession probability: {recession_prob:.1%}")
    print(f"  Expansion probability: {expansion_prob:.1%}")
    
    # Duration recommendation
    base_duration = 6.0
    duration_adjustment = 2.0 * (recession_prob - expansion_prob)
    target_duration = np.clip(base_duration + duration_adjustment, 4.0, 10.0)
    
    print(f"\n  → Target duration: {target_duration:.1f} years")
    
    if target_duration > 7:
        stance = "DEFENSIVE (long duration)"
        rationale = "High recession probability → extend duration to capture rally"
    elif target_duration < 5:
        stance = "AGGRESSIVE (short duration)"
        rationale = "High expansion probability → shorten duration, avoid rate risk"
    else:
        stance = "NEUTRAL"
        rationale = "Mixed signals → maintain balanced duration"
    
    print(f"  → Portfolio stance: {stance}")
    print(f"  → Rationale: {rationale}")
    
    # Sample allocation
    print("\n\nSample Allocation:")
    print("-" * 60)
    
    if target_duration > 7:
        allocation = {'2Y': 0.10, '5Y': 0.20, '10Y': 0.40, '30Y': 0.30}
    elif target_duration < 5:
        allocation = {'2Y': 0.50, '5Y': 0.30, '10Y': 0.15, '30Y': 0.05}
    else:
        allocation = {'2Y': 0.25, '5Y': 0.30, '10Y': 0.30, '30Y': 0.15}
    
    for maturity, weight in allocation.items():
        print(f"  {maturity}: {weight:.1%}")
    
    # Expected returns
    print("\n\nExpected Monthly Returns:")
    print("-" * 60)
    
    for maturity in ['2Y', '5Y', '10Y', '30Y']:
        if maturity in expected_returns.index:
            exp_ret = expected_returns.loc[maturity, 'total_return']
            print(f"  {maturity}: {exp_ret:.2%}")

def example_backtest_summary():
    """Example: Summarize backtest performance"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Backtest Performance Summary")
    print("="*60)
    
    # Load backtest results
    results = pd.read_csv('data/processed/backtest_results.csv',
                         index_col=0, parse_dates=True)
    
    # Calculate metrics
    strategy_ret = results['return']
    benchmark_ret = results['benchmark_return']
    
    def calc_metrics(returns):
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (12 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'Total Return': total_ret,
            'Ann. Return': ann_ret,
            'Ann. Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }
    
    strat_metrics = calc_metrics(strategy_ret)
    bench_metrics = calc_metrics(benchmark_ret)
    
    print("\nStrategy Performance:")
    print("-" * 60)
    for metric, value in strat_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.1%}")
    
    print("\n\nBenchmark Performance:")
    print("-" * 60)
    for metric, value in bench_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.1%}")
    
    print("\n\nAlpha Analysis:")
    print("-" * 60)
    excess_ret = strat_metrics['Ann. Return'] - bench_metrics['Ann. Return']
    info_ratio = (strategy_ret - benchmark_ret).mean() / (strategy_ret - benchmark_ret).std() * np.sqrt(12)
    
    print(f"  Excess return: {excess_ret:.1%} per year")
    print(f"  Information ratio: {info_ratio:.2f}")
    print(f"  Win rate: {(strategy_ret > benchmark_ret).mean():.1%}")

def main():
    """Run all examples"""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "INTERACTIVE ANALYSIS EXAMPLES" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")
    
    try:
        example_regime_analysis()
    except Exception as e:
        print(f"\n⚠ Could not run regime analysis: {e}")
        print("  Make sure to run the full pipeline first: python run_analysis.py")
    
    try:
        example_yield_curve_forecast()
    except Exception as e:
        print(f"\n⚠ Could not run yield curve forecast: {e}")
    
    try:
        example_portfolio_recommendation()
    except Exception as e:
        print(f"\n⚠ Could not run portfolio recommendation: {e}")
    
    try:
        example_backtest_summary()
    except Exception as e:
        print(f"\n⚠ Could not run backtest summary: {e}")
    
    print("\n" + "═"*70)
    print("For full analysis, run: python run_analysis.py")
    print("═"*70 + "\n")

if __name__ == "__main__":
    main()
