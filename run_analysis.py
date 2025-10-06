"""
Main Runner Script
Executes the complete analysis pipeline
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_collection.collect_data import DataCollector
from src.regime_model.regime_model import RegimeModel
from src.yield_curve.yield_curve_model import YieldCurveModel
from src.portfolio_opt.portfolio_optimization import PortfolioOptimizer
import config

def run_full_pipeline():
    """Execute complete analysis pipeline"""
    
    print("\n" + "="*80)
    print(" "*20 + "MACRO REGIME DURATION PROJECT")
    print(" "*15 + "Complete Analysis Pipeline Execution")
    print("="*80)
    
    # Step 1: Data Collection
    print("\n" + "─"*80)
    print("STEP 1/4: DATA COLLECTION")
    print("─"*80)
    
    try:
        collector = DataCollector()
        datasets = collector.save_all_data(output_dir=config.PATHS['data_raw'])
        print("✓ Data collection successful")
    except Exception as e:
        print(f"✗ Data collection failed: {e}")
        print("\nPlease ensure:")
        print("1. You have a FRED API key in .env file")
        print("2. Internet connection is active")
        return False
    
    # Step 2: Regime Model
    print("\n" + "─"*80)
    print("STEP 2/4: REGIME IDENTIFICATION")
    print("─"*80)
    
    try:
        regime_model = RegimeModel(n_regimes=config.REGIME_CONFIG['n_regimes'])
        regime_model.prepare_data(data_path=f"{config.PATHS['data_raw']}/macro_data.csv")
        regime_model.estimate_model(
            dependent_var=config.REGIME_CONFIG['dependent_var'],
            exog_vars=config.REGIME_CONFIG['exog_vars']
        )
        regime_model.print_summary()
        regime_model.plot_regime_probabilities(
            save_path=f"{config.PATHS['output_figures']}/regime_probabilities.png"
        )
        regime_model.plot_regime_scatter(
            save_path=f"{config.PATHS['output_figures']}/regime_scatter.png"
        )
        regime_model.save_results(output_dir=config.PATHS['data_processed'])
        print("✓ Regime modeling successful")
    except Exception as e:
        print(f"✗ Regime modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Yield Curve Model
    print("\n" + "─"*80)
    print("STEP 3/4: YIELD CURVE MODELING")
    print("─"*80)
    
    try:
        yc_model = YieldCurveModel()
        yc_model.load_data(
            yields_path=f"{config.PATHS['data_raw']}/treasury_yields.csv",
            regime_path=f"{config.PATHS['data_processed']}/regime_probabilities.csv"
        )
        yc_model.extract_factors()
        yc_model.estimate_var(lags=config.YIELD_CURVE_CONFIG['var_lags'])
        yc_model.forecast_factors(steps=config.YIELD_CURVE_CONFIG['forecast_horizon'])
        yc_model.calculate_expected_returns()
        yc_model.plot_factors(
            save_path=f"{config.PATHS['output_figures']}/ns_factors.png"
        )
        yc_model.save_results(output_dir=config.PATHS['data_processed'])
        print("✓ Yield curve modeling successful")
    except Exception as e:
        print(f"✗ Yield curve modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Portfolio Optimization
    print("\n" + "─"*80)
    print("STEP 4/4: PORTFOLIO OPTIMIZATION & BACKTESTING")
    print("─"*80)
    
    try:
        optimizer = PortfolioOptimizer()
        optimizer.load_data(
            returns_path=f"{config.PATHS['data_raw']}/treasury_returns.csv",
            regime_path=f"{config.PATHS['data_processed']}/regime_probabilities.csv"
        )
        optimizer.backtest(
            rebalance_freq=config.PORTFOLIO_CONFIG['rebalance_freq'],
            lookback=config.PORTFOLIO_CONFIG['lookback_window'],
            start_date=config.BACKTEST_CONFIG['start_date']
        )
        optimizer.plot_performance(
            save_path=f"{config.PATHS['output_figures']}/portfolio_performance.png"
        )
        optimizer.save_results(output_dir=config.PATHS['data_processed'])
        print("✓ Portfolio optimization successful")
    except Exception as e:
        print(f"✗ Portfolio optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nResults saved to:")
    print(f"  📊 Figures: {config.PATHS['output_figures']}/")
    print(f"  📁 Data: {config.PATHS['data_processed']}/")
    
    print("\nNext steps:")
    print("  1. Review visualizations in output/figures/")
    print("  2. Examine detailed results in data/processed/")
    print("  3. Write research paper using insights and figures")
    print("  4. Create presentation deck for interviews")
    
    return True

def run_data_only():
    """Run only data collection step"""
    print("\nRunning data collection only...")
    collector = DataCollector()
    collector.save_all_data(output_dir=config.PATHS['data_raw'])
    print("\n✓ Data collection complete")

def run_regime_only():
    """Run only regime modeling step"""
    print("\nRunning regime modeling only...")
    regime_model = RegimeModel(n_regimes=config.REGIME_CONFIG['n_regimes'])
    regime_model.prepare_data()
    regime_model.estimate_model()
    regime_model.print_summary()
    regime_model.plot_regime_probabilities()
    regime_model.plot_regime_scatter()
    regime_model.save_results()
    print("\n✓ Regime modeling complete")

def run_yield_curve_only():
    """Run only yield curve modeling step"""
    print("\nRunning yield curve modeling only...")
    yc_model = YieldCurveModel()
    yc_model.load_data()
    yc_model.extract_factors()
    yc_model.estimate_var()
    yc_model.forecast_factors()
    yc_model.calculate_expected_returns()
    yc_model.plot_factors()
    yc_model.save_results()
    print("\n✓ Yield curve modeling complete")

def run_portfolio_only():
    """Run only portfolio optimization step"""
    print("\nRunning portfolio optimization only...")
    optimizer = PortfolioOptimizer()
    optimizer.load_data()
    optimizer.backtest()
    optimizer.plot_performance()
    optimizer.save_results()
    print("\n✓ Portfolio optimization complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Macro Regime Duration Project - Analysis Pipeline'
    )
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'data', 'regime', 'yield_curve', 'portfolio'],
        help='Which step to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run selected step
    if args.step == 'all':
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
    elif args.step == 'data':
        run_data_only()
    elif args.step == 'regime':
        run_regime_only()
    elif args.step == 'yield_curve':
        run_yield_curve_only()
    elif args.step == 'portfolio':
        run_portfolio_only()
