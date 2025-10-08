"""
Main Runner Script
Executes the complete analysis pipeline
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_collection.collect_data import DataCollector
from src.regime_model.model import RegimeModel
from src.yield_curve.yield_curve_model import YieldCurveModel
from src.portfolio_opt.portfolio_optimization import PortfolioOptimizer
import config


# =====================================================================
# FULL PIPELINE EXECUTION
# =====================================================================
def run_full_pipeline():
    """Execute complete analysis pipeline"""

    print("\n" + "="*80)
    print(" "*20 + "MACRO REGIME DURATION PROJECT")
    print(" "*15 + "Complete Analysis Pipeline Execution")
    print("="*80)

    # ------------------------------------------------------------
    # STEP 1: DATA COLLECTION
    # ------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 1/4: DATA COLLECTION")
    print("─"*80)

    try:
        collector = DataCollector()
        collector.save_all_data(output_dir=config.PATHS['data_raw'])
        print("✓ Data collection successful")
    except Exception as e:
        print(f"✗ Data collection failed: {e}")
        return False

    # ------------------------------------------------------------
    # STEP 2: REGIME IDENTIFICATION
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # STEP 3: YIELD CURVE MODELING
    # ------------------------------------------------------------
    print("\n" + "─"*80)
    print("STEP 3/4: YIELD CURVE MODELING")
    print("─"*80)

    try:
        # Load processed inputs
        yields = pd.read_csv(
            f"{config.PATHS['data_raw']}/treasury_yields.csv", index_col=0, parse_dates=True
        )
        regimes = pd.read_csv(
            f"{config.PATHS['data_processed']}/regime_probabilities.csv", index_col=0, parse_dates=True
        )

        yc_model = YieldCurveModel(yield_data=yields, regime_probs=regimes)
        yc_model.extract_ns_factors()
        yc_model.estimate_var(lags=config.YIELD_CURVE_CONFIG['var_lags'])
        yc_model.forecast_next(steps=config.YIELD_CURVE_CONFIG['forecast_horizon'])
        print("✓ Yield curve modeling successful")

    except Exception as e:
        print(f"✗ Yield curve modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ------------------------------------------------------------
    # STEP 4: PORTFOLIO OPTIMIZATION
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)

    print("\nResults saved to:")
    print(f"  📊 Figures: {config.PATHS['output_figures']}/")
    print(f"  📁 Data: {config.PATHS['data_processed']}/")

    return True


# =====================================================================
# INDIVIDUAL STEP RUNNERS
# =====================================================================
def run_data_only():
    print("\nRunning data collection only...")
    collector = DataCollector()
    collector.save_all_data(output_dir=config.PATHS['data_raw'])
    print("✓ Data collection complete")


def run_regime_only():
    print("\nRunning regime modeling only...")
    regime_model = RegimeModel(n_regimes=config.REGIME_CONFIG['n_regimes'])
    regime_model.prepare_data(data_path=f"{config.PATHS['data_raw']}/macro_data.csv")
    regime_model.estimate_model(
        dependent_var=config.REGIME_CONFIG['dependent_var'],
        exog_vars=config.REGIME_CONFIG['exog_vars']
    )
    regime_model.print_summary()
    regime_model.save_results(output_dir=config.PATHS['data_processed'])
    print("✓ Regime modeling complete")


def run_yield_curve_only():
    print("\nRunning yield curve modeling only...")
    yields = pd.read_csv("data/raw/treasury_yields.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/processed/regime_probabilities.csv", index_col=0, parse_dates=True)
    yc_model = YieldCurveModel(yield_data=yields, regime_probs=regimes)
    yc_model.extract_ns_factors()
    yc_model.estimate_var()
    yc_model.forecast_factors()
    print("✓ Yield curve modeling complete")


def run_portfolio_only():
    print("\nRunning portfolio optimization only...")
    optimizer = PortfolioOptimizer()
    optimizer.load_data(
        returns_path=f"{config.PATHS['data_raw']}/treasury_returns.csv",
        regime_path=f"{config.PATHS['data_processed']}/regime_probabilities.csv"
    )
    optimizer.backtest()
    optimizer.plot_performance()
    optimizer.save_results()
    print("✓ Portfolio optimization complete")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Macro Regime Duration Project - Analysis Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "data", "regime", "yield_curve", "portfolio"],
        help="Which step to run (default: all)"
    )

    args = parser.parse_args()

    if args.step == "all":
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
    elif args.step == "data":
        run_data_only()
    elif args.step == "regime":
        run_regime_only()
    elif args.step == "yield_curve":
        run_yield_curve_only()
    elif args.step == "portfolio":
        run_portfolio_only()
