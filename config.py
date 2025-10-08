"""
Configuration file for Macro Regime Duration Project
Adjust parameters here to customize the analysis
"""

# Data Collection Parameters
DATA_CONFIG = {
    'start_date': '1990-01-01',
    'fred_api_key': None,  # Will be loaded from .env file
}

# Regime Model Parameters
REGIME_CONFIG = {
    'n_regimes': 3,  # Number of regimes (2 or 3 recommended)
    'dependent_var': 'gdp_growth',        # Main variable for regime-switching
    'exog_vars': ['inflation', 'unemployment'],  # Additional variables
    'switching_variance': True,           # Allow variance to differ by regime
}

# Yield Curve Model Parameters
YIELD_CURVE_CONFIG = {
    'var_lags': 2,                  # Number of lags in VAR model
    'forecast_horizon': 6,          # Months ahead to forecast
    'lambda_init': 0.0609,          # Initial Nelson-Siegel decay parameter
    'maturities': ['2Y', '5Y', '10Y', '30Y'],  # Key maturities to analyze
}

# Portfolio Optimization Parameters
PORTFOLIO_CONFIG = {
    'universe': ['2Y', '5Y', '10Y', '30Y'],  # Investable universe
    'durations': {
        '2Y': 1.9,
        '5Y': 4.5,
        '10Y': 8.5,
        '30Y': 18.0
    },
    'target_duration': 6.0,           # Base duration target
    'duration_tolerance': 2.0,        # +/- tolerance
    'max_weight': 0.5,                # Maximum position size
    'risk_aversion': 2.0,             # Risk aversion parameter (lambda)
    'rebalance_freq': 1,              # Rebalance frequency in months
    'lookback_window': 120,           # Historical window for statistics (months) (unused in NumPy version)
}

# Backtest Parameters
BACKTEST_CONFIG = {
    'start_date': '2005-01-01',  # When to start backtest (unused in NumPy version)
    'transaction_costs': 0.0,    # Per-trade cost (as % of position)
}

# Output Paths
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'output_figures': 'output/figures',
    'output_tables': 'output/tables',
    'output_results': 'output/results',
    'paper': 'paper',
}

# Files generated/consumed across steps
VAR_FORECAST_FILE = "var_forecast.csv"        # produced by YieldCurveModel.forecast_next
NEXT_ALLOC_FILE = "next_allocation.csv"       # produced by PortfolioOptimizer.suggest_next_allocation_from_var

# Visualization Settings
VIZ_CONFIG = {
    'figure_size': (14, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
}

# Paper/Report Settings
PAPER_CONFIG = {
    'title': 'Macro Regimes and Treasury Curve Positioning: A Quantitative Framework',
    'author': 'Matteo Craviotto',
    'date': 'October 2025',
    'format': 'pdf',  # 'pdf' or 'docx'
}
