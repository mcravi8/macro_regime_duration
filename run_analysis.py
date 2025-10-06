"""End-to-end pipeline runner."""
from dotenv import load_dotenv
from config import DATA_PROC, OUTPUT_FIGS, OUTPUT_TABLES, OUTPUT_RESULTS
from src.data_collection.collect_data import fetch_and_save_all
from src.regime_model.regime_model import estimate_regimes
from src.yield_curve.yield_curve_model import fit_ns_and_forecast
from src.portfolio_opt.portfolio_optimization import backtest_strategy
from src.utils.plotting import plot_regime_probs, plot_ns_factors, plot_performance
from src.utils.metrics import summarize_performance

def main():
    load_dotenv()
    fetch_and_save_all()
    regime_probs, regime_labels, regime_stats = estimate_regimes()
    ns_factors, expected_returns = fit_ns_and_forecast(regime_probs)
    backtest_df, weights_df = backtest_strategy(expected_returns)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    regime_probs.to_csv(DATA_PROC / "regime_probabilities.csv")
    regime_labels.to_csv(DATA_PROC / "regime_labels.csv", index=False)
    regime_stats.to_csv(OUTPUT_TABLES / "regime_statistics.csv", index=False)
    ns_factors.to_csv(DATA_PROC / "ns_factors.csv")
    expected_returns.to_csv(DATA_PROC / "expected_returns.csv")
    backtest_df.to_csv(DATA_PROC / "backtest_results.csv")
    weights_df.to_csv(DATA_PROC / "portfolio_weights.csv")

    plot_regime_probs(regime_probs, save_path=OUTPUT_FIGS / "regime_probabilities.png")
    plot_ns_factors(ns_factors, save_path=OUTPUT_FIGS / "ns_factors.png")
    plot_performance(backtest_df, save_path=OUTPUT_FIGS / "portfolio_performance.png")

    backtest_df.dropna().to_csv(OUTPUT_TABLES / "performance_metrics.csv")
    summary = summarize_performance(backtest_df)
    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS / "model_summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()