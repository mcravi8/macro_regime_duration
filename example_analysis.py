"""Minimal example usage after artifacts exist."""
import pandas as pd
from config import DATA_PROC

def main():
    try:
        regimes = pd.read_csv(DATA_PROC / "regime_probabilities.csv", index_col=0, parse_dates=True)
        print(regimes.tail())
    except FileNotFoundError:
        print("Run `python run_analysis.py` first to generate processed data.")

if __name__ == "__main__":
    main()