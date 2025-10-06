from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGS = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "output" / "tables"
OUTPUT_RESULTS = PROJECT_ROOT / "output" / "results"

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

MARKOV_STATES = 3
NS_TAU = 30.0
VAR_LAGS = 2

REBAL_FREQ = "M"
TARGET_DURATION_RANGE = (5.0, 8.0)
MAX_WEIGHT = 0.40
TURNOVER_PENALTY = 0.001

UNIVERSE = {
    "UST_2Y": 2.0,
    "UST_5Y": 5.0,
    "UST_10Y": 10.0,
    "UST_30Y": 20.0,
}

BENCHMARK = "Duration_Matched"