import pandas as pd
import requests
from dotenv import load_dotenv
from config import DATA_RAW, FRED_API_KEY

FRED_SERIES = {
    "GDP": "A191RL1Q225SBEA",
    "UNRATE": "UNRATE",
    "CPI": "CPIAUCSL",
    "ISM": "NAPM",
    "DGS2": "DGS2",
    "DGS5": "DGS5",
    "DGS10": "DGS10",
    "DGS30": "DGS30",
    "BAMLIG": "BAMLC0A0CM",
    "BAMLHY": "BAMLH0A0HYM2",
}

def _fred(series_id, api_key):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = dict(series_id=series_id, api_key=api_key, file_type="json", observation_start="1985-01-01")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["observations"])[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")

def fetch_and_save_all():
    load_dotenv()
    if not FRED_API_KEY:
        print("WARNING: FRED_API_KEY not set; skipping real fetch. Placeholders remain.")
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        return

    # Macro
    macro_parts = []
    for k in ["GDP","UNRATE","CPI","ISM"]:
        df = _fred(FRED_SERIES[k], FRED_API_KEY).rename(columns={"value": k})
        macro_parts.append(df)
    macro = pd.concat(macro_parts, axis=1).resample("M").last()
    macro["GDP"] = macro["GDP"].pct_change(4)
    macro["CPI_INFL"] = macro["CPI"].pct_change(12)
    macro = macro.drop(columns=["CPI"])
    macro.to_csv(DATA_RAW / "macro_data.csv")

    # Yields
    ylds = []
    for k in ["DGS2","DGS5","DGS10","DGS30"]:
        df = _fred(FRED_SERIES[k], FRED_API_KEY).rename(columns={"value": k})
        ylds.append(df)
    yields_df = pd.concat(ylds, axis=1).resample("M").last() / 100.0
    yields_df.to_csv(DATA_RAW / "treasury_yields.csv")

    # Credit (optional)
    try:
        ig = _fred(FRED_SERIES["BAMLIG"], FRED_API_KEY).rename(columns={"value": "IG"})
        hy = _fred(FRED_SERIES["BAMLHY"], FRED_API_KEY).rename(columns={"value": "HY"})
        spreads = pd.concat([ig, hy], axis=1).resample("M").last() / 100.0
        spreads.to_csv(DATA_RAW / "credit_spreads.csv")
    except Exception as e:
        print("Credit spreads fetch failed or requires permission; skipping.", e)

    (DATA_RAW / "treasury_returns.csv").touch()
    print("Data saved to data/raw/.")