"""Creates the folder structure and placeholder files."""
from pathlib import Path
from config import (
    DATA_RAW, DATA_PROC, OUTPUT_FIGS, OUTPUT_TABLES, OUTPUT_RESULTS, PROJECT_ROOT
)

def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).touch(exist_ok=True)

def main():
    for p in [
        DATA_RAW, DATA_PROC, OUTPUT_FIGS, OUTPUT_TABLES, OUTPUT_RESULTS,
        PROJECT_ROOT / "src" / "data_collection",
        PROJECT_ROOT / "src" / "regime_model",
        PROJECT_ROOT / "src" / "yield_curve",
        PROJECT_ROOT / "src" / "portfolio_opt",
        PROJECT_ROOT / "src" / "utils",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "output" / "figures",
        PROJECT_ROOT / "output" / "tables",
        PROJECT_ROOT / "output" / "results",
        PROJECT_ROOT / "paper" / "figures",
        PROJECT_ROOT / "presentation",
        PROJECT_ROOT / "tests",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    for p in [
        PROJECT_ROOT / "data" / "raw" / ".gitkeep",
        PROJECT_ROOT / "data" / "processed" / ".gitkeep",
        PROJECT_ROOT / "output" / "figures" / ".gitkeep",
        PROJECT_ROOT / "output" / "tables" / ".gitkeep",
        PROJECT_ROOT / "output" / "results" / ".gitkeep",
    ]:
        touch(p)

    for name in ["macro_data.csv","treasury_yields.csv","treasury_returns.csv","credit_spreads.csv"]:
        touch(DATA_RAW / name)

    touch(PROJECT_ROOT / "paper" / "final_paper.pdf")
    touch(PROJECT_ROOT / "presentation" / "slides.pptx")
    touch(PROJECT_ROOT / "presentation" / "slides.pdf")
    print("Project structure created.")

if __name__ == "__main__":
    main()