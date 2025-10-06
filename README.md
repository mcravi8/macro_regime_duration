# Macro Regimes and Treasury Curve Positioning

**Purpose:** Recruiter-ready research project demonstrating fixed income understanding (PIMCO-aligned):  
1) Macro regime identification (Markov-switching)  
2) Yield curve dynamics & forecasting (Nelson–Siegel + VAR)  
3) Tactical duration positioning (constrained optimizer & walk-forward backtest)

**How to run (quick):**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env  # and set FRED_API_KEY
python build_project.py           # (recreates folders if needed)
python run_analysis.py            # end-to-end pipeline (~15-20 min when data present)
```

**Artifacts produced:**
- `data/processed/*.csv`: regimes, NS factors, expected returns, backtest results, weights
- `output/figures/*.png`: regimes, NS factors, performance
- `output/tables/*.csv`: regime stats, VAR results, performance metrics
- `output/results/model_summary.txt`: summary for paper & deck

**Notes:**
- Data are intentionally **not** committed; they are regenerated from code.
- See `paper/research_paper_outline.md` and `presentation/presentation_outline.md` for write-up & deck structure.
- This repo is structured for *clarity first*, then performance.

_Last generated: 2025-10-04_