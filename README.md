# 📈 Macro Regime Duration Model

A Python research project modeling **macroeconomic regimes** (Recession / Moderate Growth / Expansion) using **Markov-switching regression**, and extending to **yield-curve dynamics** and **portfolio optimization**.

---

## 🔍 Overview

This project analyzes how macroeconomic conditions evolve over time by identifying latent regimes in GDP growth, inflation, and unemployment.

It builds a **3-state Markov-switching model** to:
- Detect hidden business-cycle regimes  
- Estimate transition probabilities and expected regime durations  
- Integrate regime information into yield-curve and portfolio models

---

## 🧠 Concepts Used

| Domain | Core Idea |
|--------|------------|
| **Time Series** | Markov-switching regression on standardized macro variables |
| **Statistics** | Hidden-state inference via smoothed marginal probabilities |
| **Econometrics** | Transition-matrix estimation, expected regime duration |
| **Macro Finance** | Nelson–Siegel yield-curve factor extraction |
| **Optimization** | Mean–variance portfolio backtesting under dynamic regimes |

---

## ⚙️ Model Pipeline

1. **Data Preparation**  
   Loads and standardizes macro data from `data/raw/macro_data.csv`.

2. **Regime Identification**  
   Fits a MarkovRegression model with 3 regimes (Recession, Moderate Growth, Expansion).  
   Generates smoothed probabilities and regime labels.

3. **Yield-Curve Modeling**  
   Applies the Nelson–Siegel factor model to Treasury maturities  
   and links yield dynamics to detected regimes.

4. **Portfolio Optimization**  
   Constructs and backtests a dynamic bond-equity portfolio adapting to regimes.

---

## 📊 Key Outputs

### 1️⃣ Regime Probabilities
![Regime Probabilities](output/figures/regime_probabilities.png)

### 2️⃣ Regime Panels
![Regime Panels](output/figures/regime_panels.png)

### 3️⃣ Dominant Regime Scatter
![Regime Scatter](output/figures/regime_scatter.png)

---

## 📈 Results & Implications

- **Recession** phases (~13 months average) show lower mean growth and higher variance.  
- **Moderate Growth** dominates most of the sample, capturing stable periods.  
- **Expansion** episodes are short, explosive upswings in GDP growth.  
- The transition matrix shows **high regime persistence**, validating macro-cycle structure.  
- Integrating these probabilities improves risk-adjusted portfolio allocation.

---

## 🧩 Tools & Libraries

- Python 3.11  
- `statsmodels`, `pandas`, `numpy`, `matplotlib`  
- Optional: `cvxpy`, `scikit-learn`, `plotly` (for future dashboard extension)

---

## 🚀 Usage

```bash
# Run regime model only
python run_analysis.py --step=regime

# Run full pipeline (regime + yield curve + portfolio)
python run_analysis.py --step=all
