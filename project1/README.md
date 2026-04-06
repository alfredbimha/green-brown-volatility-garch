# Volatility Modeling of Green vs. Brown Stocks (GARCH Analysis)

## Research Question

**Do clean energy stocks exhibit different volatility dynamics compared to fossil fuel stocks?**

Understanding how volatility behaves differently across "green" and "brown" assets is critical for portfolio risk management during the energy transition. This project applies GARCH-family models to compare volatility clustering, persistence, and asymmetric (leverage) effects.

---

## Data

| Ticker | Asset | Description |
|--------|-------|-------------|
| ICLN | Clean Energy (Green) | iShares Global Clean Energy ETF |
| XLE | Fossil Fuel (Brown) | Energy Select Sector SPDR Fund |
| SPY | Benchmark | S&P 500 ETF |

- **Source:** Yahoo Finance (free, via `yfinance`)
- **Period:** January 2015 – December 2025
- **Frequency:** Daily closing prices

---

## Methodology

### Models Estimated

| Model | What It Captures |
|-------|-----------------|
| **GARCH(1,1)** | Standard volatility clustering |
| **EGARCH(1,1)** | Asymmetric effects (log-scale, no positivity constraint) |
| **GJR-GARCH(1,1)** | Asymmetric effects via threshold indicator |

All models use a **Student's t-distribution** for residuals to account for fat tails commonly observed in financial returns.

### Pre-Estimation Tests

- **Jarque-Bera test** — confirms non-normality of returns (fat tails)
- **Ljung-Box test on squared returns** — confirms volatility clustering (ARCH effects)

---

## Key Findings

1. **Volatility Persistence:** All three assets show high persistence (α + β close to 1), meaning volatility shocks dissipate slowly.

2. **Asymmetric Leverage Effects:** The gamma parameter in EGARCH and GJR-GARCH models reveals whether negative shocks ("bad news") increase volatility more than positive shocks — and this effect differs across green and brown stocks.

3. **COVID-19 Impact:** All assets experienced a massive volatility spike in March 2020, but the recovery pattern differs across sectors.

4. **Risk Implications:** Clean energy and fossil fuel stocks show distinct risk profiles, with implications for portfolio construction during the energy transition.

---

## Output

### Figures

| Figure | Description |
|--------|-------------|
| `fig1_prices_and_returns.png` | Normalized price performance and daily returns |
| `fig2_return_distributions.png` | Return distributions vs. normal distribution |
| `fig3_conditional_volatility.png` | Conditional volatility (best model per asset) |
| `fig4_volatility_comparison.png` | Smoothed volatility overlay comparison |

### Tables

| Table | Description |
|-------|-------------|
| `summary_statistics.csv` | Descriptive statistics of daily returns |
| `model_comparison.csv` | Parameters, AIC/BIC, persistence for all 9 models |

---

## How to Run

### Requirements

```
pip install -r requirements.txt
```

### Run the Analysis

```
python code/project1_garch_volatility.py
```

This will download the data, run all models, and save figures and tables to the `output/` folder.

---

## Tools Used

- **Python 3.10+**
- `yfinance` — data download
- `arch` — GARCH model estimation
- `statsmodels` — statistical tests
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scipy` — distribution tests

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── code/
│   └── project1_garch_volatility.py
├── data/
│   ├── daily_prices.csv
│   └── daily_returns.csv
└── output/
    ├── figures/
    │   ├── fig1_prices_and_returns.png
    │   ├── fig2_return_distributions.png
    │   ├── fig3_conditional_volatility.png
    │   └── fig4_volatility_comparison.png
    └── tables/
        ├── summary_statistics.csv
        └── model_comparison.csv
```

---

## Author

Alfred Bimha

## License

MIT
