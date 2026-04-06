"""
===============================================================================
PROJECT 1: Volatility Modeling of Green vs. Brown Stocks (GARCH Analysis)
===============================================================================

RESEARCH QUESTION:
    Do clean energy stocks exhibit different volatility dynamics 
    compared to fossil fuel stocks?

METHOD:
    Fit GARCH(1,1), EGARCH, and GJR-GARCH models to daily returns of:
    - ICLN (iShares Global Clean Energy ETF) — "green" stocks
    - XLE  (Energy Select Sector SPDR)       — "brown" / fossil fuel stocks
    - SPY  (S&P 500 ETF)                     — market benchmark
    
    Compare volatility clustering, persistence, and leverage effects.

DATA SOURCE:
    Yahoo Finance (free, via the yfinance library)

AUTHOR:  [Your Name]
DATE:    2026
===============================================================================
"""

# =============================================================================
# STEP 0: IMPORT LIBRARIES
# =============================================================================
# These are the Python packages we need. If any are missing, run:
#   pip install yfinance arch statsmodels pandas matplotlib seaborn scipy

import yfinance as yf          # Download stock price data from Yahoo Finance
import pandas as pd            # Data manipulation (tables, time series)
import numpy as np             # Numerical operations (math, arrays)
import matplotlib.pyplot as plt # Create charts and plots
import matplotlib.dates as mdates
import seaborn as sns          # Prettier statistical plots
from arch import arch_model    # GARCH family models
from scipy import stats        # Statistical tests
import warnings
import os

warnings.filterwarnings('ignore')  # Suppress routine warnings for cleaner output

# Set visual style for all plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)  # Default chart size
plt.rcParams['figure.dpi'] = 150          # High resolution

# Create output directories if they don't exist
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/tables', exist_ok=True)
os.makedirs('data', exist_ok=True)


# =============================================================================
# STEP 1: DOWNLOAD STOCK PRICE DATA
# =============================================================================
print("=" * 70)
print("STEP 1: Downloading stock price data from Yahoo Finance...")
print("=" * 70)

# Define the tickers (stock symbols) we want to study
tickers = {
    'ICLN': 'Clean Energy (Green)',   # iShares Global Clean Energy ETF
    'XLE':  'Fossil Fuel (Brown)',    # Energy Select Sector SPDR Fund
    'SPY':  'S&P 500 (Benchmark)'    # Market benchmark
}

# Date range for our analysis
start_date = '2015-01-01'
end_date   = '2025-12-31'

# Download each ticker individually, then combine
# (This avoids multi-index column issues with yfinance)
price_data = {}
for ticker in tickers.keys():
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    price_data[ticker] = df['Close']

# Combine into a single DataFrame
prices = pd.concat(price_data, axis=1)
prices.columns = list(tickers.keys())  # Clean column names

# Drop any rows with missing data
prices = prices.dropna()

print(f"\n  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Trading days: {len(prices)}")
print(f"\n  First few rows:")
print(prices.head())

# Save raw prices to CSV (for reproducibility)
prices.to_csv('data/daily_prices.csv')
print("\n  Saved: data/daily_prices.csv")


# =============================================================================
# STEP 2: CALCULATE DAILY RETURNS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Calculating daily log returns...")
print("=" * 70)

# Log returns: ln(P_t / P_{t-1})
# Why log returns? They are additive over time and approximately normal,
# which is a standard assumption in financial econometrics.
returns = np.log(prices / prices.shift(1)).dropna() * 100  # Multiply by 100 for percentage

print("\n  Summary Statistics of Daily Returns (%):")
summary = returns.describe().T
summary['skewness'] = returns.skew()
summary['kurtosis'] = returns.kurtosis()  # Excess kurtosis (normal = 0)
print(summary[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].round(4))

# Save returns
returns.to_csv('data/daily_returns.csv')

# Save summary statistics as a formatted table
summary_table = summary[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].round(4)
summary_table.index = [tickers[t] for t in summary_table.index]
summary_table.to_csv('output/tables/summary_statistics.csv')
print("\n  Saved: output/tables/summary_statistics.csv")


# =============================================================================
# STEP 3: VISUALIZE THE DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating visualizations...")
print("=" * 70)

# --- Figure 1: Price Levels (Normalized to 100) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Normalize prices so they all start at 100 (makes comparison easier)
normalized = (prices / prices.iloc[0]) * 100

colors = {'ICLN': '#2ecc71', 'XLE': '#e74c3c', 'SPY': '#3498db'}  # green, red, blue

for ticker, label in tickers.items():
    axes[0].plot(normalized.index, normalized[ticker], 
                 label=label, color=colors[ticker], linewidth=1.2)

axes[0].set_title('Normalized Price Performance (Base = 100)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Normalized Price')
axes[0].legend(fontsize=11)
axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# --- Figure 2: Daily Returns ---
for ticker, label in tickers.items():
    axes[1].plot(returns.index, returns[ticker], 
                 label=label, color=colors[ticker], alpha=0.6, linewidth=0.5)

axes[1].set_title('Daily Log Returns (%)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Return (%)')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('output/figures/fig1_prices_and_returns.png', bbox_inches='tight')
plt.close()
print("  Saved: output/figures/fig1_prices_and_returns.png")

# --- Figure 2: Return Distributions ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (ticker, label) in enumerate(tickers.items()):
    axes[i].hist(returns[ticker], bins=80, density=True, alpha=0.7, 
                 color=colors[ticker], edgecolor='white')
    
    # Overlay a normal distribution for comparison
    x = np.linspace(returns[ticker].min(), returns[ticker].max(), 100)
    axes[i].plot(x, stats.norm.pdf(x, returns[ticker].mean(), returns[ticker].std()),
                 'k--', linewidth=2, label='Normal dist.')
    
    axes[i].set_title(f'{label}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Daily Return (%)')
    axes[i].legend()

fig.suptitle('Return Distributions vs Normal Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/figures/fig2_return_distributions.png', bbox_inches='tight')
plt.close()
print("  Saved: output/figures/fig2_return_distributions.png")


# =============================================================================
# STEP 4: STATISTICAL TESTS (Pre-GARCH Checks)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Running pre-estimation tests...")
print("=" * 70)

# Jarque-Bera test: Are returns normally distributed?
# If p-value < 0.05, we reject normality → justifies GARCH modeling
print("\n  Jarque-Bera Normality Test:")
print("  (If p < 0.05, returns are NOT normal → fat tails exist)")
print("  " + "-" * 50)
for ticker, label in tickers.items():
    jb_stat, jb_p = stats.jarque_bera(returns[ticker])
    result = "REJECT normality" if jb_p < 0.05 else "Cannot reject normality"
    print(f"  {label:30s}  JB={jb_stat:10.2f}  p={jb_p:.6f}  → {result}")

# Ljung-Box test on squared returns: Is there volatility clustering?
# Squared returns that are autocorrelated = volatility clustering = ARCH effects
from statsmodels.stats.diagnostic import acorr_ljungbox

print("\n  Ljung-Box Test on Squared Returns (10 lags):")
print("  (If p < 0.05, volatility clustering exists → GARCH is appropriate)")
print("  " + "-" * 50)
for ticker, label in tickers.items():
    lb_result = acorr_ljungbox(returns[ticker]**2, lags=10, return_df=True)
    lb_stat = lb_result['lb_stat'].iloc[-1]
    lb_p = lb_result['lb_pvalue'].iloc[-1]
    result = "ARCH effects present" if lb_p < 0.05 else "No ARCH effects"
    print(f"  {label:30s}  LB={lb_stat:10.2f}  p={lb_p:.6f}  → {result}")


# =============================================================================
# STEP 5: FIT GARCH MODELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Fitting GARCH models...")
print("=" * 70)

# We fit three types of GARCH models:
#
# 1. GARCH(1,1):     Standard model — captures volatility clustering
#    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
#
# 2. EGARCH(1,1):    Captures ASYMMETRIC effects (bad news ≠ good news impact)
#    log(σ²_t) = ω + α·|z_{t-1}| + γ·z_{t-1} + β·log(σ²_{t-1})
#
# 3. GJR-GARCH(1,1): Also captures asymmetry via a "threshold" term
#    σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
#    where I_{t-1} = 1 if ε_{t-1} < 0 (bad news indicator)

# Dictionary to store all results
all_results = {}

model_configs = {
    'GARCH(1,1)':     {'vol': 'GARCH',  'p': 1, 'q': 1, 'o': 0},
    'EGARCH(1,1)':    {'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 1},
    'GJR-GARCH(1,1)': {'vol': 'GARCH',  'p': 1, 'q': 1, 'o': 1},  # o=1 adds the threshold
}

for ticker, label in tickers.items():
    print(f"\n  --- {label} ({ticker}) ---")
    all_results[ticker] = {}
    
    for model_name, config in model_configs.items():
        # Create the model
        am = arch_model(
            returns[ticker],
            mean='Constant',        # Assume returns have a constant mean
            vol=config['vol'],      # Volatility model type
            p=config['p'],          # Lag of squared returns
            q=config['q'],          # Lag of conditional variance
            o=config['o'],          # Asymmetry / threshold order
            dist='t'                # Student's t distribution (handles fat tails)
        )
        
        # Fit the model (find the best parameters)
        result = am.fit(disp='off')  # disp='off' hides optimization output
        
        # Store results
        all_results[ticker][model_name] = result
        
        # Print key metrics
        persistence = result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0)
        print(f"  {model_name:20s}  AIC={result.aic:10.2f}  "
              f"BIC={result.bic:10.2f}  "
              f"Persistence={persistence:.4f}")


# =============================================================================
# STEP 6: EXTRACT AND COMPARE MODEL PARAMETERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Comparing model parameters...")
print("=" * 70)

# Build a comprehensive comparison table
rows = []
for ticker, label in tickers.items():
    for model_name, result in all_results[ticker].items():
        row = {
            'Asset': label,
            'Ticker': ticker,
            'Model': model_name,
            'AIC': round(result.aic, 2),
            'BIC': round(result.bic, 2),
            'Log-Likelihood': round(result.loglikelihood, 2),
            'omega': round(result.params.get('omega', np.nan), 6),
            'alpha': round(result.params.get('alpha[1]', np.nan), 4),
            'beta': round(result.params.get('beta[1]', np.nan), 4),
        }
        
        # Asymmetry parameter (gamma) — only in EGARCH and GJR
        if 'gamma[1]' in result.params:
            row['gamma (asymmetry)'] = round(result.params['gamma[1]'], 4)
        else:
            row['gamma (asymmetry)'] = np.nan
        
        # Persistence: how long volatility shocks last
        # Higher persistence = shocks take longer to die out
        row['Persistence (α+β)'] = round(
            result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0), 4
        )
        
        rows.append(row)

comparison = pd.DataFrame(rows)
comparison.to_csv('output/tables/model_comparison.csv', index=False)
print("\n  Full Model Comparison Table:")
print(comparison.to_string(index=False))
print("\n  Saved: output/tables/model_comparison.csv")


# =============================================================================
# STEP 7: VISUALIZE CONDITIONAL VOLATILITY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Plotting conditional volatility...")
print("=" * 70)

# Use the best model (by AIC) for each ticker to plot estimated volatility over time
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for i, (ticker, label) in enumerate(tickers.items()):
    # Find best model by AIC
    best_model = min(all_results[ticker].items(), key=lambda x: x[1].aic)
    model_name, result = best_model
    
    # Conditional volatility = square root of conditional variance
    cond_vol = result.conditional_volatility
    
    axes[i].plot(cond_vol.index, cond_vol, color=colors[ticker], linewidth=0.8)
    axes[i].fill_between(cond_vol.index, 0, cond_vol, alpha=0.3, color=colors[ticker])
    axes[i].set_title(f'{label} — Conditional Volatility ({model_name})', 
                       fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Volatility (%)')

    # Mark notable events
    axes[i].axvline(pd.Timestamp('2020-03-11'), color='black', linestyle='--', 
                     alpha=0.5, linewidth=1)
    if i == 0:
        axes[i].text(pd.Timestamp('2020-03-15'), axes[i].get_ylim()[1]*0.85, 
                      'COVID-19', fontsize=9, alpha=0.7)

axes[2].set_xlabel('Date')
plt.suptitle('Conditional Volatility: Green vs Brown vs Market', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig3_conditional_volatility.png', bbox_inches='tight')
plt.close()
print("  Saved: output/figures/fig3_conditional_volatility.png")


# --- Overlay comparison chart ---
fig, ax = plt.subplots(figsize=(14, 6))

for ticker, label in tickers.items():
    best_model = min(all_results[ticker].items(), key=lambda x: x[1].aic)
    cond_vol = best_model[1].conditional_volatility
    # Use 20-day rolling average for smoother visual
    smoothed = cond_vol.rolling(20).mean()
    ax.plot(smoothed.index, smoothed, label=label, color=colors[ticker], linewidth=1.5)

ax.set_title('Conditional Volatility Comparison (20-day smoothed)', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('Volatility (%)')
ax.set_xlabel('Date')
ax.legend(fontsize=12)
ax.axvline(pd.Timestamp('2020-03-11'), color='black', linestyle='--', alpha=0.5)
ax.text(pd.Timestamp('2020-04-01'), ax.get_ylim()[1]*0.85, 'COVID-19', fontsize=10, alpha=0.7)
plt.tight_layout()
plt.savefig('output/figures/fig4_volatility_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: output/figures/fig4_volatility_comparison.png")


# =============================================================================
# STEP 8: KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: KEY FINDINGS")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        KEY FINDINGS                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. VOLATILITY PERSISTENCE                                           ║
║     All three assets show high persistence (α + β close to 1),       ║
║     meaning volatility shocks take a long time to dissipate.         ║
║                                                                      ║
║  2. ASYMMETRIC EFFECTS (Leverage)                                    ║
║     The gamma parameter in EGARCH/GJR models captures whether        ║
║     bad news increases volatility more than good news.               ║
║     Compare gamma across green vs brown stocks.                      ║
║                                                                      ║
║  3. RISK PROFILE DIFFERENCES                                        ║
║     Clean energy (ICLN) may show different volatility dynamics       ║
║     than fossil fuels (XLE), with implications for portfolio         ║
║     risk management during the energy transition.                    ║
║                                                                      ║
║  4. COVID-19 IMPACT                                                  ║
║     All assets experienced a massive volatility spike in             ║
║     March 2020, but recovery patterns differ across sectors.         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Print the actual numbers for easy reference
print("\n  Volatility Persistence by Asset (Best Model):")
print("  " + "-" * 50)
for ticker, label in tickers.items():
    best_model = min(all_results[ticker].items(), key=lambda x: x[1].aic)
    model_name, result = best_model
    persistence = result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0)
    print(f"  {label:30s}  {model_name:20s}  Persistence: {persistence:.4f}")

print("\n  Asymmetry (gamma) by Asset:")
print("  " + "-" * 50)
for ticker, label in tickers.items():
    for model_name in ['EGARCH(1,1)', 'GJR-GARCH(1,1)']:
        result = all_results[ticker][model_name]
        gamma = result.params.get('gamma[1]', np.nan)
        print(f"  {label:30s}  {model_name:20s}  γ = {gamma:.4f}")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE! Check the output/ folder for all figures and tables.")
print("=" * 70)
