"""
==============================================================================
  S&P 500 Sector-Based Pairs Trading Strategy
  Seminar: Econometric Models in Financial Applications (F) — 2025/2026
  By Viktor Christian Lohmann Laursen (KU-ID: QNW176)
==============================================================================

STRATEGY OVERVIEW
-----------------
1.  Formation period : 3 years (756 trading days) — estimate VECM via MLE
2.  Trading period   : 6 months (126 trading days) — execute on valid pairs
3.  Rolling window   : slide forward 6 months, repeat from 1998 to present
4.  Pair selection   : λ₁ > 0  AND  λ₂ > 0  (both participate in error correction)
5.  Pairs formed     : within GICS sectors only (economic justification)
6.  Position sizing  : Black-Scholes spread-option deltas (using ATM IV)
7.  Entry trigger    : |z_t| > (1 − λ₁ − λ₂) · σ_spread
8.  Exit trigger     : sign-change of  (Γ_x² / Γ_y²) − (P_y / P_x)
9.  Returns          : equal-weighted across all active pairs

DATA SOURCES
------------
- Prices          : Yahoo Finance via yfinance
- Implied Vol     : ATM IV from yfinance options chain (30-day rolling
                    realised vol used as fallback for historical periods)
- Risk-free rate  : FRED DTB6 (6-month T-bill, discount basis) via
                    pandas_datareader.  Falls back to ^IRX from Yahoo Finance.

DEPENDENCIES
------------
    pip install yfinance pandas numpy scipy statsmodels joblib pandas-datareader matplotlib seaborn

NOTE ON IMPLIED VOLATILITY
--------------------------
Yahoo Finance only exposes *current* options chains (not historical).
For historical periods (the 2000-present backtest) this script computes a
21-trading-day rolling annualised realised volatility as a proxy for the
ATM implied volatility.  This is standard practice in academic replications
when a proprietary options database (e.g. FactSet, OptionMetrics) is
unavailable.  When running in "live" mode the script fetches the true ATM IV
from the live options chain.
Inspiration from Guerrero Gallego, 2023
"""



# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import warnings
warnings.filterwarnings("ignore")

import itertools
import multiprocessing
import os
import ssl
import requests

# --- Global SSL fix for macOS Wikipedia fetch -------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from datetime import datetime, timedelta
from scipy.stats import norm, skew, probplot, ttest_1samp
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from joblib import Parallel, delayed

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance:  pip install yfinance")

try:
    from pandas_datareader import data as pdr
    PANDAS_DATAREADER_OK = True
except ImportError:
    PANDAS_DATAREADER_OK = False
    print("[WARNING] pandas_datareader not found.  "
          "Risk-free rate will be fetched from Yahoo Finance (^IRX).\n"
          "For best results:  pip install pandas-datareader")


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# --- Window sizes -----------------------------------------------------------
YEAR      = 252          # approximate trading days per year
WINDOW    = 3 * YEAR     # formation period length   (756 days)
STEP      = YEAR // 2    # trading / step period     (126 days)

# --- Date configuration -----------------------------------------------------
DATA_START = "1998-01-01"   # start of price data download (trading begins in 2000)
DATA_END   = datetime.today().strftime("%Y-%m-%d")

# --- S&P 500 sector-based stock universe ------------------------------------
# Pairs are only formed WITHIN GICS sectors so that cointegration has an
# economic justification (common factor exposure).
# The full current S&P 500 is fetched dynamically from Wikipedia.
# Fallback: a representative 200-stock subset.

def _fetch_sp500_from_wikipedia() -> dict:
    """
    Fetch current S&P 500 constituents and their GICS sectors from Wikipedia.
    Returns dict  sector_name → list of tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks simple urllib; use requests with a real User-Agent
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/91.0.4472.124 Safari/537.36")
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        # Column names vary; use positional fallback
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        sec_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[3]
        df[sym_col] = df[sym_col].str.replace(".", "-", regex=False)  # BRK.B → BRK-B
        sectors = {}
        for _, row in df.iterrows():
            sec  = row[sec_col]
            tick = row[sym_col]
            sectors.setdefault(sec, []).append(tick)
        print(f"[DATA] Fetched {sum(len(v) for v in sectors.values())} S&P 500 "
              f"tickers across {len(sectors)} GICS sectors from Wikipedia.")
        return sectors
    except Exception as e:
        print(f"[WARNING] Wikipedia fetch failed ({e}).  Using fallback list.")
        return {}

_FALLBACK_SECTORS = {
    "Information Technology": [
        "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "TXN", "QCOM", "IBM", "ADBE", "AVGO",
        "CRM", "AMD", "NOW", "PANW", "AMAT", "MU", "LRCX", "ADI", "KLAC", "SNPS",
    ],
    "Financials": [
        "JPM", "BAC", "GS", "WFC", "MS", "C", "BK", "USB", "PNC", "AIG",
        "V", "MA", "AXP", "BLK", "SCHW", "CB", "MMC", "MET", "PRU", "TRV",
    ],
    "Health Care": [
        "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "MDT", "UNH", "GILD",
        "ABBV", "TMO", "DHR", "ISRG", "CVS", "CI", "ELV", "SYK", "BDX", "HCA",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "CL", "WMT", "COST", "MO", "GIS", "PM", "SJM",
        "MDLZ", "ADM", "STZ", "SYY", "KDP", "KR", "HSY", "KHC", "MKC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO",
        "WMB", "OKE", "HAL", "KMI", "BKR", "CTRA", "MRO", "FANG", "DVN", "APA",
    ],
    "Industrials": [
        "HON", "MMM", "GE", "CAT", "BA", "LMT", "UPS", "RTX", "DE", "EMR",
        "UNP", "ETN", "CSX", "NSC", "ITW", "WM", "FDX", "PH",
    ],
    "Consumer Discretionary": [
        "AMZN", "HD", "MCD", "NKE", "LOW", "TJX", "SBUX", "TGT", "F", "GM",
        "TSLA", "BKNG", "ORLY", "AZO", "MAR", "RCL", "CCL", "DHI", "LEN",
    ],
    "Utilities": [
        "DUK", "SO", "NEE", "D", "AEP", "EXC", "SRE", "XEL", "ES", "WEC",
        "PCG", "PEG", "EIX", "ED", "CNP", "AWK", "NRG", "NI", "LNT",
    ],
    "Materials": [
        "LIN", "APD", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM", "PPG",
        "SHW", "CTVA", "ALB", "CF", "MOS", "FMC", "IFF", "PKG", "WRK", "STLD",
    ],
    "Communication Services": [
        "GOOG", "META", "DIS", "VZ", "T", "CMCSA", "NFLX", "TMUS", "CHTR", "EA",
        "WBD", "PARA", "TTWO", "FOXA", "NWSA", "LYV",
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "WELL", "DLR", "O", "VICI",
    ],
}

# Attempt dynamic fetch; fall back to hardcoded list
SP500_SECTORS = _fetch_sp500_from_wikipedia() or _FALLBACK_SECTORS

# Flat list for downloading
ALL_TICKERS = sorted(set(t for tl in SP500_SECTORS.values() for t in tl))

# Reverse map: ticker → sector
TICKER_TO_SECTOR = {}
for _sec, _tks in SP500_SECTORS.items():
    for _t in _tks:
        TICKER_TO_SECTOR[_t] = _sec

# --- Implied volatility proxy -----------------------------------------------
IV_PROXY_WINDOW = 21     # ~1 calendar month

# --- Transaction costs (basis points per leg, one-way) ----------------------
TC_BPS = 5               # 5 bps per leg (frictionless model)

# --- Signal-strength filtering (trade-limiting) -----------------------------
MAX_NEW_ENTRIES_PER_DAY = 5     # K: max new positions opened per day
MIN_ZSCORE_ENTRY = 1.5          # minimum |z-score| required to open

# --- Correlation pre-screening (Stage 1) ------------------------------------
# Following Gatev et al. (2006), we pre-screen within-sector pairs by
# Pearson correlation of log-returns before running the expensive
# cointegration tests.  Only the top-K most correlated pairs per sector
# (and above a minimum threshold) are passed to Stage 2.
CORR_TOP_K_PER_SECTOR = 30     # keep top-K most correlated pairs per sector
MIN_CORR_THRESHOLD    = 0.50   # minimum Pearson ρ of log-returns to consider

# --- Output directories -----------------------------------------------------
OUTPUT_DIR = "output_9_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Vanilla strategy parameters --------------------------------------------
VANILLA_ENTRY_ZSCORE = 2.0     # open when |z-score| > this threshold
VANILLA_EXIT_ZSCORE  = 0.5     # close when |z-score| < this threshold
VANILLA_POSITION_SIZE = 1.0    # fixed ±1 position (dollar-neutral)


# ===========================================================================
# 2. DATA RETRIEVAL
# ===========================================================================

def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance for all tickers.
    Drops any ticker that has *any* missing value in the full date range
    (i.e. was not listed for the full period — identical to the original's
    survivorship-bias treatment).

    Returns
    -------
    pd.DataFrame  shape (dates × tickers), index = DatetimeIndex
    """
    print(f"\n[DATA] Downloading prices for {len(tickers)} tickers "
          f"({start} → {end}) ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Flatten multi-level columns if only one ticker was requested
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index)

    # Drop tickers with ANY missing value (mirrors original dropna(axis=1))
    raw = raw.dropna(axis=1)
    print(f"[DATA] {raw.shape[1]} tickers retained after dropping "
          f"incomplete series.")
    return raw


def compute_realised_vol(prices: pd.DataFrame, window: int = IV_PROXY_WINDOW
                         ) -> pd.DataFrame:
    """
    Compute rolling annualised realised volatility as a proxy for ATM IV.

    σ_t = std(log returns over last `window` days) × √252

    Returns
    -------
    pd.DataFrame  same shape as `prices`
    """
    log_rets = np.log(prices / prices.shift(1))
    rv = log_rets.rolling(window).std() * np.sqrt(YEAR)
    # Forward-fill the first `window` NaN rows with the first valid value
    rv = rv.ffill().bfill()
    return rv


def download_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Download 6-month T-bill rate (DTB6, discount basis, %) from FRED.
    Falls back to Yahoo Finance ^IRX (13-week T-bill) if pandas_datareader
    is not installed.

    Returns
    -------
    pd.Series  index = DatetimeIndex, values = annualised rate in %
    """
    if PANDAS_DATAREADER_OK:
        try:
            print("[DATA] Fetching DTB6 from FRED ...")
            rates = pdr.DataReader("DTB6", "fred",
                                   start=start, end=end)["DTB6"]
            rates = pd.to_numeric(rates, errors="coerce")
            rates = rates.interpolate().ffill().bfill()
            rates.index = pd.to_datetime(rates.index)
            print(f"[DATA] DTB6: {len(rates)} observations retrieved.")
            return rates
        except Exception as e:
            print(f"[WARNING] FRED download failed ({e}).  "
                  "Falling back to ^IRX from Yahoo Finance.")

    # --- Fallback: Yahoo Finance ^IRX (annualised %) ------------------------
    print("[DATA] Fetching ^IRX from Yahoo Finance ...")
    irx = yf.download("^IRX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    # yfinance may return a DataFrame (multi-index cols); squeeze to Series
    if isinstance(irx, pd.DataFrame):
        irx = irx.squeeze()
    irx = irx.sort_index()
    irx.index = pd.to_datetime(irx.index)
    irx = irx.interpolate().ffill().bfill()
    print(f"[DATA] ^IRX: {len(irx)} observations retrieved.")
    return irx


def align_series(prices: pd.DataFrame,
                 impl_vols: pd.DataFrame,
                 rates: pd.Series) -> tuple:
    """
    Align all three data series to the common date index (inner join).
    """
    common_idx = prices.index.intersection(impl_vols.index).intersection(
        rates.index)
    return (prices.loc[common_idx],
            impl_vols.loc[common_idx],
            rates.loc[common_idx])


# ===========================================================================
# 3. MLE PARAMETER ESTIMATION
# ===========================================================================

def estimate_parameters(y: np.ndarray,
                        x: np.ndarray,
                        Delta_t: float = 1.0) -> tuple:
    """
    Estimate continuous-time VECM parameters via Maximum Likelihood.

    Model
    -----
        dY_t / Y_t = μ_y dt − λ₁ z_t dt + σ_y dW_y
        dX_t / X_t = μ_x dt + λ₂ z_t dt + σ_x dW_x
        z_t        = ln(Y_t) − ln(X_t)

    MLE closed-form solution (Figuerola-Ferretti et al., 2018)
    -----------------------------------------------------------
    λ̂₁ = 2 Σ (A_i − Ā)(z_{i−1} − z̄) / [Δt · Σ (z_{i−1} − z̄)²]
    λ̂₂ = 2 Σ (B_i − B̄)(z_{i−1} − z̄) / [Δt · Σ (z_{i−1} − z̄)²]
    σ̂_y = √[Σ ((A_i − Ā) − λ̂₁ Δt (z_{i−1} − z̄))² / (n Δt)]
    σ̂_x = √[Σ ((B_i − B̄) − λ̂₂ Δt (z_{i−1} − z̄))² / (n Δt)]
    μ̂_y = (1/n) Σ [A_i/Δt + λ̂₁ z_{i−1} + ½ σ̂_y²]
    μ̂_x = (1/n) Σ [B_i/Δt − λ̂₂ z_{i−1} + ½ σ̂_x²]
    ρ̂_xy = (1/n) Σ Z_y,i · Z_x,i

    Parameters
    ----------
    y, x   : price arrays (NOT log-transformed — the function does that)
    Delta_t: time interval between observations (1 trading day = 1)

    Returns
    -------
    (mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy)
    """
    Y = np.log(y)
    X = np.log(x)

    # Differences (log returns) and lagged spread
    A = Y[1:] - Y[:-1]       # ΔY_i
    B = X[1:] - X[:-1]       # ΔX_i
    z = Y[:-1] - X[:-1]      # z_{i-1}

    n = len(z)

    # Demeaned versions
    A_bar = np.mean(A)
    B_bar = np.mean(B)
    z_bar = np.mean(z)
    dA = A - A_bar
    dB = B - B_bar
    dz = z - z_bar

    denom = Delta_t * np.sum(dz ** 2)
    if denom == 0:
        return (np.nan,) * 7

    # λ estimates
    lambda_1 = 2.0 * np.sum(dA * dz) / denom
    lambda_2 = 2.0 * np.sum(dB * dz) / denom

    # σ estimates
    sigma_y = np.sqrt(
        np.sum((dA - lambda_1 * Delta_t * dz) ** 2) / (n * Delta_t))
    sigma_x = np.sqrt(
        np.sum((dB - lambda_2 * Delta_t * dz) ** 2) / (n * Delta_t))

    if sigma_y == 0 or sigma_x == 0:
        return (np.nan,) * 7

    # μ estimates
    mu_y = np.mean(A / Delta_t + lambda_1 * z + 0.5 * sigma_y ** 2)
    mu_x = np.mean(B / Delta_t - lambda_2 * z + 0.5 * sigma_x ** 2)

    # Standardised residuals for correlation
    Z_y = (A - (mu_y - lambda_1 * z - 0.5 * sigma_y ** 2) * Delta_t) / \
          (sigma_y * np.sqrt(Delta_t))
    Z_x = (B - (mu_x + lambda_2 * z - 0.5 * sigma_x ** 2) * Delta_t) / \
          (sigma_x * np.sqrt(Delta_t))

    rho_xy = np.mean(Z_y * Z_x)

    return mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy


def process_pair_mle(pair: tuple,
                     prices: pd.DataFrame,
                     window_size: int,
                     step_size: int) -> list:
    """
    Run rolling MLE estimation for a single stock pair over all windows.

    Pair selection conditions
    -------------------------
    Stability      :  λ₁ + λ₂ > 0   (stationarity of the spread)
    Participation  :  λ₁ > 0 AND λ₂ > 0  (both assets contribute to error correction)

    Returns
    -------
    list of dicts (one per valid window)
    """
    stock_y, stock_x = pair
    valid = []

    py = prices[stock_y]
    px = prices[stock_x]

    n_windows = (len(py) - window_size + 1) // step_size

    for i in range(n_windows):
        # We pass one extra sample at the start for lag-1 differences
        start_idx = i * step_size          # inclusive, 0-based
        end_idx   = start_idx + window_size + 1  # exclusive

        wy = py.iloc[start_idx:end_idx].values
        wx = px.iloc[start_idx:end_idx].values

        if len(wy) < window_size + 1:
            continue

        params = estimate_parameters(wy, wx, Delta_t=1.0)

        if any(np.isnan(p) for p in params):
            continue

        mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy = params

        stability      = (lambda_1 + lambda_2) > 0
        participation  = (lambda_1 > 0) and (lambda_2 > 0)  # both participate

        if stability and participation:
            idx = py.iloc[start_idx:end_idx].index
            # Formation-window spread statistics for z-score
            log_spread_form = np.log(wy / wx)
            valid.append({
                "Stock_y"    : stock_y,
                "Stock_x"    : stock_x,
                "Sector"     : TICKER_TO_SECTOR.get(stock_y, "Unknown"),
                "Start_Date" : idx[1],       # first observation used
                "End_Date"   : idx[-1],      # last observation (formation end)
                "mu_y"       : mu_y,
                "mu_x"       : mu_x,
                "lambda_1"   : lambda_1,
                "lambda_2"   : lambda_2,
                "sigma_y"    : sigma_y,
                "sigma_x"    : sigma_x,
                "rho_xy"     : rho_xy,
                "half_life"  : np.log(2) / max(lambda_1 + lambda_2, 1e-12),
                "spread_mean": float(np.mean(log_spread_form)),
                "spread_std" : float(np.std(log_spread_form, ddof=1)),
                "spread_n"   : int(len(log_spread_form)),
            })

    return valid


# ---------------------------------------------------------------------------
#  CORRELATION PRE-SCREENING (Stage 1 of two-stage pipeline)
# ---------------------------------------------------------------------------

def correlation_prescreen(prices: pd.DataFrame,
                          window_start: int,
                          window_end: int,
                          top_k: int = CORR_TOP_K_PER_SECTOR,
                          min_rho: float = MIN_CORR_THRESHOLD) -> list:
    """
    Stage 1: Pre-screen within-sector pairs by Pearson correlation of
    log-returns over a given formation window.

    For each GICS sector, compute all pairwise correlations and retain
    only the top-K pairs (with ρ ≥ min_rho).  This dramatically reduces
    the number of pairs sent to the expensive Stage 2 cointegration tests.

    References
    ----------
    Gatev, Goetzmann & Rouwenhorst (2006), Krauss (2017)

    Parameters
    ----------
    prices       : full price DataFrame (dates × tickers)
    window_start : integer index of the start of the formation window
    window_end   : integer index of the end of the formation window
    top_k        : max pairs to keep per sector
    min_rho      : minimum Pearson correlation threshold

    Returns
    -------
    list of (stock_y, stock_x) tuples that passed the screen
    """
    window_prices = prices.iloc[window_start:window_end]
    log_rets = np.log(window_prices / window_prices.shift(1)).dropna()

    prescreened = []
    for sector, tickers in SP500_SECTORS.items():
        available = [t for t in tickers if t in log_rets.columns]
        if len(available) < 2:
            continue

        # Compute full correlation matrix for this sector
        corr_matrix = log_rets[available].corr()

        # Extract upper-triangle pairwise correlations
        pair_corrs = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                rho = corr_matrix.iloc[i, j]
                if not np.isnan(rho) and rho >= min_rho:
                    pair_corrs.append(((available[i], available[j]), rho))

        # Sort by descending correlation, keep top-K
        pair_corrs.sort(key=lambda x: x[1], reverse=True)
        for (pair, _) in pair_corrs[:top_k]:
            prescreened.append(pair)

    return prescreened


def run_mle_pair_selection(prices: pd.DataFrame,
                           window_size: int = WINDOW,
                           step_size: int   = STEP) -> pd.DataFrame:
    """
    Two-stage parallelised MLE pair selection:
      Stage 1: Correlation pre-screening (fast, per window)
      Stage 2: MLE VECM estimation (slow, only on pre-screened pairs)

    Returns
    -------
    pd.DataFrame with columns:
        Stock_y, Stock_x, Sector, Start_Date, End_Date,
        mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy, half_life
    """
    # Collect unique pairs across all windows via correlation pre-screening
    n_windows = (len(prices) - window_size) // step_size
    all_pairs_set = set()
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx   = start_idx + window_size + 1
        window_pairs = correlation_prescreen(prices, start_idx, end_idx)
        all_pairs_set.update(window_pairs)

    all_pairs = sorted(all_pairs_set)

    # Also compute total within-sector pairs for reporting
    total_sector_pairs = 0
    for sector, tickers in SP500_SECTORS.items():
        available = [t for t in tickers if t in prices.columns]
        total_sector_pairs += len(list(itertools.combinations(available, 2)))

    print(f"\n[MLE] Correlation pre-screening: {len(all_pairs)} unique pairs "
          f"selected from {total_sector_pairs} within-sector candidates "
          f"(top-{CORR_TOP_K_PER_SECTOR}/sector, ρ≥{MIN_CORR_THRESHOLD})")
    print(f"[MLE] window={window_size}, step={step_size}")

    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_pair_mle)(p, prices, window_size, step_size)
        for p in all_pairs
    )

    flat = [rec for sublist in results for rec in sublist]
    pairs_df = pd.DataFrame(flat)

    if not pairs_df.empty:
        pairs_df["Start_Date"] = pd.to_datetime(pairs_df["Start_Date"])
        pairs_df["End_Date"]   = pd.to_datetime(pairs_df["End_Date"])

    print(f"[MLE] {len(pairs_df)} valid (pair, window) records found.")
    return pairs_df


# ===========================================================================
# 4. JOHANSEN TEST (benchmark pair selection)
# ===========================================================================

def process_pair_johansen(pair: tuple,
                          prices: pd.DataFrame,
                          window_size: int,
                          step_size: int,
                          confidence_level: int = 95) -> list:
    """
    Rolling Johansen trace test for a single stock pair.
    Stores detailed test output for econometric diagnostics.

    Returns list of dicts for windows where cointegration is confirmed.
    """
    conf_col = {90: 0, 95: 1, 99: 2}[confidence_level]
    stock_y, stock_x = pair
    records = []

    log_py = np.log(prices[stock_y])
    log_px = np.log(prices[stock_x])

    n_windows = (len(log_py) - window_size + 1) // step_size

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx   = start_idx + window_size + 1

        wy = log_py.iloc[start_idx:end_idx].values
        wx = log_px.iloc[start_idx:end_idx].values

        if len(wy) < window_size + 1:
            continue

        # Try multiple lag orders and pick best by AIC-like criterion
        best_lag = 1
        best_trace = -np.inf
        best_res = None
        for lag in range(1, 6):
            try:
                res = coint_johansen(np.column_stack((wy, wx)),
                                     det_order=0, k_ar_diff=lag)
                if res.lr1[0] > best_trace:
                    best_trace = res.lr1[0]
                    best_lag = lag
                    best_res = res
            except Exception:
                continue

        if best_res is None:
            continue

        crit = best_res.cvt[:, conf_col]
        if best_res.lr1[0] >= crit[0]:
            idx = prices[stock_y].iloc[start_idx:end_idx].index

            # ADF test on the spread
            spread = wy - wx
            try:
                adf_stat, adf_pval = adfuller(spread, maxlag=10)[:2]
            except Exception:
                adf_stat, adf_pval = np.nan, np.nan

            records.append({
                "Stock_y"      : stock_y,
                "Stock_x"      : stock_x,
                "Sector"       : TICKER_TO_SECTOR.get(stock_y, "Unknown"),
                "Start_Date"   : idx[1],
                "End_Date"     : idx[-1],
                "Trace_Stat"   : best_res.lr1[0],
                "Crit_90"      : best_res.cvt[0, 0],
                "Crit_95"      : best_res.cvt[0, 1],
                "Crit_99"      : best_res.cvt[0, 2],
                "Max_Eigen"    : best_res.lr2[0],
                "Eigenvalue_1" : best_res.eig[0],
                "Eigenvalue_2" : best_res.eig[1],
                "Lag_Order"    : best_lag,
                "ADF_Stat"     : adf_stat,
                "ADF_pvalue"   : adf_pval,
            })

    return records


def run_johansen_pair_selection(prices: pd.DataFrame,
                                window_size: int = WINDOW,
                                step_size: int   = STEP,
                                confidence_level: int = 95) -> pd.DataFrame:
    """
    Two-stage parallelised Johansen pair selection:
      Stage 1: Correlation pre-screening (fast)
      Stage 2: Johansen trace test (slow, only on pre-screened pairs)

    Returns
    -------
    pd.DataFrame with columns: Stock_y, Stock_x, Sector, Start_Date, End_Date,
        Trace_Stat, Crit_90/95/99, Max_Eigen, Eigenvalues, Lag_Order,
        ADF_Stat, ADF_pvalue
    """
    # Collect unique pairs across all windows via correlation pre-screening
    n_windows = (len(prices) - window_size) // step_size
    all_pairs_set = set()
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx   = start_idx + window_size + 1
        window_pairs = correlation_prescreen(prices, start_idx, end_idx)
        all_pairs_set.update(window_pairs)

    all_pairs = sorted(all_pairs_set)

    total_sector_pairs = 0
    for sector, tickers in SP500_SECTORS.items():
        available = [t for t in tickers if t in prices.columns]
        total_sector_pairs += len(list(itertools.combinations(available, 2)))

    print(f"\n[Johansen] Correlation pre-screening: {len(all_pairs)} unique pairs "
          f"selected from {total_sector_pairs} within-sector candidates "
          f"(top-{CORR_TOP_K_PER_SECTOR}/sector, ρ≥{MIN_CORR_THRESHOLD})")
    print(f"[Johansen] confidence={confidence_level}%")

    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_pair_johansen)(
            p, prices, window_size, step_size, confidence_level)
        for p in all_pairs
    )

    flat = [rec for sublist in results for rec in sublist]
    df   = pd.DataFrame(flat)

    if not df.empty:
        df["Start_Date"] = pd.to_datetime(df["Start_Date"])
        df["End_Date"]   = pd.to_datetime(df["End_Date"])

    print(f"[Johansen] {len(df)} valid (pair, window) records found.")
    return df


# ===========================================================================
# 5. OUT-OF-SAMPLE PROCESSING (deltas, gammas, triggers)
# ===========================================================================

def process_pair_oos(pairs_df_filtered: pd.DataFrame,
                     prices_oos: pd.DataFrame,
                     impl_vols_oos: pd.DataFrame,
                     rates_oos: pd.Series) -> list:
    """
    Compute spread metrics, deltas, gammas and trading triggers for each
    valid pair during the 6-month out-of-sample (OOS) trading window.

    Black-Scholes inputs
    --------------------
    S = F = P_y / P_x  (spread "forward" price)
    K = 1  (ATM spread option has equal forward and strike)
    σ_spread = √(σ_y² + σ_x² − 2 ρ σ_y σ_x)  (spread volatility, risk-neutral)
    r         : 6-month T-bill rate (daily, fraction)
    k         : time-to-maturity in years (counted backwards from 0 → ~0.5)

    d₁ = (z + ½ σ_spread² τ) / (σ_spread √τ)     [Margrabe, no r]
    d₂ = d₁ − σ_spread √τ

    Δ_y = N(d₁)          [long leg delta]
    Δ_x = N(d₂)          [short leg delta]
    Γ_y = N'(d₁) / (P_y · σ_spread √k)
    Γ_x = N'(d₂) / (P_x · σ_spread √k)

    Entry trigger (open)
    --------------------
    Open when:  z_t  > (1 − λ₁ − λ₂) · σ_spread_t

    Exit trigger (close)
    --------------------
    Det J = Γ_x² / Γ_y² − P_y / P_x
    Close when this quantity crosses zero (sign-change between consecutive days)

    Trigger encoding
    ----------------
    1   = open position
    0   = close position
    NaN = hold (no change)

    Parameters
    ----------
    pairs_df_filtered : rows of pairs_df for this formation-window end date
    prices_oos        : price DataFrame for the next 126 trading days
    impl_vols_oos     : IV DataFrame for the same 126 days
    rates_oos         : risk-free rate Series for the same 126 days

    Returns
    -------
    list of dicts — one per (date, pair) — to be collected into oos_df
    """
    records = []
    r = rates_oos / 100.0           # convert % → decimal

    for _, row in pairs_df_filtered.iterrows():
        sy = row["Stock_y"]
        sx = row["Stock_x"]

        if sy not in prices_oos.columns or sx not in prices_oos.columns:
            continue
        if sy not in impl_vols_oos.columns or sx not in impl_vols_oos.columns:
            continue

        price_y  = prices_oos[sy]
        price_x  = prices_oos[sx]
        sigma_y  = impl_vols_oos[sy]    # already annualised (fraction)
        sigma_x  = impl_vols_oos[sx]

        # Log spread
        z = np.log(price_y / price_x)

        # ---- Z-score using formation-window statistics --------------------
        spread_mu  = row.get("spread_mean", z.mean())
        spread_sig = row.get("spread_std", z.std())
        spread_n   = row.get("spread_n", WINDOW)
        if spread_sig < 1e-12:
            spread_sig = 1e-12
        z_score   = (z - spread_mu) / spread_sig
        z_score_se = spread_sig / np.sqrt(spread_n)  # SE of the mean

        # Risk-neutral spread volatility
        rho      = row["rho_xy"]
        sigmaRN  = np.sqrt(sigma_y**2 + sigma_x**2 - 2 * sigma_y * sigma_x * rho)
        sigmaRN  = sigmaRN.clip(lower=1e-8)

        # Time-to-maturity (years): day 1 of window ≈ 0.5 years, day 126 ≈ 0
        T   = STEP               # total length of trading window (days)
        idx = np.arange(T, 0, -1)[:len(z)]
        k   = idx / YEAR         # fraction of year remaining

        # Avoid division by zero at last day
        k   = np.where(k < 1e-8, 1e-8, k)
        k_s = pd.Series(k, index=z.index)

        sigma_T  = sigmaRN * np.sqrt(k_s)
        sigma_T  = sigma_T.clip(lower=1e-8)

        d1 = (z + 0.5 * sigmaRN**2 * k_s) / sigma_T   # Margrabe: signed z, no r
        d2 = d1 - sigma_T

        delta_y  = norm.cdf(d1.values)
        delta_x  = norm.cdf(d2.values)
        gamma_y  = norm.pdf(d1.values) / (price_y.values * sigma_T.values)
        gamma_x  = norm.pdf(d2.values) / (price_x.values * sigma_T.values)

        # ---- Entry trigger -------------------------------------------------
        open_thresh = (1 - row["lambda_1"] - row["lambda_2"]) * sigmaRN

        # ---- Exit trigger --------------------------------------------------
        close_formula = (gamma_x**2 / np.where(gamma_y**2 < 1e-300, 1e-300,
                                                gamma_y**2)
                         ) - (price_y.values / price_x.values)
        # Sign-change between consecutive days → zero crossing
        sign_change   = close_formula[1:] * close_formula[:-1]
        # Pad first element with NaN (no prior day available)
        sign_change   = np.concatenate([[np.nan], sign_change])

        # ---- Combine into trigger array -----------------------------------
        # close (0) takes priority over open (1) over hold (NaN)
        trigger = np.full(len(z), np.nan)
        for j in range(len(z)):
            sc = sign_change[j]
            zt = z.iloc[j]
            ot = open_thresh.iloc[j]
            if not np.isnan(sc) and sc <= 0:
                trigger[j] = -1         # close condition
            elif zt > ot:
                trigger[j] = 1         # open condition
            # else: NaN → hold

        # ---- Collect records -----------------------------------------------
        for j, date in enumerate(z.index):
            records.append({
                "Date"             : date,
                "Stock_y"          : sy,
                "Stock_x"          : sx,
                "Price_y"          : price_y.iloc[j],
                "Price_x"          : price_x.iloc[j],
                "Vol_y"            : sigma_y.iloc[j],
                "Vol_x"            : sigma_x.iloc[j],
                "Time_to_mat_years": k_s.iloc[j],
                "spread_z"         : z.iloc[j],
                "sigmaRN_z"        : sigmaRN.iloc[j],
                "Z_Score"          : z_score.iloc[j],
                "Z_Score_SE"       : z_score_se,
                "Delta_y"          : delta_y[j],
                "Delta_x"          : delta_x[j],
                "Gamma_y"          : gamma_y[j],
                "Gamma_x"          : gamma_x[j],
                "Trigger"          : trigger[j],
            })

    return records


def run_oos_processing(pairs_df: pd.DataFrame,
                       prices: pd.DataFrame,
                       impl_vols: pd.DataFrame,
                       rates: pd.Series) -> pd.DataFrame:
    """
    Loop over every formation-window end date, calling process_pair_oos
    in parallel for each trading window.

    The OOS window for a given formation end-date starts on that date and
    spans the next `step_size` trading days (matching the original notebook).

    Returns
    -------
    pd.DataFrame  (the full OOS panel, saved to output/oos_MLE_SP500.csv)
    """
    unique_end_dates = sorted(pairs_df["End_Date"].unique())
    print(f"\n[OOS] Processing {len(unique_end_dates)} trading windows ...")

    n_jobs = max(1, multiprocessing.cpu_count() - 1)

    def _process_one_window(end_date):
        pf   = pairs_df[pairs_df["End_Date"] == end_date]
        p_oos = prices[prices.index >= end_date].head(STEP)
        v_oos = impl_vols[impl_vols.index >= end_date].head(STEP)
        r_oos = rates[rates.index >= end_date].head(STEP)

        # Align indices (inner join on common dates)
        common = p_oos.index.intersection(v_oos.index).intersection(r_oos.index)
        if len(common) < 5:
            return []
        return process_pair_oos(pf,
                                p_oos.loc[common],
                                v_oos.loc[common],
                                r_oos.loc[common])

    all_records = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one_window)(ed) for ed in unique_end_dates
    )

    flat   = [rec for sublist in all_records for rec in sublist]
    oos_df = pd.DataFrame(flat)

    if not oos_df.empty:
        oos_df["Date"] = pd.to_datetime(oos_df["Date"])
        out_path = os.path.join(OUTPUT_DIR, "oos_MLE_SP500.csv")
        oos_df.to_csv(out_path, index=False)
        print(f"[OOS] Saved to {out_path}  ({len(oos_df):,} rows)")

    return oos_df


def filter_oos_signals(oos_df: pd.DataFrame,
                       max_entries_per_day: int = MAX_NEW_ENTRIES_PER_DAY,
                       min_zscore: float = MIN_ZSCORE_ENTRY) -> pd.DataFrame:
    """
    Limit overtrading by keeping only the strongest entry signals each day.

    Motivation (Gatev et al., 2006; Krauss, 2017)
    -----------------------------------------------
    In pairs-trading strategies, transaction costs erode profits from weak
    mean-reversion signals.  By ranking daily entry candidates by the
    absolute z-score of the spread (a measure of divergence strength) and
    retaining only the top-K, the strategy focuses on the pairs with the
    highest expected mean-reversion profit relative to the spread's
    historical variability.

    Algorithm
    ---------
    For each trading day t:
      1. Identify all (pair, date) rows where Trigger == 1 (entry signal).
      2. Remove any candidate with |Z_Score| < min_zscore (too weak).
      3. Rank remaining candidates by |Z_Score| in descending order.
      4. Keep the top-K entries; set Trigger = NaN (hold) for the rest.

    Close signals (Trigger == 0) and hold signals (Trigger == NaN) are
    never modified — we only restrict *new* openings.

    Parameters
    ----------
    oos_df             : full OOS DataFrame with Trigger and Z_Score columns
    max_entries_per_day: K, maximum number of new positions opened per day
    min_zscore         : minimum |Z_Score| to consider opening a position

    Returns
    -------
    pd.DataFrame  (same shape, with some Trigger==1 values set to NaN)
    """
    df = oos_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    n_before = (df["Trigger"] == 1).sum()

    # --- Step 1-2: Apply minimum z-score filter on entry signals -------------
    weak_mask = (df["Trigger"] == 1) & (df["Z_Score"].abs() < min_zscore)
    df.loc[weak_mask, "Trigger"] = np.nan
    n_weak_removed = weak_mask.sum()

    # --- Step 3-4: Per-day top-K ranking by |z-score| ------------------------
    #     We process only rows that still have Trigger == 1
    entry_mask = df["Trigger"] == 1
    entry_rows = df.loc[entry_mask].copy()
    entry_rows["abs_zscore"] = entry_rows["Z_Score"].abs()

    # For each date, rank and mark rows beyond top-K for demotion
    demote_indices = []
    for date, grp in entry_rows.groupby("Date"):
        if len(grp) <= max_entries_per_day:
            continue  # fewer candidates than K → keep all
        # Sort descending by |z-score|; indices beyond K are demoted
        ranked = grp.sort_values("abs_zscore", ascending=False)
        demote_indices.extend(ranked.index[max_entries_per_day:].tolist())

    df.loc[demote_indices, "Trigger"] = np.nan
    n_cap_removed = len(demote_indices)

    n_after = (df["Trigger"] == 1).sum()

    print(f"\n{'='*60}")
    print("  SIGNAL-STRENGTH FILTER (Gatev et al., 2006; Krauss, 2017)")
    print(f"{'='*60}")
    print(f"  Entry signals before filtering:  {n_before:,}")
    print(f"  Removed (|z| < {min_zscore}):          {n_weak_removed:,}")
    print(f"  Removed (daily cap K={max_entries_per_day}):        {n_cap_removed:,}")
    print(f"  Entry signals after filtering:   {n_after:,}")
    print(f"  Reduction:                       {(1 - n_after/max(n_before,1))*100:.1f}%")
    print(f"{'='*60}")

    # Drop helper column
    if "abs_zscore" in df.columns:
        df.drop(columns=["abs_zscore"], inplace=True)

    return df


# ===========================================================================
# 5B. VANILLA (PLAIN) PAIRS TRADING — OOS PROCESSING
# ===========================================================================

def process_pair_oos_vanilla(pairs_df_filtered: pd.DataFrame,
                              prices_oos: pd.DataFrame) -> list:
    """
    Plain vanilla pairs trading: fixed position sizes, fixed z-score
    entry/exit thresholds.  No options, no Black-Scholes, no gammas.

    Entry trigger:  |z_score| > VANILLA_ENTRY_ZSCORE (default 2.0)
    Exit  trigger:  |z_score| < VANILLA_EXIT_ZSCORE  (default 0.5)
    Position size:  ±VANILLA_POSITION_SIZE

    Parameters
    ----------
    pairs_df_filtered : rows of pairs_df for this formation-window end date
    prices_oos        : price DataFrame for the next 126 trading days

    Returns
    -------
    list of dicts — one per (date, pair)
    """
    records = []

    for _, row in pairs_df_filtered.iterrows():
        sy = row["Stock_y"]
        sx = row["Stock_x"]

        if sy not in prices_oos.columns or sx not in prices_oos.columns:
            continue

        price_y = prices_oos[sy]
        price_x = prices_oos[sx]

        # Log spread
        z = np.log(price_y / price_x)

        # Z-score using formation-window statistics
        spread_mu  = row.get("spread_mean", z.mean())
        spread_sig = row.get("spread_std", z.std())
        if spread_sig < 1e-12:
            spread_sig = 1e-12
        z_score = (z - spread_mu) / spread_sig

        # ---- Trigger logic (fixed thresholds) ----
        # 1 = open long Y / short X  (z_score > entry)
        # -1 = open short Y / long X (z_score < -entry)
        # 0 = close
        # NaN = hold
        trigger = np.full(len(z), np.nan)
        for j in range(len(z)):
            zs = z_score.iloc[j]
            if abs(zs) < VANILLA_EXIT_ZSCORE:
                trigger[j] = 0        # close — spread reverted
            elif zs > VANILLA_ENTRY_ZSCORE:
                trigger[j] = 1        # open long Y / short X
            elif zs < -VANILLA_ENTRY_ZSCORE:
                trigger[j] = -1       # open short Y / long X
            # else: NaN → hold

        # ---- Collect records ----
        for j, date in enumerate(z.index):
            records.append({
                "Date"      : date,
                "Stock_y"   : sy,
                "Stock_x"   : sx,
                "Price_y"   : price_y.iloc[j],
                "Price_x"   : price_x.iloc[j],
                "Z_Score"   : z_score.iloc[j],
                "Trigger"   : trigger[j],
            })

    return records


def run_oos_processing_vanilla(pairs_df: pd.DataFrame,
                                prices: pd.DataFrame) -> pd.DataFrame:
    """
    Run vanilla OOS processing for all formation windows in parallel.
    """
    unique_end_dates = sorted(pairs_df["End_Date"].unique())
    print(f"\n[OOS-VANILLA] Processing {len(unique_end_dates)} trading windows ...")

    n_jobs = max(1, multiprocessing.cpu_count() - 1)

    def _process_one_window(end_date):
        pf    = pairs_df[pairs_df["End_Date"] == end_date]
        p_oos = prices[prices.index >= end_date].head(STEP)
        if len(p_oos) < 5:
            return []
        return process_pair_oos_vanilla(pf, p_oos)

    all_records = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one_window)(ed) for ed in unique_end_dates
    )

    flat    = [rec for sublist in all_records for rec in sublist]
    oos_van = pd.DataFrame(flat)

    if not oos_van.empty:
        oos_van["Date"] = pd.to_datetime(oos_van["Date"])
        out_path = os.path.join(OUTPUT_DIR, "oos_vanilla_SP500.csv")
        oos_van.to_csv(out_path, index=False)
        print(f"[OOS-VANILLA] Saved to {out_path}  ({len(oos_van):,} rows)")

    return oos_van


def vanilla_position_sizes(trigger: np.ndarray) -> tuple:
    """
    State-machine for vanilla fixed-size positions.

    trigger ==  1  → open: long Y (+1), short X (-1)
    trigger == -1  → open: short Y (-1), long X (+1)
    trigger ==  0  → close
    trigger == NaN → hold
    """
    pos_y = np.full(len(trigger), np.nan)
    pos_x = np.full(len(trigger), np.nan)
    d_y   = np.nan
    d_x   = np.nan

    for i in range(len(trigger)):
        pos_y[i] = d_y
        pos_x[i] = d_x

        t = trigger[i]
        if t == 0:
            d_y = np.nan
            d_x = np.nan
        elif t == 1:
            d_y =  VANILLA_POSITION_SIZE
            d_x = -VANILLA_POSITION_SIZE
        elif t == -1:
            d_y = -VANILLA_POSITION_SIZE
            d_x =  VANILLA_POSITION_SIZE
        # else NaN → hold previous

    return pos_y, pos_x


def calculate_pair_returns_vanilla(pair_data: pd.DataFrame,
                                    tc_bps: float = TC_BPS) -> np.ndarray:
    """
    Daily P&L for a single pair using fixed ±1 positions.
    """
    pos_y, pos_x = vanilla_position_sizes(pair_data["Trigger"].values)

    py   = pair_data["Price_y"].values
    px   = pair_data["Price_x"].values
    trig = pair_data["Trigger"].values

    denom = np.abs(pos_y[:-1]) * py[:-1] + np.abs(pos_x[:-1]) * px[:-1]
    denom = np.where(denom < 1e-12, np.nan, denom)

    ret_y = pos_y[1:] * np.diff(py) / denom
    ret_x = (-pos_x[1:]) * np.diff(px) / denom

    pair_ret = ret_y + ret_x

    tc_per_event = 2.0 * tc_bps / 10_000.0
    for j in range(len(pair_ret)):
        t = trig[j + 1] if (j + 1) < len(trig) else np.nan
        if t == 0 or t == 1 or t == -1:
            pair_ret[j] -= tc_per_event

    return pair_ret


def aggregate_returns_vanilla(oos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily returns for the vanilla strategy — same logic as
    aggregate_returns() but using fixed-position returns.
    """
    returns_dict = {}
    count_dict   = {}

    oos_df = oos_df.copy()
    oos_df["Date"] = pd.to_datetime(oos_df["Date"])

    for (sy, sx), grp in oos_df.groupby(["Stock_y", "Stock_x"]):
        grp = grp.sort_values("Date").reset_index(drop=True)
        n_windows = max(1, len(grp) // STEP)

        for w in range(n_windows):
            batch = grp.iloc[w * STEP: (w + 1) * STEP].copy()
            batch.index = batch["Date"]

            if batch["Trigger"].isna().all():
                continue

            rets = calculate_pair_returns_vanilla(batch)

            for date, ret in zip(batch.index, rets):
                if np.isnan(ret):
                    continue
                if date in returns_dict:
                    returns_dict[date] += ret
                    count_dict[date]   += 1
                else:
                    returns_dict[date]  = ret
                    count_dict[date]    = 1

    if not returns_dict:
        return pd.DataFrame()

    dates     = sorted(returns_dict.keys())
    total_ret = [returns_dict[d] for d in dates]
    counts    = [count_dict[d]   for d in dates]
    daily_ret = [r / c for r, c in zip(total_ret, counts)]

    combined_df = pd.DataFrame({
        "Returns"     : total_ret,
        "Count"       : counts,
        "Daily_Return": daily_ret,
    }, index=dates)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df.index.name = "Date"

    return combined_df


# ===========================================================================
# 6. POSITION SIZING & RETURNS
# ===========================================================================

def position_sizes(delta_y: np.ndarray,
                   delta_x: np.ndarray,
                   trigger: np.ndarray) -> tuple:
    """
    State-machine that converts trigger signals into position sizes.

    Rules
    -----
    trigger == 0      → close (set position to 0/NaN)
    trigger == 1      → open  (long Y at delta_y, short X at delta_x)
    trigger == NaN    → hold  (keep previous position unchanged)

    Opening is Y-long / X-short because when z > open_thresh the spread
    is above its threshold → Y is relatively expensive → short Y, long X.
    However, the original notebook applies the sign externally via the
    return formula (returns_y positive for Y_long), so the position sizes
    follow the sign convention used there.

    Returns
    -------
    (pos_y, pos_x)  : numpy arrays of position sizes (NaN when flat)
    """
    pos_y  = np.full(len(trigger), np.nan)
    pos_x  = np.full(len(trigger), np.nan)
    d_y    = np.nan
    d_x    = np.nan
    k_prev = np.nan

    for i in range(len(trigger)):
        pos_y[i] = d_y   # record position for *this* day before updating
        pos_x[i] = d_x

        t = trigger[i]
        if t == 0:
            d_y = np.nan
            d_x = np.nan
            k_prev = np.nan
        elif not np.isnan(t) and t != k_prev:
            k_prev = t
            d_y    =  t * delta_y[i]    # long Y
            d_x    = -t * delta_x[i]    # short X

    return pos_y, pos_x


def calculate_pair_returns(pair_data: pd.DataFrame,
                           tc_bps: float = TC_BPS) -> np.ndarray:
    """
    Compute daily P&L for a single pair using the delta-weighted positions.

    Transaction costs are deducted on every open (trigger==1) and close
    (trigger==0) event.  Cost per event = 2 × tc_bps / 10_000  (both legs).

    Returns
    -------
    np.ndarray  (length = len(pair_data), first element always NaN)
    """
    pos_y, pos_x = position_sizes(
        pair_data["Delta_y"].values,
        pair_data["Delta_x"].values,
        pair_data["Trigger"].values,
    )

    py   = pair_data["Price_y"].values
    px   = pair_data["Price_x"].values
    trig = pair_data["Trigger"].values

    # Normalise by total portfolio value (both legs)
    denom_total = np.abs(pos_y[:-1]) * py[:-1] + np.abs(pos_x[:-1]) * px[:-1]
    denom_total = np.where(denom_total < 1e-12, np.nan, denom_total)

    ret_y = pos_y[1:] * np.diff(py) / denom_total
    ret_x = (-pos_x[1:]) * np.diff(px) / denom_total

    pair_ret = ret_y + ret_x

    # --- Deduct transaction costs on open/close triggers --------------------
    tc_per_event = 2.0 * tc_bps / 10_000.0  # both legs, one-way
    for j in range(len(pair_ret)):
        t = trig[j + 1] if (j + 1) < len(trig) else np.nan
        if t == 0 or t == 1:            # open or close event
            pair_ret[j] -= tc_per_event

    return pair_ret


def aggregate_returns(oos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over every (pair, window) batch in oos_df, compute daily
    returns, and aggregate cross-sectionally (equal weight = average).

    The OOS df is ordered chronologically by batch (each batch is one
    6-month window for all active pairs).  We iterate over each unique
    (Stock_y, Stock_x, first_date_of_window) group which mirrors the
    original notebook's step-based batching.

    Returns
    -------
    pd.DataFrame with columns: Returns, Count, Daily_Return
    """
    returns_dict = {}
    count_dict   = {}

    # Group by pair and then by 6-month window start
    oos_df = oos_df.copy()
    oos_df["Date"] = pd.to_datetime(oos_df["Date"])

    # Identify each (pair, window) by its minimum date
    for (sy, sx), grp in oos_df.groupby(["Stock_y", "Stock_x"]):
        # Split into 6-month windows
        grp = grp.sort_values("Date").reset_index(drop=True)
        n_windows = max(1, len(grp) // STEP)

        for w in range(n_windows):
            batch = grp.iloc[w * STEP: (w + 1) * STEP].copy()
            batch.index = batch["Date"]

            if batch["Trigger"].isna().all():
                continue

            rets = calculate_pair_returns(batch)

            for date, ret in zip(batch.index, rets):
                if np.isnan(ret):
                    continue
                if date in returns_dict:
                    returns_dict[date] += ret
                    count_dict[date]   += 1
                else:
                    returns_dict[date]  = ret
                    count_dict[date]    = 1

    if not returns_dict:
        return pd.DataFrame()

    dates        = sorted(returns_dict.keys())
    total_ret    = [returns_dict[d] for d in dates]
    counts       = [count_dict[d]   for d in dates]
    daily_ret    = [r / c for r, c in zip(total_ret, counts)]

    combined_df  = pd.DataFrame({
        "Returns"     : total_ret,
        "Count"       : counts,
        "Daily_Return": daily_ret,
    }, index=dates)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df.index.name = "Date"

    return combined_df


# ===========================================================================
# 7. PERFORMANCE METRICS
# ===========================================================================

def compute_performance(combined_df: pd.DataFrame,
                        rates: pd.Series) -> dict:
    """
    Compute annual returns, annual excess returns, and Sharpe ratio.

    Risk-free rate is aligned to the combined_df index (inner join) and
    expressed as a daily rate (annualised % / 365).

    Returns
    -------
    dict with keys:
        'annual'   : DataFrame (Year × [Annual Returns, Annual Excess Returns,
                                         Sharpe Ratio])
        'summary'  : describe() of the annual DataFrame
        'skewness' : float
        'min_daily': float
        'max_daily': float
    """
    # Align rate to combined_df dates
    rf_aligned = rates.reindex(combined_df.index).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0    # daily risk-free return

    combined_df = combined_df.copy()
    combined_df["RF_daily"] = rf_daily.values

    # Annual returns
    combined_df["Year"] = combined_df.index.year
    grouped = combined_df.groupby("Year")

    annual_records = []
    for year, grp in grouped:
        ann_ret    = grp["Daily_Return"].add(1).prod() - 1
        daily_excess = grp["Daily_Return"] - grp["RF_daily"]
        excess_ret = daily_excess.add(1).prod() - 1
        ann_mean   = daily_excess.mean() * YEAR
        ann_vol    = daily_excess.std() * np.sqrt(YEAR)
        sharpe     = ann_mean / ann_vol if ann_vol > 1e-12 else np.nan
        annual_records.append({
            "Year"                 : year,
            "Annual Returns"       : ann_ret,
            "Annual Excess Returns": excess_ret,
            "Sharpe Ratio"         : sharpe,
        })

    annual_df = pd.DataFrame(annual_records).set_index("Year")

    result = {
        "annual"   : annual_df,
        "summary"  : annual_df.describe(),
        "skewness" : skew(combined_df["Daily_Return"].dropna()),
        "min_daily": combined_df["Daily_Return"].min(),
        "max_daily": combined_df["Daily_Return"].max(),
    }
    return result


def compute_semiannual_performance(combined_df: pd.DataFrame,
                                   rates: pd.Series) -> pd.DataFrame:
    """
    Compute 6-month returns and Sharpe ratios (matching original notebook).

    Returns
    -------
    pd.DataFrame with columns: '6-months Returns', 'Sharpe Ratio'
    """
    rf_aligned = rates.reindex(combined_df.index).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    combined_df = combined_df.copy()
    combined_df["RF_daily"]   = rf_daily.values
    combined_df["Excess_ret"] = combined_df["Daily_Return"] - combined_df["RF_daily"]

    # Resample to 6-month periods (semi-annual, period ending on last day of month)
    resampled = combined_df.resample("6ME")

    records = []
    for period_end, grp in resampled:
        if len(grp) < 10:
            continue
        ret      = grp["Excess_ret"].add(1).prod() - 1
        ann_mean = grp["Excess_ret"].mean() * YEAR
        ann_vol  = grp["Excess_ret"].std() * np.sqrt(YEAR)
        sharpe   = ann_mean / ann_vol if ann_vol > 1e-12 else np.nan
        records.append({
            "Period_End"      : period_end,
            "6-months Returns": ret,
            "Sharpe Ratio"    : sharpe,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("Period_End")


# ===========================================================================
# 8. PLOTTING
# ===========================================================================

def plot_cumulative_returns(combined_df: pd.DataFrame,
                            rates: pd.Series,
                            save_dir: str = OUTPUT_DIR):
    """
    Plot (and save) cumulative returns and cumulative excess returns as
    separate PNG files.
    """
    rf_aligned = rates.reindex(combined_df.index).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    cum_ret    = (1 + combined_df["Daily_Return"]).cumprod()
    cum_excess = (1 + combined_df["Daily_Return"] - rf_daily).cumprod()

    # --- Figure 1: Cumulative Returns -----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined_df.index, cum_ret, color="steelblue", lw=1.2)
    ax.set_title("Cumulative Returns Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Return")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(save_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure 2: Cumulative Excess Returns ----------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined_df.index, cum_excess, color="darkorange", lw=1.2)
    ax.set_title("Cumulative Excess Returns Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Excess Return")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(save_dir, "cumulative_excess_returns.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)


def plot_annual_performance(annual_df: pd.DataFrame,
                            save_dir: str = OUTPUT_DIR):
    """
    Bar charts of annual returns and Sharpe ratio — one figure each.
    """
    # --- Figure 1: Annual Returns -------------------------------------------
    colors_ret = ["steelblue" if v >= 0 else "salmon"
                  for v in annual_df["Annual Returns"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual_df.index.astype(str),
           annual_df["Annual Returns"] * 100,
           color=colors_ret)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual Returns (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "annual_returns.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure 2: Annual Sharpe Ratio --------------------------------------
    colors_sh = ["steelblue" if v >= 0 else "salmon"
                 for v in annual_df["Sharpe Ratio"].fillna(0)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual_df.index.astype(str),
           annual_df["Sharpe Ratio"].fillna(0),
           color=colors_sh)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual Sharpe Ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sharpe Ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "annual_sharpe.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)


def plot_pair_counts(pairs_df: pd.DataFrame,
                     pairs_df_johansen: pd.DataFrame,
                     save_dir: str = OUTPUT_DIR):
    """
    Compare number of selected pairs per window: MLE vs Johansen.
    """
    mle_counts = pairs_df.groupby("End_Date")["Stock_y"].count()
    joh_counts = pairs_df_johansen.groupby("End_Date")["Stock_y"].count()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(mle_counts.index, mle_counts.values, label="MLE",
            marker="o", ms=4, color="steelblue")
    ax.plot(joh_counts.index, joh_counts.values, label="Johansen",
            marker="s", ms=4, color="darkorange")
    ax.set_title("Number of Selected Pairs per Formation Window")
    ax.set_xlabel("Window End Date")
    ax.set_ylabel("Pair Count")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(save_dir, "pair_counts.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[PLOT] Saved → {path}")


# ===========================================================================
# 9. ECONOMETRIC ANALYSIS FIGURES
# ===========================================================================

def plot_cointegration_diagnostics(pairs_df_johansen: pd.DataFrame,
                                    pairs_df_mle: pd.DataFrame,
                                    save_dir: str = OUTPUT_DIR):
    """
    Cointegration diagnostics — each panel saved as a separate PNG.
    """
    if pairs_df_johansen.empty:
        print("[PLOT] No Johansen results to plot.")
        return

    # --- Figure A: Trace statistic by sector ---------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    if "Sector" in pairs_df_johansen.columns and "Trace_Stat" in pairs_df_johansen.columns:
        sector_stats = pairs_df_johansen.groupby("Sector")["Trace_Stat"].agg(
            ["mean", "count"]).sort_values("mean", ascending=False)
        colors = plt.cm.RdYlGn(
            (sector_stats["mean"] - sector_stats["mean"].min()) /
            max(sector_stats["mean"].max() - sector_stats["mean"].min(), 1e-8))
        bars = ax.barh(sector_stats.index, sector_stats["mean"], color=colors)
        ax.set_xlabel("Mean Trace Statistic")
        ax.set_title("Average Johansen Trace Statistic by Sector", fontweight="bold")
        ax.axvline(x=15.49, color="red", linestyle="--", alpha=0.7, label="95% Critical Value")
        ax.legend(fontsize=9)
        for bar, cnt in zip(bars, sector_stats["count"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"n={cnt}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "coint_trace_by_sector.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure B: Lag order distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if "Lag_Order" in pairs_df_johansen.columns:
        lag_counts = pairs_df_johansen["Lag_Order"].value_counts().sort_index()
        ax.bar(lag_counts.index.astype(str), lag_counts.values,
               color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Lag Order (k)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Optimal Lag Orders", fontweight="bold")
        total = lag_counts.sum()
        for i, (lag, cnt) in enumerate(zip(lag_counts.index, lag_counts.values)):
            ax.text(i, cnt + total * 0.01, f"{cnt/total*100:.1f}%",
                    ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "coint_lag_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure C: ADF p-value distribution ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if "ADF_pvalue" in pairs_df_johansen.columns:
        adf_vals = pairs_df_johansen["ADF_pvalue"].dropna()
        ax.hist(adf_vals, bins=30, color="steelblue", edgecolor="white",
                alpha=0.8, density=True)
        ax.axvline(x=0.01, color="red", linestyle="--", alpha=0.7, label="1% level")
        ax.axvline(x=0.05, color="orange", linestyle="--", alpha=0.7, label="5% level")
        ax.axvline(x=0.10, color="green", linestyle="--", alpha=0.7, label="10% level")
        ax.set_xlabel("ADF p-value (spread stationarity)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of ADF Test p-values on Spread", fontweight="bold")
        ax.legend(fontsize=9)
        pct_sig = (adf_vals < 0.05).mean() * 100
        ax.text(0.95, 0.95, f"{pct_sig:.1f}% reject H₀\nat 5% level",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "coint_adf_pvalues.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure D: Top 10 cointegrated pairs table ---------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    if not pairs_df_johansen.empty and "Trace_Stat" in pairs_df_johansen.columns:
        latest_window = pairs_df_johansen["End_Date"].max()
        latest = pairs_df_johansen[pairs_df_johansen["End_Date"] == latest_window]
        top = latest.nlargest(10, "Trace_Stat")
        if len(top) > 0:
            table_data = []
            for _, r in top.iterrows():
                table_data.append([
                    f"{r['Stock_y']}/{r['Stock_x']}",
                    r.get("Sector", ""),
                    f"{r['Trace_Stat']:.2f}",
                    f"{r.get('Crit_95', 0):.2f}",
                    str(int(r.get("Lag_Order", 1))),
                    f"{r.get('ADF_pvalue', np.nan):.3f}" if not np.isnan(r.get("ADF_pvalue", np.nan)) else "N/A",
                ])
            table = ax.table(
                cellText=table_data,
                colLabels=["Pair", "Sector", "Trace Stat", "Crit 95%", "Lags", "ADF p-val"],
                loc="center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.6)
            for j in range(6):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", fontweight="bold")
            ax.set_title("Top 10 Cointegrated Pairs (Latest Window)",
                         fontweight="bold", fontsize=11, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "coint_top10_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)


def plot_spread_analysis(pairs_df: pd.DataFrame,
                          prices: pd.DataFrame,
                          n_pairs: int = 5,
                          save_dir: str = OUTPUT_DIR):
    """
    Time series of the log-spread z_t for top pairs with mean-reversion bands.
    """
    if pairs_df.empty:
        print("[PLOT] No pairs to plot spreads.")
        return

    # Pick top N pairs by frequency across windows
    pair_freq = pairs_df.groupby(["Stock_y", "Stock_x"]).size().nlargest(n_pairs)
    top_pairs = pair_freq.index.tolist()

    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 4 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, (sy, sx) in zip(axes, top_pairs):
        if sy not in prices.columns or sx not in prices.columns:
            continue
        z = np.log(prices[sy] / prices[sx])
        mu = z.expanding(min_periods=50).mean()
        sigma = z.expanding(min_periods=50).std()

        ax.plot(z.index, z.values, color="steelblue", lw=0.8, label="Spread z_t")
        ax.plot(z.index, mu.values, color="black", lw=1.0, ls="--",
                label="Expanding Mean", alpha=0.7)
        ax.fill_between(z.index, (mu - sigma).values, (mu + sigma).values,
                        alpha=0.15, color="green", label="±1σ band")
        ax.fill_between(z.index, (mu - 2 * sigma).values, (mu + 2 * sigma).values,
                        alpha=0.08, color="orange", label="±2σ band")
        sector = TICKER_TO_SECTOR.get(sy, "")
        ax.set_title(f"{sy} / {sx}  ({sector})  —  "
                     f"{pair_freq[(sy, sx)]} valid windows", fontweight="bold")
        ax.set_ylabel("Log Spread")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Spread Analysis: Top Pairs with Mean-Reversion Bands",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "spread_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")


def plot_parameter_evolution(pairs_df: pd.DataFrame,
                              n_pairs: int = 4,
                              save_dir: str = OUTPUT_DIR):
    """
    Rolling VECM parameter estimates and half-life of mean reversion.
    """
    if pairs_df.empty or "lambda_1" not in pairs_df.columns:
        print("[PLOT] No MLE parameter data to plot.")
        return

    pair_freq = pairs_df.groupby(["Stock_y", "Stock_x"]).size().nlargest(n_pairs)
    top_pairs = pair_freq.index.tolist()

    fig, axes = plt.subplots(n_pairs, 3, figsize=(20, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, (sy, sx) in enumerate(top_pairs):
        sub = pairs_df[(pairs_df["Stock_y"] == sy) & (pairs_df["Stock_x"] == sx)].copy()
        sub = sub.sort_values("End_Date")
        sector = TICKER_TO_SECTOR.get(sy, "")

        # λ₁, λ₂ over time
        axes[i, 0].plot(sub["End_Date"], sub["lambda_1"], marker="o", ms=3,
                        color="steelblue", label="λ₁")
        axes[i, 0].plot(sub["End_Date"], sub["lambda_2"], marker="s", ms=3,
                        color="darkorange", label="λ₂")
        axes[i, 0].axhline(0, color="black", lw=0.5)
        axes[i, 0].set_title(f"{sy}/{sx} ({sector}) — Speed of Adjustment",
                              fontweight="bold", fontsize=10)
        axes[i, 0].set_ylabel("λ estimate")
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # Half-life
        hl = sub["half_life"].clip(upper=500)
        axes[i, 1].plot(sub["End_Date"], hl, marker="D", ms=3,
                        color="seagreen")
        axes[i, 1].set_title(f"{sy}/{sx} — Half-Life of Mean Reversion",
                              fontweight="bold", fontsize=10)
        axes[i, 1].set_ylabel("Half-life (trading days)")
        axes[i, 1].grid(True, alpha=0.3)

        # σ_y, σ_x over time
        axes[i, 2].plot(sub["End_Date"], sub["sigma_y"], marker="o", ms=3,
                        color="steelblue", label="σ_y")
        axes[i, 2].plot(sub["End_Date"], sub["sigma_x"], marker="s", ms=3,
                        color="darkorange", label="σ_x")
        axes[i, 2].set_title(f"{sy}/{sx} — Diffusion Coefficients",
                              fontweight="bold", fontsize=10)
        axes[i, 2].set_ylabel("σ estimate")
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Formation Window End Date")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("VECM Parameter Evolution Over Rolling Windows",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "parameter_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")


def plot_residual_diagnostics(pairs_df: pd.DataFrame,
                               prices: pd.DataFrame,
                               n_pairs: int = 3,
                               save_dir: str = OUTPUT_DIR):
    """
    Residual diagnostics for VECM estimation:
      - ACF/PACF of spread changes
      - Histogram + QQ plot of standardised residuals
      - Ljung-Box p-values
    """
    if pairs_df.empty:
        print("[PLOT] No pairs for residual diagnostics.")
        return

    pair_freq = pairs_df.groupby(["Stock_y", "Stock_x"]).size().nlargest(n_pairs)
    top_pairs = pair_freq.index.tolist()

    fig, axes = plt.subplots(n_pairs, 4, figsize=(22, 5 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, (sy, sx) in enumerate(top_pairs):
        if sy not in prices.columns or sx not in prices.columns:
            continue

        z = np.log(prices[sy] / prices[sx]).dropna()
        dz = z.diff().dropna()
        sector = TICKER_TO_SECTOR.get(sy, "")

        # Standardise
        dz_std = (dz - dz.mean()) / dz.std()

        # ACF
        n_lags = min(40, len(dz) // 2 - 1)
        acf_vals = acf(dz, nlags=n_lags, fft=True)
        axes[i, 0].bar(range(len(acf_vals)), acf_vals, color="steelblue",
                       width=0.6)
        conf = 1.96 / np.sqrt(len(dz))
        axes[i, 0].axhline(conf, color="red", ls="--", alpha=0.5)
        axes[i, 0].axhline(-conf, color="red", ls="--", alpha=0.5)
        axes[i, 0].set_title(f"{sy}/{sx} ({sector}) — ACF",
                              fontweight="bold", fontsize=10)
        axes[i, 0].set_xlabel("Lag")
        axes[i, 0].set_ylabel("Autocorrelation")
        axes[i, 0].grid(True, alpha=0.3)

        # PACF
        pacf_vals = pacf(dz, nlags=n_lags)
        axes[i, 1].bar(range(len(pacf_vals)), pacf_vals, color="darkorange",
                       width=0.6)
        axes[i, 1].axhline(conf, color="red", ls="--", alpha=0.5)
        axes[i, 1].axhline(-conf, color="red", ls="--", alpha=0.5)
        axes[i, 1].set_title(f"{sy}/{sx} — PACF", fontweight="bold",
                              fontsize=10)
        axes[i, 1].set_xlabel("Lag")
        axes[i, 1].set_ylabel("Partial Autocorrelation")
        axes[i, 1].grid(True, alpha=0.3)

        # Histogram + fitted normal
        axes[i, 2].hist(dz_std, bins=50, density=True, color="steelblue",
                        alpha=0.7, edgecolor="white")
        x_range = np.linspace(-4, 4, 200)
        axes[i, 2].plot(x_range, norm.pdf(x_range), color="red", lw=2,
                        label="N(0,1)")
        axes[i, 2].set_title(f"{sy}/{sx} — Standardised Residual Dist.",
                              fontweight="bold", fontsize=10)
        axes[i, 2].set_xlabel("Standardised Δz")
        axes[i, 2].set_ylabel("Density")
        excess_kurt = ((dz_std ** 4).mean() - 3)
        axes[i, 2].text(0.95, 0.95,
                        f"Skew={skew(dz_std):.2f}\nExc.Kurt={excess_kurt:.2f}",
                        transform=axes[i, 2].transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat",
                                              alpha=0.5))
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].grid(True, alpha=0.3)

        # QQ plot
        probplot(dz_std, dist="norm", plot=axes[i, 3])
        axes[i, 3].set_title(f"{sy}/{sx} — QQ Plot",
                              fontweight="bold", fontsize=10)
        axes[i, 3].grid(True, alpha=0.3)

    fig.suptitle("Residual Diagnostics for Spread Changes (Δz_t)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "residual_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")


def plot_sector_pair_heatmap(pairs_df_mle: pd.DataFrame,
                              pairs_df_johansen: pd.DataFrame,
                              save_dir: str = OUTPUT_DIR):
    """
    Heatmaps: number of valid cointegrated pairs per sector over time.
    MLE and Johansen saved as separate PNG files.
    """
    for df, method, fname in [
        (pairs_df_mle,      "MLE",      "sector_heatmap_mle.png"),
        (pairs_df_johansen, "Johansen", "sector_heatmap_johansen.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        if df.empty or "Sector" not in df.columns:
            ax.text(0.5, 0.5, "No sector data", ha="center", va="center")
            ax.set_title(f"{method}: No data")
        else:
            df = df.copy()
            df["Year"] = pd.to_datetime(df["End_Date"]).dt.year
            pivot = df.groupby(["Sector", "Year"]).size().unstack(fill_value=0)
            sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                        linewidths=0.5, cbar_kws={"label": "Number of Pairs"})
            ax.set_title(f"{method}: Cointegrated Pairs per Sector per Year",
                         fontweight="bold", fontsize=11)
            ax.set_xlabel("Year")
            ax.set_ylabel("Sector")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[PLOT] Saved → {path}")
        plt.close(fig)


def plot_performance_decomposition(combined_df: pd.DataFrame,
                                    rates: pd.Series,
                                    save_dir: str = OUTPUT_DIR):
    """
    Performance decomposition:
      Panel A: Cumulative returns + drawdown
      Panel B: Rolling 6-month Sharpe ratio
      Panel C: Distribution of daily returns
      Panel D: Returns by year (violin / box plot)
    """
    if combined_df.empty:
        print("[PLOT] No returns data for performance decomposition.")
        return

    rf_aligned = rates.reindex(combined_df.index).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    cum_ret = (1 + combined_df["Daily_Return"]).cumprod()
    excess  = combined_df["Daily_Return"] - rf_daily

    # --- Figure A: Cumulative return + drawdown ------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_ret.index, cum_ret.values, color="steelblue", lw=1.2)
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    ax_dd = ax.twinx()
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       alpha=0.3, color="salmon", label="Drawdown")
    ax_dd.set_ylabel("Drawdown", color="salmon")
    ax.set_title("Cumulative Returns & Drawdown", fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    max_dd = drawdown.min()
    ax.text(0.02, 0.02, f"Max Drawdown: {max_dd*100:.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path = os.path.join(save_dir, "perf_cum_returns_drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure B: Rolling 6-month Sharpe ------------------------------------
    rolling_window = STEP
    rolling_mean   = excess.rolling(rolling_window).mean() * YEAR
    rolling_vol    = excess.rolling(rolling_window).std() * np.sqrt(YEAR)
    rolling_sharpe = rolling_mean / rolling_vol.clip(lower=1e-8)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="seagreen", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(rolling_sharpe.mean(), color="red", ls="--", alpha=0.5,
               label=f"Mean = {rolling_sharpe.mean():.2f}")
    ax.set_title("Rolling 6-Month Annualised Sharpe Ratio", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "perf_rolling_sharpe.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure C: Daily returns histogram -----------------------------------
    dr = combined_df["Daily_Return"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dr * 100, bins=80, density=True, color="steelblue",
            alpha=0.7, edgecolor="white")
    x_range = np.linspace(dr.min() * 100, dr.max() * 100, 200)
    ax.plot(x_range, norm.pdf(x_range, dr.mean() * 100, dr.std() * 100),
            color="red", lw=2, label="Normal fit")
    ax.set_title("Distribution of Daily Returns", fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.text(0.95, 0.95,
            f"Mean={dr.mean()*100:.3f}%\nStd={dr.std()*100:.3f}%\nSkew={skew(dr):.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "perf_daily_return_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure D: Annual returns boxplot ------------------------------------
    combined_df_copy = combined_df.copy()
    combined_df_copy["Year"] = combined_df_copy.index.year
    year_groups = [grp["Daily_Return"].values * 100
                   for _, grp in combined_df_copy.groupby("Year")]
    year_labels = [str(y) for y in combined_df_copy["Year"].unique()]
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(year_groups, labels=year_labels, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Daily Return Distribution by Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Daily Return (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "perf_annual_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

def plot_zscore_analysis(oos_df: pd.DataFrame,
                          save_dir: str = OUTPUT_DIR):
    """
    Z-score diagnostics — each panel saved as a separate PNG.
    """
    if oos_df.empty or "Z_Score" not in oos_df.columns:
        print("[PLOT] No Z_Score data available.")
        return

    # --- Figure A: Z-score time series for top 5 pairs -----------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    pair_freq = oos_df.groupby(["Stock_y", "Stock_x"]).size().nlargest(5)
    top_pairs = pair_freq.index.tolist()
    for (sy, sx) in top_pairs:
        sub = oos_df[(oos_df["Stock_y"] == sy) & (oos_df["Stock_x"] == sx)].copy()
        sub = sub.sort_values("Date")
        ax.plot(sub["Date"], sub["Z_Score"], lw=0.7, label=f"{sy}/{sx}", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2, color="red", ls="--", alpha=0.5, label="±2σ")
    ax.axhline(-2, color="red", ls="--", alpha=0.5)
    ax.set_title("Z-Score Time Series (Top 5 Pairs)", fontweight="bold")
    ax.set_ylabel("Z-Score")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "zscore_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure B: Z-score distribution across all pairs/dates ----------------
    zs = oos_df["Z_Score"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(zs, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    x_range = np.linspace(zs.quantile(0.001), zs.quantile(0.999), 200)
    ax.plot(x_range, norm.pdf(x_range), color="red", lw=2, label="N(0,1)")
    ax.axvline(zs.mean(), color="black", ls="--", lw=1, label=f"Mean = {zs.mean():.2f}")
    ax.set_title("Z-Score Distribution (All Pairs × Dates)", fontweight="bold")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Density")
    pct_outside = ((zs.abs() > 2).mean() * 100)
    ax.text(0.95, 0.95,
            f"N = {len(zs):,}\nMean = {zs.mean():.3f}\nStd = {zs.std():.3f}\n|z| > 2: {pct_outside:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "zscore_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # --- Figure C: Standard error by pair (top 15) ----------------------------
    if "Z_Score_SE" in oos_df.columns:
        se_by_pair = (
            oos_df.groupby(["Stock_y", "Stock_x"])["Z_Score_SE"]
            .first().sort_values(ascending=False).head(15)
        )
        labels = [f"{sy}/{sx}" for (sy, sx) in se_by_pair.index]
        colors = plt.cm.RdYlGn_r(
            (se_by_pair - se_by_pair.min()) /
            max(se_by_pair.max() - se_by_pair.min(), 1e-8))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(labels, se_by_pair.values, color=colors)
        ax.set_xlabel("Standard Error of Z-Score")
        ax.set_title("Z-Score Standard Error by Pair (Top 15)", fontweight="bold")
        ax.invert_yaxis()
        ax.text(0.95, 0.95,
                f"SE = σ_spread / √n\nMedian SE = {se_by_pair.median():.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "zscore_se_by_pair.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[PLOT] Saved → {path}")
        plt.close(fig)

    # --- Figure D: Z-score distribution conditional on trigger ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    trigger_labels = {0: "Close", 1: "Open"}
    for trig_val, trig_label in trigger_labels.items():
        sub = oos_df[oos_df["Trigger"] == trig_val]["Z_Score"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=50, density=True, alpha=0.5,
                    label=f"{trig_label} (n={len(sub):,}, μ={sub.mean():.2f})")
    hold = oos_df[oos_df["Trigger"].isna()]["Z_Score"].dropna()
    if len(hold) > 0:
        ax.hist(hold, bins=50, density=True, alpha=0.3, color="gray",
                label=f"Hold (n={len(hold):,}, μ={hold.mean():.2f})")
    ax.set_title("Z-Score by Trigger State", fontweight="bold")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "zscore_by_trigger.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# 9B. S&P 500 BENCHMARK, STATISTICAL SIGNIFICANCE, COINT. vs. CORRELATION
# ===========================================================================

def download_benchmark(start: str, end: str, ticker: str = "^GSPC"
                       ) -> pd.Series:
    """
    Download S&P 500 (^GSPC) adjusted-close prices from Yahoo Finance.

    Returns
    -------
    pd.Series  index = DatetimeIndex, values = daily prices
    """
    print(f"\n[BENCH] Downloading benchmark {ticker} ({start} → {end}) ...")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()
    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.dropna()
    print(f"[BENCH] {len(raw)} observations retrieved for {ticker}.")
    return raw


def plot_strategy_comparison(combined_df_options: pd.DataFrame,
                              combined_df_vanilla: pd.DataFrame,
                              rates: pd.Series,
                              benchmark_prices: pd.Series,
                              save_dir: str = OUTPUT_DIR):
    """
    Three-way comparison chart:
      1. Options-based pairs trading strategy (BS deltas, dynamic entry/exit)
      2. Vanilla pairs trading strategy      (fixed size, fixed z-score)
      3. S&P 500 buy-and-hold

    Produces:
      - Cumulative returns overlay
      - Drawdown comparison
      - Rolling 1-year excess return
      - Summary metrics table
    """
    if combined_df_options.empty or combined_df_vanilla.empty or benchmark_prices.empty:
        print("[COMPARE] Missing data for three-way comparison.")
        return

    # -- Find common date range across all three --
    common = (
        combined_df_options.index
        .intersection(combined_df_vanilla.index)
        .intersection(benchmark_prices.index)
    )
    if len(common) < 10:
        print("[COMPARE] Insufficient overlapping dates.")
        return

    opt_ret = combined_df_options.loc[common, "Daily_Return"]
    van_ret = combined_df_vanilla.loc[common, "Daily_Return"]
    bench_p = benchmark_prices.loc[common]
    bench_ret = bench_p.pct_change().fillna(0)

    rf_aligned = rates.reindex(common).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    cum_opt   = (1 + opt_ret).cumprod()
    cum_van   = (1 + van_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()

    n_years = len(common) / YEAR

    def _metrics(daily_ret, label):
        cum   = (1 + daily_ret).cumprod()
        ann_r = cum.iloc[-1] ** (1 / n_years) - 1
        vol   = daily_ret.std() * np.sqrt(YEAR)
        rf_a  = rf_aligned.mean() / 100.0
        sr    = (ann_r - rf_a) / vol if vol > 1e-12 else np.nan
        dd    = (cum / cum.cummax() - 1).min()
        return ann_r, vol, sr, dd

    ann_opt,   vol_opt,   sr_opt,   dd_opt   = _metrics(opt_ret, "Options")
    ann_van,   vol_van,   sr_van,   dd_van   = _metrics(van_ret, "Vanilla")
    ann_bench, vol_bench, sr_bench, dd_bench = _metrics(bench_ret, "S&P 500")

    # ==== Drawdowns ====
    dd_opt_s   = (cum_opt   / cum_opt.cummax()   - 1)
    dd_van_s   = (cum_van   / cum_van.cummax()   - 1)
    dd_bench_s = (cum_bench / cum_bench.cummax() - 1)

    # ---- Panel A: Cumulative Returns ----
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(cum_opt.index, cum_opt.values, color="steelblue",
            lw=1.4, label="Options-Based Strategy")
    ax.plot(cum_van.index, cum_van.values, color="seagreen",
            lw=1.4, label="Vanilla Strategy")
    ax.plot(cum_bench.index, cum_bench.values, color="salmon",
            lw=1.4, label="S&P 500 Buy-and-Hold")
    ax.set_title("Cumulative Returns: Options vs Vanilla vs S&P 500",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("Cumulative Return (×)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(save_dir, "compare_cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # ---- Panel B: Drawdowns ----
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dd_opt_s.index, dd_opt_s.values, 0,
                    alpha=0.35, color="steelblue", label="Options")
    ax.fill_between(dd_van_s.index, dd_van_s.values, 0,
                    alpha=0.35, color="seagreen", label="Vanilla")
    ax.fill_between(dd_bench_s.index, dd_bench_s.values, 0,
                    alpha=0.25, color="salmon", label="S&P 500")
    ax.set_title("Drawdown Comparison", fontweight="bold", fontsize=12)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "compare_drawdowns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # ---- Panel C: Rolling 1-Year Excess Return (vs bench) ----
    excess_opt = pd.Series(opt_ret.values - bench_ret.values, index=common)
    excess_van = pd.Series(van_ret.values - bench_ret.values, index=common)
    roll_opt   = excess_opt.rolling(YEAR).sum()
    roll_van   = excess_van.rolling(YEAR).sum()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(roll_opt.index, roll_opt.values * 100, color="steelblue",
            lw=1.2, label="Options − S&P 500")
    ax.plot(roll_van.index, roll_van.values * 100, color="seagreen",
            lw=1.2, label="Vanilla − S&P 500")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 1-Year Excess Return vs S&P 500", fontweight="bold")
    ax.set_ylabel("Excess Return (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "compare_rolling_excess.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # ---- Panel D: Summary table ----
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    table_data = [
        ["Annualised Return",
         f"{ann_opt*100:.2f}%", f"{ann_van*100:.2f}%", f"{ann_bench*100:.2f}%"],
        ["Annualised Vol",
         f"{vol_opt*100:.2f}%", f"{vol_van*100:.2f}%", f"{vol_bench*100:.2f}%"],
        ["Sharpe Ratio",
         f"{sr_opt:.2f}", f"{sr_van:.2f}", f"{sr_bench:.2f}"],
        ["Max Drawdown",
         f"{dd_opt*100:.1f}%", f"{dd_van*100:.1f}%", f"{dd_bench*100:.1f}%"],
        ["Alpha (vs S&P 500)",
         f"{(ann_opt - ann_bench)*100:.2f}%",
         f"{(ann_van - ann_bench)*100:.2f}%", "—"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Options Strategy", "Vanilla Strategy", "S&P 500"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Strategy Comparison Summary",
                 fontweight="bold", fontsize=12, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "compare_performance_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print("  THREE-WAY STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Options':>12} {'Vanilla':>12} {'S&P 500':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Ann. Return':<30} {ann_opt*100:>11.2f}% {ann_van*100:>11.2f}% {ann_bench*100:>11.2f}%")
    print(f"  {'Ann. Volatility':<30} {vol_opt*100:>11.2f}% {vol_van*100:>11.2f}% {vol_bench*100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<30} {sr_opt:>12.2f} {sr_van:>12.2f} {sr_bench:>12.2f}")
    print(f"  {'Max Drawdown':<30} {dd_opt*100:>11.1f}% {dd_van*100:>11.1f}% {dd_bench*100:>11.1f}%")
    print(f"  {'Alpha vs S&P':<30} {(ann_opt-ann_bench)*100:>11.2f}% {(ann_van-ann_bench)*100:>11.2f}%")
    print(f"{'='*70}")

    return {
        "ann_opt": ann_opt, "ann_van": ann_van, "ann_bench": ann_bench,
        "sr_opt": sr_opt, "sr_van": sr_van, "sr_bench": sr_bench,
        "dd_opt": dd_opt, "dd_van": dd_van, "dd_bench": dd_bench,
    }


def plot_benchmark_comparison(combined_df: pd.DataFrame,
                              rates: pd.Series,
                              benchmark_prices: pd.Series,
                              save_dir: str = OUTPUT_DIR):
    """
    Plot cumulative returns of the pairs trading strategy vs S&P 500
    buy-and-hold, and report annualised alpha.
    """
    if combined_df.empty or benchmark_prices.empty:
        print("[BENCH] No data for benchmark comparison.")
        return

    # Align benchmark to strategy dates
    common = combined_df.index.intersection(benchmark_prices.index)
    if len(common) < 10:
        print("[BENCH] Insufficient overlapping dates.")
        return

    strat_ret = combined_df.loc[common, "Daily_Return"]
    bench_p   = benchmark_prices.loc[common]
    bench_ret = bench_p.pct_change().fillna(0)

    rf_aligned = rates.reindex(common).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    cum_strat = (1 + strat_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()

    # --- Annualised metrics ---
    n_years = len(common) / YEAR
    ann_strat = cum_strat.iloc[-1] ** (1 / n_years) - 1
    ann_bench = cum_bench.iloc[-1] ** (1 / n_years) - 1
    alpha     = ann_strat - ann_bench

    vol_strat = strat_ret.std() * np.sqrt(YEAR)
    vol_bench = bench_ret.std() * np.sqrt(YEAR)

    rf_ann = rf_aligned.mean() / 100.0
    sharpe_strat = (ann_strat - rf_ann) / vol_strat if vol_strat > 1e-12 else np.nan
    sharpe_bench = (ann_bench - rf_ann) / vol_bench if vol_bench > 1e-12 else np.nan

    # --- Drawdown ---
    dd_strat = (cum_strat / cum_strat.cummax() - 1)
    dd_bench = (cum_bench / cum_bench.cummax() - 1)

    # --- Plot (4 separate figures) ---

    # Panel A: Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_strat.index, cum_strat.values, color="steelblue",
            lw=1.2, label="Pairs Trading Strategy")
    ax.plot(cum_bench.index, cum_bench.values, color="salmon",
            lw=1.2, label="S&P 500 Buy-and-Hold")
    ax.set_title("Cumulative Returns: Strategy vs Benchmark", fontweight="bold")
    ax.set_ylabel("Cumulative Return (×)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "bench_cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # Panel B: Drawdowns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dd_strat.index, dd_strat.values, 0,
                    alpha=0.4, color="steelblue", label="Strategy")
    ax.fill_between(dd_bench.index, dd_bench.values, 0,
                    alpha=0.4, color="salmon", label="S&P 500")
    ax.set_title("Drawdown Comparison", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "bench_drawdowns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # Panel C: Rolling 1-year excess return
    excess_daily = strat_ret.values - bench_ret.values
    rolling_excess = pd.Series(excess_daily, index=common).rolling(YEAR).sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_excess.index, rolling_excess.values * 100, color="seagreen", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 1-Year Excess Return (Strategy − S&P 500)", fontweight="bold")
    ax.set_ylabel("Excess Return (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "bench_rolling_excess.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # Panel D: Summary table
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    table_data = [
        ["Annualised Return",  f"{ann_strat*100:.2f}%",  f"{ann_bench*100:.2f}%"],
        ["Annualised Vol",     f"{vol_strat*100:.2f}%",  f"{vol_bench*100:.2f}%"],
        ["Sharpe Ratio",       f"{sharpe_strat:.2f}",    f"{sharpe_bench:.2f}"],
        ["Max Drawdown",       f"{dd_strat.min()*100:.1f}%",
                               f"{dd_bench.min()*100:.1f}%"],
        ["Alpha (vs S&P 500)", f"{alpha*100:.2f}%",      "—"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Pairs Trading", "S&P 500"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for j in range(3):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Summary", fontweight="bold", fontsize=11, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "bench_performance_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Strategy':>12} {'S&P 500':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Ann. Return':<30} {ann_strat*100:>11.2f}% {ann_bench*100:>11.2f}%")
    print(f"  {'Ann. Volatility':<30} {vol_strat*100:>11.2f}% {vol_bench*100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<30} {sharpe_strat:>12.2f} {sharpe_bench:>12.2f}")
    print(f"  {'Max Drawdown':<30} {dd_strat.min()*100:>11.1f}% {dd_bench.min()*100:>11.1f}%")
    print(f"  {'Alpha':<30} {alpha*100:>11.2f}%")

    return {
        "ann_strat": ann_strat, "ann_bench": ann_bench, "alpha": alpha,
        "sharpe_strat": sharpe_strat, "sharpe_bench": sharpe_bench,
        "max_dd_strat": dd_strat.min(), "max_dd_bench": dd_bench.min(),
    }


# ---------------------------------------------------------------------------
#  STATISTICAL SIGNIFICANCE (Newey-West HAC)
# ---------------------------------------------------------------------------

def compute_statistical_significance(combined_df: pd.DataFrame,
                                     rates: pd.Series,
                                     save_dir: str = OUTPUT_DIR) -> dict:
    """
    Test whether mean excess returns are statistically significant using:
      1. Simple t-test
      2. Newey-West HAC standard errors (accounts for serial correlation)

    Saves results to a CSV file and prints a summary.

    Returns
    -------
    dict with significance results
    """
    rf_aligned = rates.reindex(combined_df.index).interpolate().ffill().bfill()
    rf_daily   = (rf_aligned / 100.0) / 365.0

    excess = (combined_df["Daily_Return"] - rf_daily).dropna()
    n = len(excess)

    # --- 1. Simple t-test ---------------------------------------------------
    t_stat_simple, p_val_simple = ttest_1samp(excess, 0)

    # --- 2. Newey-West HAC --------------------------------------------------
    #     Regress excess returns on a constant; use HAC-robust SEs
    y = excess.values
    X = add_constant(np.ones(n))  # just the intercept
    ols_res = OLS(y, X[:, :1]).fit()  # single constant column
    # Newey-West with automatic bandwidth = int(4 * (n/100)^(2/9))
    nw_lags = int(4 * (n / 100) ** (2.0 / 9.0))
    nw_res  = OLS(y, add_constant(np.zeros(n))).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    # The intercept in a regression of excess_ret on constant = mean
    nw_t   = nw_res.tvalues[0]
    nw_p   = nw_res.pvalues[0]
    nw_se  = nw_res.bse[0]

    # --- 3. Sharpe ratio CI (Lo, 2002) --------------------------------------
    mean_daily = excess.mean()
    std_daily  = excess.std()
    sr_daily   = mean_daily / std_daily if std_daily > 1e-12 else np.nan
    sr_annual  = sr_daily * np.sqrt(YEAR)

    # SE of Sharpe ratio ≈ sqrt((1 + 0.5 * SR²) / n)  (Lo, 2002)
    sr_se   = np.sqrt((1 + 0.5 * sr_daily**2) / n) * np.sqrt(YEAR)
    sr_ci_l = sr_annual - 1.96 * sr_se
    sr_ci_u = sr_annual + 1.96 * sr_se

    # --- 4. Durbin-Watson ---------------------------------------------------
    dw_stat = durbin_watson(excess.values)

    # --- Results dict -------------------------------------------------------
    results = {
        "n_obs"              : n,
        "mean_excess_daily"  : mean_daily,
        "mean_excess_annual" : mean_daily * YEAR,
        "t_stat_simple"      : t_stat_simple,
        "p_val_simple"       : p_val_simple,
        "nw_lags"            : nw_lags,
        "nw_se"              : nw_se,
        "nw_t_stat"          : nw_t,
        "nw_p_val"           : nw_p,
        "sharpe_annual"      : sr_annual,
        "sharpe_ci_lower"    : sr_ci_l,
        "sharpe_ci_upper"    : sr_ci_u,
        "durbin_watson"      : dw_stat,
    }

    # Print
    print(f"\n{'='*60}")
    print("  STATISTICAL SIGNIFICANCE OF EXCESS RETURNS")
    print(f"{'='*60}")
    print(f"  Observations:              {n:,}")
    print(f"  Mean daily excess return:  {mean_daily*100:.4f}%")
    print(f"  Mean annual excess return: {mean_daily*YEAR*100:.2f}%")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Simple t-statistic:        {t_stat_simple:.3f}")
    print(f"  Simple p-value:            {p_val_simple:.4e}")
    sig = "***" if p_val_simple < 0.01 else ("**" if p_val_simple < 0.05
           else ("*" if p_val_simple < 0.10 else ""))
    print(f"  Significance:              {sig if sig else 'Not significant'}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Newey-West lags:           {nw_lags}")
    print(f"  Newey-West SE:             {nw_se:.6f}")
    print(f"  Newey-West t-statistic:    {nw_t:.3f}")
    print(f"  Newey-West p-value:        {nw_p:.4e}")
    nw_sig = "***" if nw_p < 0.01 else ("**" if nw_p < 0.05
              else ("*" if nw_p < 0.10 else ""))
    print(f"  NW Significance:           {nw_sig if nw_sig else 'Not significant'}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Annualised Sharpe:         {sr_annual:.3f}")
    print(f"  Sharpe 95% CI:             [{sr_ci_l:.3f}, {sr_ci_u:.3f}]")
    print(f"  Durbin-Watson:             {dw_stat:.3f}")

    # Save to CSV
    sig_df = pd.DataFrame([results])
    path = os.path.join(save_dir, "statistical_significance.csv")
    sig_df.to_csv(path, index=False)
    print(f"\n[SIG] Saved → {path}")

    return results


# ---------------------------------------------------------------------------
#  COINTEGRATION VS. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------

def plot_cointegration_vs_correlation(pairs_df_mle: pd.DataFrame,
                                      pairs_df_johansen: pd.DataFrame,
                                      prices: pd.DataFrame,
                                      save_dir: str = OUTPUT_DIR):
    """
    Compare cointegration test results with simple Pearson correlation.

    Produces a 2×2 dashboard:
      Panel A: Scatter of correlation vs Johansen trace statistic
      Panel B: Venn-style bar chart (high corr only, cointegrated only, both)
      Panel C: Distribution of correlations for cointegrated vs non-coint pairs
      Panel D: Summary statistics table
    """
    # Build all within-sector pairs with both correlation and cointegration info
    all_pairs = []
    for sector, tickers in SP500_SECTORS.items():
        available = [t for t in tickers if t in prices.columns]
        all_pairs.extend(list(itertools.combinations(available, 2)))

    if not all_pairs:
        print("[COINT/CORR] No pairs to analyse.")
        return

    # Use latest formation window for Johansen results
    if not pairs_df_johansen.empty:
        latest_joh = pairs_df_johansen["End_Date"].max()
        joh_latest = pairs_df_johansen[pairs_df_johansen["End_Date"] == latest_joh]
        joh_pairs  = set(zip(joh_latest["Stock_y"], joh_latest["Stock_x"]))
    else:
        joh_pairs = set()

    # Compute pairwise correlations on the last 3 years of log returns
    log_rets = np.log(prices / prices.shift(1)).iloc[-WINDOW:]

    records = []
    for (sy, sx) in all_pairs:
        if sy not in log_rets.columns or sx not in log_rets.columns:
            continue
        corr = log_rets[sy].corr(log_rets[sx])
        is_coint = (sy, sx) in joh_pairs or (sx, sy) in joh_pairs
        sector = TICKER_TO_SECTOR.get(sy, "Unknown")

        # Trace stat (if available)
        trace = np.nan
        if not pairs_df_johansen.empty:
            match = joh_latest[(joh_latest["Stock_y"] == sy) &
                               (joh_latest["Stock_x"] == sx)]
            if match.empty:
                match = joh_latest[(joh_latest["Stock_y"] == sx) &
                                   (joh_latest["Stock_x"] == sy)]
            if not match.empty:
                trace = match["Trace_Stat"].values[0]

        records.append({
            "Stock_y": sy, "Stock_x": sx, "Sector": sector,
            "Correlation": corr, "Cointegrated": is_coint,
            "Trace_Stat": trace,
        })

    df = pd.DataFrame(records)

    # --- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # Panel A: Scatter of correlation vs trace statistic
    coint_df   = df[df["Cointegrated"]]
    nocoint_df = df[~df["Cointegrated"]]
    axes[0, 0].scatter(nocoint_df["Correlation"], nocoint_df["Trace_Stat"],
                       alpha=0.3, s=20, color="gray", label="Not cointegrated")
    axes[0, 0].scatter(coint_df["Correlation"], coint_df["Trace_Stat"],
                       alpha=0.6, s=30, color="steelblue", label="Cointegrated")
    axes[0, 0].axhline(15.49, color="red", ls="--", alpha=0.7,
                       label="95% Trace Crit.")
    axes[0, 0].set_xlabel("Pearson Correlation (log returns)")
    axes[0, 0].set_ylabel("Johansen Trace Statistic")
    axes[0, 0].set_title("A. Correlation vs Cointegration Strength",
                         fontweight="bold", fontsize=11)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Classification bar chart
    high_corr_thresh = 0.5
    high_corr = df["Correlation"].abs() > high_corr_thresh
    coint_flag = df["Cointegrated"]

    cats = {
        "High corr only\n(not cointegrated)": (high_corr & ~coint_flag).sum(),
        "Cointegrated only\n(low corr)":       (~high_corr & coint_flag).sum(),
        "Both\n(high corr + coint)":            (high_corr & coint_flag).sum(),
        "Neither":                              (~high_corr & ~coint_flag).sum(),
    }
    colors = ["salmon", "steelblue", "seagreen", "lightgray"]
    bars = axes[0, 1].bar(cats.keys(), cats.values(), color=colors,
                          edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, cats.values()):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(v), ha="center", fontsize=10, fontweight="bold")
    axes[0, 1].set_title(f"B. Correlation (|ρ|>{high_corr_thresh}) vs "
                         f"Cointegration Classification",
                         fontweight="bold", fontsize=11)
    axes[0, 1].set_ylabel("Number of Pairs")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # Panel C: Correlation distributions
    coint_corrs = df.loc[df["Cointegrated"], "Correlation"].dropna()
    nocoint_corrs = df.loc[~df["Cointegrated"], "Correlation"].dropna()
    if len(coint_corrs) > 0:
        axes[1, 0].hist(coint_corrs, bins=30, density=True, alpha=0.6,
                        color="steelblue", edgecolor="white",
                        label=f"Cointegrated (n={len(coint_corrs)}, "
                              f"μ={coint_corrs.mean():.3f})")
    if len(nocoint_corrs) > 0:
        axes[1, 0].hist(nocoint_corrs, bins=30, density=True, alpha=0.4,
                        color="gray", edgecolor="white",
                        label=f"Not cointegrated (n={len(nocoint_corrs)}, "
                              f"μ={nocoint_corrs.mean():.3f})")
    axes[1, 0].set_xlabel("Pearson Correlation")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("C. Correlation Distributions by Cointegration Status",
                         fontweight="bold", fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: Summary statistics table
    axes[1, 1].axis("off")
    n_total = len(df)
    n_coint = coint_flag.sum()
    n_high  = high_corr.sum()
    mean_corr_coint   = coint_corrs.mean() if len(coint_corrs) > 0 else np.nan
    mean_corr_nocoint = nocoint_corrs.mean() if len(nocoint_corrs) > 0 else np.nan

    table_data = [
        ["Total within-sector pairs",       str(n_total)],
        ["Cointegrated pairs (latest)",      str(n_coint)],
        [f"High corr pairs (|ρ|>{high_corr_thresh})", str(n_high)],
        ["High corr + cointegrated",         str((high_corr & coint_flag).sum())],
        ["High corr but NOT cointegrated",   str((high_corr & ~coint_flag).sum())],
        ["Cointegrated but low corr",        str((~high_corr & coint_flag).sum())],
        ["Mean ρ (cointegrated)",            f"{mean_corr_coint:.3f}"],
        ["Mean ρ (not cointegrated)",        f"{mean_corr_nocoint:.3f}"],
    ]
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    axes[1, 1].set_title("D. Cointegration vs Correlation Summary",
                         fontweight="bold", fontsize=11, pad=20)

    fig.suptitle("Cointegration ≠ Correlation: Diagnostic Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "cointegration_vs_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")


# ===========================================================================
# 9C.  COINTEGRATING VECTOR CONSTRAINT ANALYSIS
#       Compare (1, -1) unit-beta constraint vs Johansen-estimated free β
# ===========================================================================

def analyse_cointegrating_vector_constraint(
    pairs_df_johansen: pd.DataFrame,
    prices: pd.DataFrame,
    window_size: int = WINDOW,
    step_size: int = STEP,
    save_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    For each (pair, window) that passed the Johansen trace test, extract
    the Johansen-estimated cointegrating vector β from the first eigenvector
    and compare:

        Constraint A:  z_t = ln(Y_t) − ln(X_t)                  [β = 1]
        Constraint B:  z_t = ln(Y_t) − β̂ · ln(X_t)             [β = free]

    For each pair-window we report:
      - beta_hat        : Johansen-estimated β (normalised so y-weight = 1)
      - lambda_total_11 : λ₁ + λ₂ from MLE under β = 1  (existing estimate)
      - lambda_total_fb : λ₁ + λ₂ from MLE under free β
      - halflife_11     : ln2 / lambda_total_11
      - halflife_fb     : ln2 / lambda_total_fb
      - adf_pval_11     : ADF p-value on (1,-1) spread
      - adf_pval_fb     : ADF p-value on free-β spread
      - beta_vs_1       : |β̂ − 1|   (how far β̂ deviates from 1)

    Saves:
      - beta_constraint_comparison.csv
      - beta_constraint_scatter.png   (half-life comparison)
      - beta_distribution.png         (distribution of estimated β)

    Returns
    -------
    pd.DataFrame  (one row per valid pair-window)
    """
    if pairs_df_johansen.empty:
        print("[BETA] No Johansen results — skipping constraint analysis.")
        return pd.DataFrame()

    print("\n[BETA] Analysing cointegrating vector constraint (1,-1) vs free β ...")

    csv_path = os.path.join(save_dir, "beta_constraint_comparison.csv")
    if os.path.exists(csv_path):
        print(f"[BETA] Loading cached beta analysis from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        records = []

        # Group by (Stock_y, Stock_x) to iterate windows per pair
        grouped = pairs_df_johansen.groupby(["Stock_y", "Stock_x"])

        for (sy, sx), grp in grouped:
            if sy not in prices.columns or sx not in prices.columns:
                continue

            log_py = np.log(prices[sy])
            log_px = np.log(prices[sx])

            for _, row in grp.iterrows():
                end_date   = pd.to_datetime(row["End_Date"])
                start_date = pd.to_datetime(row["Start_Date"])

                # Locate the window in the price series
                mask = (prices.index >= start_date) & (prices.index <= end_date)
                wy = log_py[mask].values
                wx = log_px[mask].values

                if len(wy) < window_size // 2:
                    continue  # too short

                # ------------------------------------------------------------------
                # Step 1: Re-run Johansen on this window to get eigenvector β
                # ------------------------------------------------------------------
                try:
                    joh_res = coint_johansen(
                        np.column_stack((wy, wx)),
                        det_order=0,
                        k_ar_diff=max(1, int(row.get("Lag_Order", 1))),
                    )
                    # First eigenvector (corresponding to largest eigenvalue)
                    # Shape: (2, r) — columns are eigenvectors
                    evec = joh_res.evec[:, 0]          # [a, b] s.t. a*Y + b*X ~ I(0)
                    if abs(evec[0]) < 1e-12:
                        continue
                    # Normalise so that coefficient on Y = 1
                    beta_hat = -evec[1] / evec[0]      # z = ln Y − β ln X
                except Exception:
                    continue

                # ------------------------------------------------------------------
                # Step 2: Spread under free β and MLE re-estimation
                # ------------------------------------------------------------------
                #  Unit-beta spread (already in pairs_df via estimate_parameters)
                z_11 = wy - wx          # ln Y − ln X

                #  Free-beta spread
                z_fb = wy - beta_hat * wx

                # MLE re-estimation under free β -----------------------------------
                def _estimate_lambda_total(z: np.ndarray) -> float:
                    """Quick OLS estimate of mean-reversion speed from the spread."""
                    dz  = np.diff(z)
                    z_l = z[:-1] - np.mean(z[:-1])
                    if np.sum(z_l ** 2) < 1e-20:
                        return np.nan
                    slope = np.sum(dz * z_l) / np.sum(z_l ** 2)
                    lambda_total = -slope
                    return float(lambda_total)

                lam_11 = _estimate_lambda_total(z_11)
                lam_fb = _estimate_lambda_total(z_fb)

                # ADF p-values on both spreads
                try:
                    adf_pval_11 = adfuller(z_11, maxlag=10)[1]
                except Exception:
                    adf_pval_11 = np.nan
                try:
                    adf_pval_fb = adfuller(z_fb, maxlag=10)[1]
                except Exception:
                    adf_pval_fb = np.nan

                hl_11 = np.log(2) / max(lam_11, 1e-12) if lam_11 > 0 else np.nan
                hl_fb = np.log(2) / max(lam_fb, 1e-12) if lam_fb > 0 else np.nan

                records.append({
                    "Stock_y"       : sy,
                    "Stock_x"       : sx,
                    "Sector"        : TICKER_TO_SECTOR.get(sy, "Unknown"),
                    "End_Date"      : end_date,
                    "beta_hat"      : beta_hat,
                    "beta_vs_1"     : abs(beta_hat - 1.0),
                    "lambda_total_11": lam_11,
                    "lambda_total_fb": lam_fb,
                    "halflife_11"   : hl_11,
                    "halflife_fb"   : hl_fb,
                    "adf_pval_11"   : adf_pval_11,
                    "adf_pval_fb"   : adf_pval_fb,
                })

        if not records:
            print("[BETA] No valid pair-windows for constraint comparison.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"[BETA] Comparison saved → {csv_path}  ({len(df):,} rows)")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  COINTEGRATING VECTOR CONSTRAINT: (1,-1) vs FREE β")
    print(f"{'='*60}")
    print(f"  Pair-windows analysed:       {len(df):,}")
    print(f"  Median β̂:                   {df['beta_hat'].median():.4f}")
    print(f"  Mean  |β̂ − 1|:              {df['beta_vs_1'].mean():.4f}")
    print(f"  % pairs with |β̂−1| > 0.20: "
          f"{(df['beta_vs_1'] > 0.20).mean()*100:.1f}%")
    print(f"  Median half-life (1,-1):     {df['halflife_11'].median():.1f} days")
    print(f"  Median half-life (free β):   {df['halflife_fb'].median():.1f} days")
    mean_adf_11 = df['adf_pval_11'].dropna().mean()
    mean_adf_fb = df['adf_pval_fb'].dropna().mean()
    print(f"  Mean ADF p-val (1,-1):       {mean_adf_11:.4f}")
    print(f"  Mean ADF p-val (free β):     {mean_adf_fb:.4f}")
    print(f"={'='*59}")

    # ------------------------------------------------------------------
    # Plot A: Half-life scatter — (1,-1) vs free β
    # ------------------------------------------------------------------
    hl_clip = 500
    plot_df = df.dropna(subset=["halflife_11", "halflife_fb"]).copy()
    plot_df["halflife_11"] = plot_df["halflife_11"].clip(upper=hl_clip)
    plot_df["halflife_fb"] = plot_df["halflife_fb"].clip(upper=hl_clip)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    ax = axes[0]
    sc = ax.scatter(
        plot_df["halflife_11"],
        plot_df["halflife_fb"],
        c=plot_df["beta_vs_1"],
        cmap="RdYlGn_r",
        alpha=0.5, s=20,
        vmin=0, vmax=0.5,
    )
    lims = [0, hl_clip]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x (identical)")
    plt.colorbar(sc, ax=ax, label="|β̂ − 1|")
    ax.set_xlabel("Half-life under (1,−1) constraint (days)")
    ax.set_ylabel("Half-life under free β (days)")
    ax.set_title("Half-Life: (1,−1) vs Johansen β", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ADF p-value comparison
    ax2 = axes[1]
    ax2.scatter(
        df["adf_pval_11"],
        df["adf_pval_fb"],
        alpha=0.4, s=20, color="steelblue",
    )
    lims2 = [0, 1]
    ax2.plot(lims2, lims2, "k--", lw=1, alpha=0.5, label="y = x")
    ax2.axhline(0.05, color="red",    ls="--", alpha=0.5, lw=1, label="5% level")
    ax2.axvline(0.05, color="orange", ls="--", alpha=0.5, lw=1)
    ax2.set_xlabel("ADF p-value under (1,−1) constraint")
    ax2.set_ylabel("ADF p-value under free β")
    ax2.set_title("Spread Stationarity: (1,−1) vs Johansen β", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Methodological Choice: Cointegrating Vector Constraint\n"
        "β = 1 (dollar-neutral) vs β = Johansen Eigenvector",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "beta_constraint_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot B: Distribution of estimated β
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    beta_vals = df["beta_hat"].dropna()
    # Plot bounded range to avoid visual collapse from extreme outliers
    ax.hist(beta_vals, bins=50, range=(-1.0, 3.0), color="steelblue", edgecolor="white",
            alpha=0.8, density=True)
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label=r"Unit constraint ($\beta=1$)")
    ax.axvline(beta_vals.median(), color="seagreen", ls="-.", lw=1.5,
               label=rf"Median $\hat{{\beta}}$ = {beta_vals.median():.3f}")
    ax.set_xlabel(r"Estimated Johansen $\hat{{\beta}}$")
    ax.set_ylabel("Density")
    ax.set_title(r"Distribution of Johansen-Estimated $\hat{{\beta}}$ (bounded)", fontweight="bold")
    pct_near_1 = (df["beta_vs_1"] < 0.10).mean() * 100
    ax.text(0.95, 0.95,
            rf"N = {len(beta_vals):,}\n"
            rf"Median $\hat{{\beta}}$ = {beta_vals.median():.3f}\n"
            rf"$|\hat{{\beta}}-1| < 0.10$: {pct_near_1:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    dev_vals = df["beta_vs_1"].dropna()
    ax2.hist(dev_vals, bins=50, range=(0.0, 2.5), color="darkorange",
             edgecolor="white", alpha=0.8, density=True)
    ax2.axvline(0.20, color="red", ls="--", lw=1.5,
                label="20% deviation threshold")
    ax2.set_xlabel(r"$|\hat{{\beta}} - 1|$  (deviation from unit constraint)")
    ax2.set_ylabel("Density")
    ax2.set_title(r"Magnitude of $\hat{{\beta}}$ Deviation (bounded)",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Cointegrating Vector β: Empirical Distribution",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "beta_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved → {path}")
    plt.close(fig)

    return df


# ===========================================================================
# 10. MAIN PIPELINE
# ===========================================================================

def run_full_backtest(
    tickers: list      = ALL_TICKERS,
    data_start: str    = DATA_START,
    data_end: str      = DATA_END,
    window_size: int   = WINDOW,
    step_size: int     = STEP,
    reload_pairs: bool = False,
    reload_oos: bool   = False,
) -> dict:
    """
    Execute the full S&P 500 sector-based pairs trading strategy pipeline.

    Parameters
    ----------
    tickers      : list of ticker symbols (default = S&P 500 sector universe)
    data_start   : price download start date (string "YYYY-MM-DD")
    data_end     : price download end date   (string "YYYY-MM-DD")
    window_size  : formation period in trading days (default 756 = 3 yr)
    step_size    : trading period in trading days   (default 126 = 6 mo)
    reload_pairs : if True, re-run MLE/Johansen even if CSVs exist
    reload_oos   : if True, re-run OOS even if CSV exists

    Returns
    -------
    dict with keys:
        prices, impl_vols, rates, pairs_df, pairs_df_johansen,
        oos_df, combined_df, perf, semi_perf
    """

    # -----------------------------------------------------------------------
    # A) DATA DOWNLOAD
    # -----------------------------------------------------------------------
    prices = download_prices(tickers, data_start, data_end)

    # Implied vols: rolling realised vol as historical proxy
    impl_vols = compute_realised_vol(prices, window=IV_PROXY_WINDOW)

    # Risk-free rate
    rates = download_risk_free_rate(data_start, data_end)

    # Align all series to common trading dates
    prices, impl_vols, rates = align_series(prices, impl_vols, rates)

    # -----------------------------------------------------------------------
    # B) PAIR SELECTION — MLE
    # -----------------------------------------------------------------------
    pairs_csv = os.path.join(OUTPUT_DIR, "pairs_MLE_SP500.csv")
    if os.path.exists(pairs_csv) and not reload_pairs:
        print(f"[MLE] Loading cached pairs from {pairs_csv}")
        pairs_df = pd.read_csv(pairs_csv, parse_dates=["Start_Date", "End_Date"])
    else:
        pairs_df = run_mle_pair_selection(prices, window_size, step_size)
        pairs_df.to_csv(pairs_csv, index=False)
        print(f"[MLE] Pairs saved to {pairs_csv}")

    # Pair count statistics
    mle_counts = pairs_df.groupby("End_Date")["Stock_y"].count()
    print("\n[MLE] Pair count statistics:")
    print(mle_counts.agg(["count", "mean", "std", "max", "min"]).to_string())

    # -----------------------------------------------------------------------
    # C) PAIR SELECTION — JOHANSEN (benchmark)
    # -----------------------------------------------------------------------
    joh_csv = os.path.join(OUTPUT_DIR, "pairs_Johansen_SP500.csv")
    if os.path.exists(joh_csv) and not reload_pairs:
        print(f"\n[Johansen] Loading cached pairs from {joh_csv}")
        pairs_df_johansen = pd.read_csv(
            joh_csv, parse_dates=["Start_Date", "End_Date"])
    else:
        pairs_df_johansen = run_johansen_pair_selection(
            prices, window_size, step_size, confidence_level=95)
        pairs_df_johansen.to_csv(joh_csv, index=False)
        print(f"[Johansen] Pairs saved to {joh_csv}")

    joh_counts = pairs_df_johansen.groupby("End_Date")["Stock_y"].count()
    print("\n[Johansen] Pair count statistics:")
    print(joh_counts.agg(["count", "mean", "std", "max", "min"]).to_string())

    # Visualise pair count comparison
    plot_pair_counts(pairs_df, pairs_df_johansen)

    # -----------------------------------------------------------------------
    # D) OUT-OF-SAMPLE PROCESSING
    # -----------------------------------------------------------------------
    oos_csv = os.path.join(OUTPUT_DIR, "oos_MLE_SP500.csv")
    if os.path.exists(oos_csv) and not reload_oos:
        print(f"\n[OOS] Loading cached OOS data from {oos_csv}")
        oos_df = pd.read_csv(oos_csv, parse_dates=["Date"])
    else:
        # Restrict price and IV series to first trading day (formation ends)
        unique_end_dates = sorted(pairs_df["End_Date"].unique())
        first_trading_day = min(unique_end_dates)
        prices_oos    = prices[prices.index >= first_trading_day]
        impl_vols_oos = impl_vols[impl_vols.index >= first_trading_day]
        rates_oos     = rates[rates.index >= first_trading_day]

        oos_df = run_oos_processing(
            pairs_df, prices_oos, impl_vols_oos, rates_oos)

    print(f"\n[OOS] Trigger distribution:\n{oos_df['Trigger'].describe()}")

    # -----------------------------------------------------------------------
    # E0) SIGNAL-STRENGTH FILTERING (reduce overtrading)
    # -----------------------------------------------------------------------
    oos_df = filter_oos_signals(oos_df)

    # -----------------------------------------------------------------------
    # D2) VANILLA OOS PROCESSING
    # -----------------------------------------------------------------------
    oos_van_csv = os.path.join(OUTPUT_DIR, "oos_vanilla_SP500.csv")
    if os.path.exists(oos_van_csv) and not reload_oos:
        print(f"\n[OOS-VANILLA] Loading cached from {oos_van_csv}")
        oos_df_vanilla = pd.read_csv(oos_van_csv, parse_dates=["Date"])
    else:
        oos_df_vanilla = run_oos_processing_vanilla(pairs_df, prices)

    # -----------------------------------------------------------------------
    # E) RETURNS AGGREGATION — OPTIONS STRATEGY
    # -----------------------------------------------------------------------
    print("\n[RETURNS] Aggregating daily returns (options strategy) ...")
    combined_df = aggregate_returns(oos_df)

    if combined_df.empty:
        print("[RETURNS] No returns to aggregate — check trigger logic.")
        return {}

    # -----------------------------------------------------------------------
    # E2) RETURNS AGGREGATION — VANILLA STRATEGY
    # -----------------------------------------------------------------------
    print("[RETURNS] Aggregating daily returns (vanilla strategy) ...")
    combined_df_vanilla = aggregate_returns_vanilla(oos_df_vanilla)

    if combined_df_vanilla.empty:
        print("[RETURNS] No vanilla returns — check trigger logic.")
    else:
        van_ret_csv = os.path.join(OUTPUT_DIR, "combined_returns_vanilla.csv")
        combined_df_vanilla.to_csv(van_ret_csv)
        print(f"[RETURNS] Vanilla returns saved to {van_ret_csv}")

    ret_csv = os.path.join(OUTPUT_DIR, "combined_returns.csv")
    combined_df.to_csv(ret_csv)
    print(f"[RETURNS] Saved to {ret_csv}")

    # -----------------------------------------------------------------------
    # E2) S&P 500 BENCHMARK DOWNLOAD
    # -----------------------------------------------------------------------
    benchmark_prices = download_benchmark(data_start, data_end)

    # -----------------------------------------------------------------------
    # F) PERFORMANCE METRICS
    # -----------------------------------------------------------------------
    print("\n[PERF] Computing performance metrics ...")
    perf      = compute_performance(combined_df, rates)
    semi_perf = compute_semiannual_performance(combined_df, rates)

    print("\n" + "=" * 60)
    print("  ANNUAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(perf["annual"].to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS (annual)")
    print("=" * 60)
    print(perf["summary"].to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 60)
    print("  SEMI-ANNUAL PERFORMANCE")
    print("=" * 60)
    print(semi_perf.to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 60)
    print("  DAILY RETURN STATISTICS")
    print("=" * 60)
    print(f"  Skewness:           {perf['skewness']:.4f}")
    print(f"  Min daily return:   {perf['min_daily']*100:.2f}%")
    print(f"  Max daily return:   {perf['max_daily']*100:.2f}%")

    # -----------------------------------------------------------------------
    # F2) STATISTICAL SIGNIFICANCE (Newey-West HAC)
    # -----------------------------------------------------------------------
    sig_results = compute_statistical_significance(combined_df, rates)

    # -----------------------------------------------------------------------
    # G) PLOTS — ORIGINAL
    # -----------------------------------------------------------------------
    plot_cumulative_returns(combined_df, rates)
    plot_annual_performance(perf["annual"])

    # -----------------------------------------------------------------------
    # G2) BENCHMARK COMPARISON
    # -----------------------------------------------------------------------
    bench_results = plot_benchmark_comparison(combined_df, rates,
                                              benchmark_prices)

    # -----------------------------------------------------------------------
    # G3) THREE-WAY STRATEGY COMPARISON
    # -----------------------------------------------------------------------
    compare_results = None
    if not combined_df_vanilla.empty:
        compare_results = plot_strategy_comparison(
            combined_df, combined_df_vanilla, rates, benchmark_prices)

    # -----------------------------------------------------------------------
    # H) PLOTS — ECONOMETRIC ANALYSIS FIGURES
    # -----------------------------------------------------------------------
    print("\n[PLOTS] Generating econometric analysis figures ...")
    plot_cointegration_diagnostics(pairs_df_johansen, pairs_df)
    plot_spread_analysis(pairs_df, prices)
    plot_parameter_evolution(pairs_df)
    plot_residual_diagnostics(pairs_df, prices)
    plot_sector_pair_heatmap(pairs_df, pairs_df_johansen)
    plot_performance_decomposition(combined_df, rates)
    plot_zscore_analysis(oos_df)

    # -----------------------------------------------------------------------
    # H2) COINTEGRATION VS. CORRELATION ANALYSIS
    # -----------------------------------------------------------------------
    plot_cointegration_vs_correlation(pairs_df, pairs_df_johansen, prices)

    # -----------------------------------------------------------------------
    # H3) COINTEGRATING VECTOR CONSTRAINT ANALYSIS (β = 1 vs free β)
    # -----------------------------------------------------------------------
    print("\n[BETA] Running cointegrating vector constraint analysis ...")
    beta_comparison_df = analyse_cointegrating_vector_constraint(
        pairs_df_johansen, prices, window_size, step_size
    )

    # Save performance tables
    perf["annual"].to_csv(os.path.join(OUTPUT_DIR, "annual_performance.csv"))
    semi_perf.to_csv(os.path.join(OUTPUT_DIR, "semiannual_performance.csv"))

    return {
        "prices"              : prices,
        "impl_vols"           : impl_vols,
        "rates"               : rates,
        "pairs_df"            : pairs_df,
        "pairs_df_johansen"   : pairs_df_johansen,
        "oos_df"              : oos_df,
        "oos_df_vanilla"      : oos_df_vanilla,
        "combined_df"         : combined_df,
        "combined_df_vanilla" : combined_df_vanilla,
        "perf"                : perf,
        "semi_perf"           : semi_perf,
        "benchmark"           : benchmark_prices,
        "bench_results"       : bench_results,
        "compare_results"     : compare_results,
        "sig_results"         : sig_results,
        "beta_comparison_df"  : beta_comparison_df,
    }


# ===========================================================================
# 11. ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    """
    Run the full S&P 500 sector-based pairs trading backtest.

    Typical first run (downloads everything, runs all computations):
        python "pairs_trading_djia-kopi 2.py"

    Subsequent runs re-use cached CSVs from the 'output/' folder unless
    you pass reload_pairs=True or reload_oos=True to run_full_backtest().

    To re-run everything from scratch, delete the 'output/' directory or
    call:
        run_full_backtest(reload_pairs=True, reload_oos=True)
    """

    results = run_full_backtest(
        tickers      = ALL_TICKERS,
        data_start   = DATA_START,
        data_end     = DATA_END,
        window_size  = WINDOW,
        step_size    = STEP,
        reload_pairs = False,   # re-run pair selection fresh
        reload_oos   = False,   # re-run OOS processing fresh
    )

    if results:
        print("\n[DONE] Backtest complete.  "
              f"Results written to '{OUTPUT_DIR}/'")
