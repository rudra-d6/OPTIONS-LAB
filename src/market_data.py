# src/market_data.py
"""
Market data integration for the Options Lab volatility surface tool.

This module fetches live option chain data from Yahoo Finance via yfinance,
cleans and filters it, and converts it into the format expected by
market_surface() in vol_surface.py.

Architecture
------------
The module is deliberately split into two layers:

  Network layer  — fetch_option_chain(), get_spot_price(), get_risk_free_rate()
                   These touch the network and cannot be unit tested without
                   mocking. Keep them thin.

  Processing layer — mid_price(), filter_chain(), chain_to_grid(),
                     build_surface_inputs()
                     These are pure functions of their inputs. Fully testable
                     without any network access.

The top-level function fetch_surface_data() composes both layers into one
call for use in the Streamlit app.

Mid-price convention
--------------------
Option prices are computed as the mid-point of the bid-ask spread:
    mid = (bid + ask) / 2

If bid is zero (common for illiquid far-OTM options), the last-traded price
is used as a fallback. If both are zero or NaN, the strike is excluded.
This avoids passing stale or crossed quotes to the IV solver.

Time to maturity convention
---------------------------
T is computed in calendar years using actual day count:
    T = (expiry_date - today).days / 365.0

This is an approximation — a production system would use ACT/365 or ACT/252
(trading days) depending on the convention of the underlying. For equity
options at the accuracy level of a portfolio tool, calendar days / 365 is
standard.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Optional

# yfinance is imported inside functions that use it so that the rest of the
# module (processing layer) remains importable even if yfinance is not
# installed. This allows tests to run without a live dependency.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_OPEN_INTEREST = 10      # Filter out strikes with fewer open contracts
_MIN_VOLUME        = 0       # Volume floor (0 = include all, raise to filter illiquid)
_MAX_SPREAD_RATIO  = 0.50    # Max (ask - bid) / mid ratio — filters crossed/wide quotes
_DAYS_IN_YEAR      = 365.0


# ---------------------------------------------------------------------------
# Processing layer — pure functions, fully testable
# ---------------------------------------------------------------------------

def mid_price(bid: float, ask: float, last: float = np.nan) -> float:
    """
    Compute the mid-point price of an option from bid, ask, and last trade.

    Logic:
      1. If both bid and ask are valid and positive, return (bid + ask) / 2.
      2. If bid is zero/NaN but ask is valid, use ask as the mid (this occurs
         for far-OTM options where the market maker quotes a one-sided market).
      3. Fall back to last-traded price if neither bid nor ask is usable.
      4. Return NaN if no valid price is available.

    Parameters
    ----------
    bid : float
        Bid price from the option chain.
    ask : float
        Ask price from the option chain.
    last : float, optional
        Last-traded price. Used as fallback. Default NaN.

    Returns
    -------
    float
        Best available mid price, or NaN if no valid price exists.
    """
    bid  = float(bid)  if not pd.isna(bid)  else np.nan
    ask  = float(ask)  if not pd.isna(ask)  else np.nan
    last = float(last) if not pd.isna(last) else np.nan

    if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if not np.isnan(ask) and ask > 0:
        return ask
    if not np.isnan(last) and last > 0:
        return last
    return np.nan


def days_to_maturity(expiry: date, reference: Optional[date] = None) -> float:
    """
    Compute time to maturity T in years from today to an expiry date.

    Parameters
    ----------
    expiry : datetime.date
        Option expiry date.
    reference : datetime.date, optional
        Reference date (today). Defaults to date.today(). Exposed as a
        parameter so tests can pass a fixed reference date.

    Returns
    -------
    float
        T in years (calendar days / 365). Returns 0.0 if expiry <= reference.

    Notes
    -----
    A minimum floor of 1/365 (~1 day) is applied to avoid T=0 being passed
    to the IV solver, which would raise a ValueError.
    """
    if reference is None:
        reference = date.today()

    # Handle both date and datetime objects
    if isinstance(expiry, datetime):
        expiry = expiry.date()
    if isinstance(reference, datetime):
        reference = reference.date()

    days = (expiry - reference).days
    T = max(days, 1) / _DAYS_IN_YEAR
    return float(T)


def filter_chain(
    chain: pd.DataFrame,
    spot: float,
    moneyness_range: float = 0.40,
    min_open_interest: int = _MIN_OPEN_INTEREST,
    max_spread_ratio: float = _MAX_SPREAD_RATIO,
) -> pd.DataFrame:
    """
    Filter a raw yfinance option chain to retain only liquid, usable strikes.

    Filters applied in order:
      1. Moneyness range  — keep only strikes within exp(-width) to exp(+width)
                            of spot in log-moneyness terms. Default ±40%.
      2. Open interest    — remove strikes with fewer than min_open_interest
                            contracts outstanding (proxy for liquidity).
      3. Mid price        — compute mid from bid/ask/last; drop rows with NaN.
      4. Spread ratio     — drop strikes where (ask - bid) / mid > threshold.
                            Wide spreads indicate illiquidity or stale quotes.

    Parameters
    ----------
    chain : pd.DataFrame
        Raw option chain as returned by yfinance Ticker.option_chain().calls
        or .puts. Expected columns: strike, bid, ask, lastPrice, openInterest.
    spot : float
        Current spot price of the underlying.
    moneyness_range : float
        Log-moneyness half-width for strike selection. Default 0.40.
    min_open_interest : int
        Minimum open interest to retain a strike. Default 10.
    max_spread_ratio : float
        Maximum (ask - bid) / mid ratio. Default 0.50 (50% spread).

    Returns
    -------
    pd.DataFrame
        Filtered chain with an additional 'mid' column. Sorted by strike.
        Empty DataFrame if no strikes survive filtering.
    """
    if chain.empty:
        return chain.copy()

    df = chain.copy()

    # --- 1. Moneyness filter ---
    lower = spot * np.exp(-moneyness_range)
    upper = spot * np.exp(+moneyness_range)
    df = df[(df["strike"] >= lower) & (df["strike"] <= upper)]

    if df.empty:
        return df

    # --- 2. Open interest filter ---
    if "openInterest" in df.columns:
        df = df[df["openInterest"].fillna(0) >= min_open_interest]

    if df.empty:
        return df

    # --- 3. Compute mid price ---
    last_col = "lastPrice" if "lastPrice" in df.columns else "last"
    df["mid"] = df.apply(
        lambda row: mid_price(
            row.get("bid", np.nan),
            row.get("ask", np.nan),
            row.get(last_col, np.nan),
        ),
        axis=1,
    )
    df = df[df["mid"].notna() & (df["mid"] > 0)]

    if df.empty:
        return df

    # --- 4. Spread ratio filter ---
    # Only applies when bid is a valid positive number. A zero or NaN bid
    # indicates a legitimate one-sided market (common for far-OTM options)
    # and should not be penalised as a wide spread.
    if "bid" in df.columns and "ask" in df.columns:
        has_valid_bid = df["bid"].fillna(0) > 0
        spread        = (df["ask"] - df["bid"]).clip(lower=0)
        spread_ratio  = spread / df["mid"]
        # Keep the row if: bid is not valid (one-sided market) OR spread is tight
        df = df[~has_valid_bid | (spread_ratio <= max_spread_ratio)]

    return df.sort_values("strike").reset_index(drop=True)


def chain_to_grid(
    chains_by_expiry: dict,
    spot: float,
    reference_date: Optional[date] = None,
) -> tuple:
    """
    Convert a dict of filtered option chains to aligned (price_matrix, strikes,
    maturities) arrays for use in market_surface().

    The chains for different expiries typically have different strike sets.
    This function finds the union of all strikes, then aligns each expiry
    chain onto the common strike grid, filling gaps with NaN. The IV solver
    in market_surface() will propagate these NaNs rather than crashing.

    Parameters
    ----------
    chains_by_expiry : dict
        Keys: expiry dates (datetime.date).
        Values: filtered DataFrames with columns ['strike', 'mid'].
    spot : float
        Spot price of the underlying.
    reference_date : datetime.date, optional
        Reference date for T computation. Defaults to today.

    Returns
    -------
    tuple: (price_matrix, strikes, maturities)
        price_matrix : np.ndarray, shape (n_maturities, n_strikes)
            Mid prices aligned to the common strike grid. NaN where no quote.
        strikes : np.ndarray, shape (n_strikes,)
            Common strike grid, sorted ascending.
        maturities : np.ndarray, shape (n_maturities,)
            Times to maturity in years, sorted ascending.

    Notes
    -----
    Expiries that produce T <= 0 (already expired) are silently dropped.
    """
    if not chains_by_expiry:
        raise ValueError("chains_by_expiry is empty — no data to process.")

    # --- Compute T for each expiry, drop expired ---
    valid = {}
    for expiry, df in chains_by_expiry.items():
        T = days_to_maturity(expiry, reference=reference_date)
        if T > 0 and not df.empty:
            valid[T] = df

    if not valid:
        raise ValueError("No valid (T > 0) expiries found in the chain data.")

    maturities = np.array(sorted(valid.keys()))

    # --- Build common strike grid from union of all expiry strikes ---
    all_strikes = set()
    for df in valid.values():
        all_strikes.update(df["strike"].tolist())
    strikes = np.array(sorted(all_strikes))

    # --- Align each expiry onto the common grid ---
    n_T = len(maturities)
    n_K = len(strikes)
    price_matrix = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(maturities):
        df = valid[T]
        for _, row in df.iterrows():
            j = np.searchsorted(strikes, row["strike"])
            if j < n_K and np.isclose(strikes[j], row["strike"], rtol=1e-4):
                price_matrix[i, j] = row["mid"]

    return price_matrix, strikes, maturities


# ---------------------------------------------------------------------------
# Network layer — yfinance calls
# ---------------------------------------------------------------------------

def get_spot_price(ticker: str) -> float:
    """
    Fetch the current spot price for a ticker via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "AAPL", "SPY", "^SPX").

    Returns
    -------
    float
        Current spot price.

    Raises
    ------
    ValueError
        If the ticker is not found or price data is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    t = yf.Ticker(ticker)
    info = t.fast_info

    # fast_info.last_price is the most reliable field across market hours
    price = getattr(info, "last_price", None)
    if price is None or np.isnan(float(price)):
        raise ValueError(
            f"Could not retrieve spot price for '{ticker}'. "
            "Check the ticker symbol and your internet connection."
        )
    return float(price)


def get_risk_free_rate() -> float:
    """
    Fetch an approximate current risk-free rate from the 13-week T-bill yield.

    Uses the Yahoo Finance ticker ^IRX (13-week T-bill annualised yield in %).
    Falls back to a hard-coded default of 0.05 (5%) if the fetch fails, with
    a warning printed to stdout.

    Returns
    -------
    float
        Annualised risk-free rate as a decimal (e.g. 0.052 for 5.2%).
    """
    try:
        import yfinance as yf
        irx = yf.Ticker("^IRX")
        rate_pct = irx.fast_info.last_price
        if rate_pct is not None and not np.isnan(float(rate_pct)):
            return float(rate_pct) / 100.0
    except Exception:
        pass

    print(
        "Warning: Could not fetch live risk-free rate. "
        "Defaulting to r = 0.05 (5%)."
    )
    return 0.05


def fetch_option_chain(
    ticker: str,
    option_type: str = "call",
    max_expiries: int = 8,
    moneyness_range: float = 0.40,
    min_open_interest: int = _MIN_OPEN_INTEREST,
    max_spread_ratio: float = _MAX_SPREAD_RATIO,
    reference_date: Optional[date] = None,
) -> tuple:
    """
    Fetch and clean a live option chain from Yahoo Finance.

    This is the main entry point for the network layer. It fetches up to
    max_expiries expiries, filters each chain using filter_chain(), and
    returns the aligned grid inputs for market_surface().

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    option_type : str
        "call" or "put".
    max_expiries : int
        Maximum number of expiry dates to fetch. Fetching is slow for many
        expiries; 8 covers the standard listed schedule. Default 8.
    moneyness_range : float
        Log-moneyness half-width for strike selection. Default 0.40.
    min_open_interest : int
        Minimum open interest to retain a strike. Default 10.
    max_spread_ratio : float
        Maximum bid-ask spread ratio. Default 0.50.
    reference_date : datetime.date, optional
        Reference date for T computation. Defaults to today.

    Returns
    -------
    tuple: (price_matrix, strikes, maturities, spot, r)
        price_matrix : np.ndarray, shape (n_maturities, n_strikes)
        strikes      : np.ndarray
        maturities   : np.ndarray
        spot         : float
        r            : float

    Raises
    ------
    ValueError
        If no valid option data is found after filtering.
    ImportError
        If yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    spot = get_spot_price(ticker)
    r    = get_risk_free_rate()

    t        = yf.Ticker(ticker)
    expiries = t.options  # tuple of expiry date strings "YYYY-MM-DD"

    if not expiries:
        raise ValueError(
            f"No option expiries found for '{ticker}'. "
            "The ticker may not have listed options."
        )

    expiries = expiries[:max_expiries]

    chains_by_expiry = {}
    for exp_str in expiries:
        try:
            expiry_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            raw = t.option_chain(exp_str)
            chain = raw.calls if option_type == "call" else raw.puts

            filtered = filter_chain(
                chain,
                spot=spot,
                moneyness_range=moneyness_range,
                min_open_interest=min_open_interest,
                max_spread_ratio=max_spread_ratio,
            )

            if not filtered.empty:
                chains_by_expiry[expiry_date] = filtered

        except Exception:
            # Skip any expiry that fails — network blip, bad data, etc.
            continue

    if not chains_by_expiry:
        raise ValueError(
            f"No usable option data found for '{ticker}' after filtering. "
            "Try relaxing moneyness_range, min_open_interest, or max_spread_ratio."
        )

    price_matrix, strikes, maturities = chain_to_grid(
        chains_by_expiry, spot=spot, reference_date=reference_date
    )

    return price_matrix, strikes, maturities, spot, r


# ---------------------------------------------------------------------------
# Top-level convenience function for Streamlit
# ---------------------------------------------------------------------------

def fetch_surface_data(
    ticker: str,
    option_type: str = "call",
    max_expiries: int = 8,
    moneyness_range: float = 0.40,
    min_open_interest: int = _MIN_OPEN_INTEREST,
    max_spread_ratio: float = _MAX_SPREAD_RATIO,
) -> dict:
    """
    Fetch a live implied volatility surface for a given ticker.

    Composes the full pipeline:
      fetch_option_chain() → market_surface() → SurfaceResult

    This is the function called by the Streamlit Market Data tab. It returns
    a SurfaceResult dict ready to pass directly to the Plotly surface plot.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g. "AAPL", "SPY").
    option_type : str
        "call" or "put".
    max_expiries : int
        Number of expiry dates to include. Default 8.
    moneyness_range : float
        Log-moneyness half-width for strike selection. Default 0.40.
    min_open_interest : int
        Minimum open interest filter. Default 10.
    max_spread_ratio : float
        Maximum bid-ask spread ratio. Default 0.50.

    Returns
    -------
    dict
        SurfaceResult: keys strikes, maturities, moneyness, iv_grid, spot.
        Same format as synthetic_surface() and market_surface().

    Raises
    ------
    ValueError
        If no usable data is found after filtering.
    ImportError
        If yfinance is not installed.
    """
    from src.vol_surface import market_surface, fill_nans

    price_matrix, strikes, maturities, spot, r = fetch_option_chain(
        ticker=ticker,
        option_type=option_type,
        max_expiries=max_expiries,
        moneyness_range=moneyness_range,
        min_open_interest=min_open_interest,
        max_spread_ratio=max_spread_ratio,
    )

    result = market_surface(
        price_matrix=price_matrix,
        strikes=strikes,
        maturities=maturities,
        S=spot,
        r=r,
        option_type=option_type,
    )

    # Fill NaNs before returning so the Plotly surface has no holes
    result["iv_grid"] = fill_nans(result["iv_grid"])

    return result