# src/vol_surface.py
"""
Volatility surface construction and preparation for visualisation.

This module sits on top of implied_vol.py and provides two modes:

  1. Synthetic mode  — generates a realistic vol surface from a parametric
                       skew/smile/term-structure model. Used for demo,
                       testing, and exploring surface shape without live
                       market data.

  2. Market mode     — accepts a 2-D array of observed option prices indexed
                       by (maturity, strike) and extracts the implied vol
                       at each grid point using the Brentq IV solver.

Both modes return the same SurfaceResult dict, which is ready to be passed
directly to the Plotly surface plotting helper.

Coordinate conventions
----------------------
Strikes are expressed both in absolute price terms (K) and as log-moneyness
m = ln(K / S), which is the natural coordinate for vol surface analysis.
ATM corresponds to m = 0; OTM calls have m > 0; OTM puts have m < 0.
"""

import numpy as np
from typing import Optional
from src.implied_vol import implied_volatility


# ---------------------------------------------------------------------------
# Type alias for the surface result dict
# ---------------------------------------------------------------------------
# SurfaceResult = {
#   "strikes"    : np.ndarray  shape (n_strikes,)         absolute strike prices
#   "maturities" : np.ndarray  shape (n_maturities,)      years to expiry
#   "moneyness"  : np.ndarray  shape (n_strikes,)         log(K/S)
#   "iv_grid"    : np.ndarray  shape (n_maturities, n_strikes)  implied vols (NaN where failed)
#   "spot"       : float                                   spot price used
# }


# ---------------------------------------------------------------------------
# Grid generation helpers
# ---------------------------------------------------------------------------

def strike_grid(S: float, n_strikes: int = 21, width: float = 0.40) -> np.ndarray:
    """
    Generate a symmetric grid of strike prices centred on the spot.

    Strikes are spaced linearly in log-moneyness space, which ensures
    equal density of OTM calls and OTM puts in relative terms. This is
    the standard convention for vol surface construction.

    Parameters
    ----------
    S : float
        Current spot price of the underlying.
    n_strikes : int
        Number of strikes. Odd number recommended so ATM (m=0) is included
        exactly. Default 21.
    width : float
        Half-width of the log-moneyness range. Default 0.40, corresponding
        to strikes from S*exp(-0.40) to S*exp(+0.40), i.e. roughly
        60% to 149% of spot for a stock-like underlying.

    Returns
    -------
    np.ndarray
        Array of absolute strike prices, shape (n_strikes,), sorted ascending.

    Examples
    --------
    >>> strike_grid(100, n_strikes=5, width=0.2)
    array([ 81.87,  90.48, 100.  , 110.52, 122.14])
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if n_strikes < 3:
        raise ValueError(f"n_strikes must be at least 3, got {n_strikes}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    log_moneyness = np.linspace(-width, width, n_strikes)
    return S * np.exp(log_moneyness)


def maturity_grid(
    tenors: Optional[list] = None,
) -> np.ndarray:
    """
    Return an array of standard option maturities in years.

    The default tenors approximate the standard listed expiry schedule:
    1 month, 2 months, 3 months, 6 months, 9 months, 1 year, 18 months,
    2 years. These cover the region of the surface where market liquidity
    is highest.

    Parameters
    ----------
    tenors : list of float, optional
        Custom maturities in years. If None, the standard schedule is used.

    Returns
    -------
    np.ndarray
        Sorted array of maturities in years.
    """
    if tenors is None:
        tenors = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0]

    tenors = np.array(sorted(tenors), dtype=float)

    if np.any(tenors <= 0):
        raise ValueError("All maturities must be positive (T > 0).")

    return tenors


# ---------------------------------------------------------------------------
# Parametric synthetic vol model
# ---------------------------------------------------------------------------

def _parametric_iv(
    log_moneyness: float,
    T: float,
    atm_vol: float,
    skew: float,
    smile: float,
    term: float,
) -> float:
    """
    Compute implied vol from a simple parametric skew/smile model.

    The model is a quadratic in log-moneyness with a square-root term
    structure, which is the simplest functional form that replicates
    the three key stylised facts of real vol surfaces:

      1. The vol smile — IV is higher for deep ITM and OTM options than ATM.
      2. The skew — the smile is asymmetric; for equities IV rises more
         steeply for OTM puts (m < 0) than OTM calls (m > 0).
      3. Term structure — short-dated options have different ATM vol than
         long-dated ones.

    The formula is:

        IV(m, T) = (atm_vol + term * sqrt(T)) + skew * m + smile * m^2

    where m = ln(K/S) is log-moneyness and T is time to maturity.

    This is NOT the Heston or SABR model — it is a local approximation
    suitable for generating realistic-looking surfaces for demo purposes.
    For a production vol surface, a proper stochastic vol model (Heston,
    SABR, SVI) would be calibrated to market prices.

    Parameters
    ----------
    log_moneyness : float
        ln(K/S). Zero at ATM, positive for OTM calls.
    T : float
        Time to maturity in years.
    atm_vol : float
        Base ATM volatility at T=0 (e.g. 0.20 for 20%).
    skew : float
        Linear skew parameter. Negative for equity-like skew (OTM puts
        are more expensive than OTM calls). Typical value: -0.10.
    smile : float
        Quadratic curvature (smile). Positive so that both wings are
        elevated relative to ATM. Typical value: 0.05.
    term : float
        Term structure slope. Positive means vol rises with maturity
        (normal regime). Negative means inverted term structure. Typical
        value: 0.02.

    Returns
    -------
    float
        Implied vol, floored at 0.01 to prevent non-positive values at
        extreme strikes.
    """
    iv = (atm_vol + term * np.sqrt(T)) + skew * log_moneyness + smile * log_moneyness ** 2
    return max(float(iv), 0.01)   # floor at 1% to stay in valid domain


# ---------------------------------------------------------------------------
# Synthetic surface (demo / testing mode)
# ---------------------------------------------------------------------------

def synthetic_surface(
    S: float,
    r: float = 0.05,
    atm_vol: float = 0.20,
    skew: float = -0.10,
    smile: float = 0.50,
    term: float = 0.02,
    n_strikes: int = 21,
    width: float = 0.40,
    tenors: Optional[list] = None,
    option_type: str = "call",
) -> dict:
    """
    Build a synthetic implied volatility surface using the parametric model.

    This mode does NOT use observed market prices. Instead it directly
    assigns implied vols from the parametric model, making it useful for:
      - Running the Streamlit app without a live data connection
      - Understanding how skew, smile, and term structure shape the surface
      - Generating test data for the surface construction pipeline

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free rate.
    atm_vol : float
        ATM volatility at the short end (e.g. 0.20 for 20%).
    skew : float
        Skew parameter. Negative = equity-like (put skew). Default -0.10.
    smile : float
        Smile curvature. Positive = convex smile. Default 0.50.
    term : float
        Term structure slope. Default 0.02.
    n_strikes : int
        Number of strikes in the grid. Default 21.
    width : float
        Log-moneyness half-width of the strike range. Default 0.40.
    tenors : list of float, optional
        Maturities in years. Default: standard listed schedule.
    option_type : str
        "call" or "put". Used only to label the result.

    Returns
    -------
    dict
        SurfaceResult with keys: strikes, maturities, moneyness, iv_grid, spot.
        iv_grid has shape (n_maturities, n_strikes). No NaN values in synthetic
        mode since the parametric model always returns a valid vol.
    """
    if S <= 0:
        raise ValueError(f"Spot must be positive, got {S}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    strikes    = strike_grid(S, n_strikes=n_strikes, width=width)
    maturities = maturity_grid(tenors)
    moneyness  = np.log(strikes / S)

    n_T = len(maturities)
    n_K = len(strikes)
    iv_grid = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(maturities):
        for j, (K, m) in enumerate(zip(strikes, moneyness)):
            iv_grid[i, j] = _parametric_iv(m, T, atm_vol, skew, smile, term)

    return {
        "strikes":    strikes,
        "maturities": maturities,
        "moneyness":  moneyness,
        "iv_grid":    iv_grid,
        "spot":       float(S),
    }


# ---------------------------------------------------------------------------
# Market surface (from observed option prices)
# ---------------------------------------------------------------------------

def market_surface(
    price_matrix: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    S: float,
    r: float,
    option_type: str = "call",
) -> dict:
    """
    Build an implied volatility surface from a matrix of observed market prices.

    For each (maturity, strike) grid point the Brentq IV solver is called.
    Points where the solver fails (e.g. due to stale prices, wide bid-ask
    spreads, or deep ITM/OTM options with poor liquidity) are recorded as
    NaN rather than crashing the surface.

    Parameters
    ----------
    price_matrix : np.ndarray, shape (n_maturities, n_strikes)
        Observed option prices. Row i corresponds to maturities[i],
        column j to strikes[j].
    strikes : np.ndarray, shape (n_strikes,)
        Absolute strike prices, sorted ascending.
    maturities : np.ndarray, shape (n_maturities,)
        Times to maturity in years, sorted ascending.
    S : float
        Current spot price.
    r : float
        Risk-free rate.
    option_type : str
        "call" or "put". Should match the option chain being passed.

    Returns
    -------
    dict
        SurfaceResult with keys: strikes, maturities, moneyness, iv_grid, spot.
        iv_grid has shape (n_maturities, n_strikes). NaN where IV extraction
        failed. Use fill_nans() before passing to Plotly if needed.

    Raises
    ------
    ValueError
        If price_matrix shape does not match (len(maturities), len(strikes)).
    """
    price_matrix = np.asarray(price_matrix, dtype=float)
    strikes      = np.asarray(strikes, dtype=float)
    maturities   = np.asarray(maturities, dtype=float)

    if price_matrix.shape != (len(maturities), len(strikes)):
        raise ValueError(
            f"price_matrix shape {price_matrix.shape} does not match "
            f"(n_maturities={len(maturities)}, n_strikes={len(strikes)})."
        )

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    moneyness = np.log(strikes / S)
    n_T, n_K  = len(maturities), len(strikes)
    iv_grid   = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            try:
                iv = implied_volatility(
                    market_price=price_matrix[i, j],
                    S=S, K=K, T=T, r=r,
                    option_type=option_type,
                )
                iv_grid[i, j] = iv
            except (ValueError, RuntimeError):
                # Leave as NaN — caller uses fill_nans() before plotting
                pass

    return {
        "strikes":    strikes,
        "maturities": maturities,
        "moneyness":  moneyness,
        "iv_grid":    iv_grid,
        "spot":       float(S),
    }


# ---------------------------------------------------------------------------
# NaN filling (linear interpolation along strike axis)
# ---------------------------------------------------------------------------

def fill_nans(iv_grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in an IV grid by linear interpolation along the strike axis.

    NaN values arise when the IV solver fails for a particular (K, T) point.
    Plotly's surface plot cannot render NaN values, so they must be filled
    before plotting. Linear interpolation along strikes (within each maturity
    slice) is the simplest defensible approach — it is equivalent to assuming
    the vol smile is locally linear across the affected region.

    Extrapolation at the edges is handled by forward/backward fill (flat
    extrapolation beyond the outermost valid point).

    Parameters
    ----------
    iv_grid : np.ndarray, shape (n_maturities, n_strikes)
        Grid potentially containing NaN values.

    Returns
    -------
    np.ndarray
        Grid with NaN values replaced. If an entire maturity row is NaN
        (no valid IV was found at that expiry), that row is left as NaN
        and a warning comment is noted — such a row should be dropped
        before plotting.
    """
    filled = iv_grid.copy()
    n_T, n_K = filled.shape

    for i in range(n_T):
        row   = filled[i, :]
        valid = ~np.isnan(row)

        if valid.sum() == 0:
            # Entire row is NaN — cannot interpolate, leave for caller to handle
            continue

        if valid.all():
            continue

        x_valid = np.where(valid)[0]
        y_valid = row[valid]
        x_all   = np.arange(n_K)

        # Linear interpolation with flat extrapolation at edges
        filled[i, :] = np.interp(x_all, x_valid, y_valid)

    return filled


# ---------------------------------------------------------------------------
# Surface summary statistics (useful for the Streamlit info panel)
# ---------------------------------------------------------------------------

def surface_stats(result: dict) -> dict:
    """
    Compute summary statistics for a completed vol surface.

    Parameters
    ----------
    result : dict
        SurfaceResult as returned by synthetic_surface() or market_surface().

    Returns
    -------
    dict with keys:
        atm_vols      : np.ndarray  ATM IV at each maturity (nearest-strike)
        min_iv        : float       Global minimum IV on the surface
        max_iv        : float       Global maximum IV on the surface
        mean_iv       : float       Global mean IV
        skew_1m       : float       1-month skew: IV(90% moneyness) - IV(ATM)
        term_slope    : float       ATM vol at longest maturity minus shortest
    """
    iv_grid    = result["iv_grid"]
    strikes    = result["strikes"]
    maturities = result["maturities"]
    S          = result["spot"]

    # ATM vol at each maturity: take the strike closest to spot
    atm_idx  = np.argmin(np.abs(strikes - S))
    atm_vols = iv_grid[:, atm_idx]

    valid = iv_grid[~np.isnan(iv_grid)]

    # 1-month skew: difference between 90% moneyness strike IV and ATM IV
    # (proxy for equity put skew — lower strike has higher IV in skewed surface)
    skew_1m = np.nan
    if len(maturities) > 0:
        T_1m_idx = np.argmin(np.abs(maturities - 1/12))
        otm_put_idx = np.argmin(np.abs(strikes - 0.90 * S))
        skew_1m = float(
            iv_grid[T_1m_idx, otm_put_idx] - iv_grid[T_1m_idx, atm_idx]
        ) if not np.isnan(iv_grid[T_1m_idx, otm_put_idx]) else np.nan

    # Term structure slope: longest ATM vol minus shortest ATM vol
    term_slope = np.nan
    valid_atm  = atm_vols[~np.isnan(atm_vols)]
    if len(valid_atm) >= 2:
        term_slope = float(valid_atm[-1] - valid_atm[0])

    return {
        "atm_vols":   atm_vols,
        "min_iv":     float(np.nanmin(iv_grid)) if valid.size > 0 else np.nan,
        "max_iv":     float(np.nanmax(iv_grid)) if valid.size > 0 else np.nan,
        "mean_iv":    float(np.nanmean(iv_grid)) if valid.size > 0 else np.nan,
        "skew_1m":    skew_1m,
        "term_slope": term_slope,
    }