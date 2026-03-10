# src/implied_vol.py
import numpy as np
from scipy.optimize import brentq
from src.black_scholes import black_scholes, _validate_inputs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IV_LOWER_BOUND = 1e-6   # Near-zero vol floor for the Brentq bracket
_IV_UPPER_BOUND = 20.0   # 2000% vol ceiling — handles even extreme markets
_IV_TOLERANCE   = 1e-8   # Convergence tolerance on sigma
_IV_MAX_ITER    = 1_000  # Safety cap on Brentq iterations


# ---------------------------------------------------------------------------
# Intrinsic and no-arbitrage bounds (used to reject unsolvable inputs early)
# ---------------------------------------------------------------------------

def _intrinsic_value(S, K, T, r, option_type):
    """
    Lower bound on option price: the discounted intrinsic value.

    For a call: max(S - K*exp(-rT), 0)
    For a put:  max(K*exp(-rT) - S, 0)

    Any market price below this implies arbitrage, so no implied vol exists.
    """
    discount = np.exp(-r * T)
    if option_type == "call":
        return max(S - K * discount, 0.0)
    else:
        return max(K * discount - S, 0.0)


def _upper_bound_price(S, K, T, r, option_type):
    """
    Upper bound on option price.

    A call can never be worth more than the spot S.
    A put can never be worth more than the discounted strike K*exp(-rT).

    Any market price above these implies arbitrage; no implied vol exists.
    """
    if option_type == "call":
        return float(S)
    else:
        return float(K * np.exp(-r * T))


# ---------------------------------------------------------------------------
# Core IV solver
# ---------------------------------------------------------------------------

def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Compute the implied volatility of a European option via Brentq root-finding.

    Implied volatility (IV) is the value of sigma that solves:

        Black-Scholes(S, K, T, r, sigma, option_type) - market_price = 0

    There is no closed-form inverse of Black-Scholes with respect to sigma,
    so we solve numerically. Brentq is used because:

      - It is a bracketed method: we provide [sigma_low, sigma_high] such that
        f(sigma_low) < 0 < f(sigma_high), guaranteeing a root exists in the
        interval.
      - It combines bisection (guaranteed convergence) with secant steps
        (superlinear speed), making it both safe and fast.
      - It converges in ~50 iterations for typical option inputs.

    Parameters
    ----------
    market_price : float
        Observed market price of the option (e.g. from an options chain).
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity in years. Must be > 0 (IV is undefined at expiry).
    r : float
        Continuously compounded risk-free interest rate.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Implied volatility as a decimal (e.g. 0.20 for 20%).

    Raises
    ------
    ValueError
        If market_price violates no-arbitrage bounds, T <= 0, or any
        input parameter is outside its valid domain.
    RuntimeError
        If Brentq fails to converge within _IV_MAX_ITER iterations.
        This should not occur under normal market conditions.

    Notes
    -----
    The vol surface is constructed by calling this function across a grid
    of (strike, maturity) pairs. For deep in-the-money or deep
    out-of-the-money options, the BS price becomes very flat with respect
    to sigma, which can slow convergence — but Brentq's bracketed nature
    ensures it still terminates.

    The returned IV is model-implied under Black-Scholes. It is the
    market's consensus estimate of future realised volatility, adjusted
    for the model's assumptions (lognormal, constant vol, etc.).
    """
    market_price = float(market_price)
    S, K, T, r = float(S), float(K), float(T), float(r)

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Reuse the shared parameter validator (checks S>0, K>0, T>=0, sigma>=0)
    # We pass sigma=0.1 as a dummy — we're not validating sigma here
    _validate_inputs(S, K, T, r, sigma=0.1)

    if T <= 0:
        raise ValueError(
            f"Implied volatility is undefined at or after expiry (T={T}). "
            "IV requires T > 0."
        )

    if market_price <= 0:
        raise ValueError(
            f"market_price must be positive, got {market_price}. "
            "A zero or negative option price has no implied volatility."
        )

    # --- No-arbitrage bound checks ---
    lower = _intrinsic_value(S, K, T, r, option_type)
    upper = _upper_bound_price(S, K, T, r, option_type)

    if market_price <= lower:
        raise ValueError(
            f"market_price {market_price:.6f} is at or below the no-arbitrage "
            f"intrinsic lower bound {lower:.6f}. No implied volatility exists."
        )
    if market_price >= upper:
        raise ValueError(
            f"market_price {market_price:.6f} is at or above the no-arbitrage "
            f"upper bound {upper:.6f}. No implied volatility exists."
        )

    # --- Objective function: BS price minus observed market price ---
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    # --- Verify the bracket contains a sign change ---
    # If f(lower_bound) and f(upper_bound) have the same sign, Brentq cannot
    # proceed — this would indicate an extreme input outside realistic markets.
    f_low  = objective(_IV_LOWER_BOUND)
    f_high = objective(_IV_UPPER_BOUND)

    if f_low * f_high > 0:
        raise ValueError(
            f"Could not bracket a root for implied volatility. "
            f"f({_IV_LOWER_BOUND:.0e}) = {f_low:.6f}, "
            f"f({_IV_UPPER_BOUND}) = {f_high:.6f}. "
            "The market price may be outside the range achievable by Black-Scholes."
        )

    # --- Brentq root-finding ---
    try:
        iv = brentq(
            objective,
            _IV_LOWER_BOUND,
            _IV_UPPER_BOUND,
            xtol=_IV_TOLERANCE,
            maxiter=_IV_MAX_ITER,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Brentq failed to converge: {e}. "
            "This is unexpected — please check your inputs."
        ) from e

    return float(iv)


# ---------------------------------------------------------------------------
# Convenience wrapper: IV from a grid of strikes (used for vol smile/surface)
# ---------------------------------------------------------------------------

def iv_surface_row(market_prices, strikes, S, T, r, option_type="call"):
    """
    Compute implied volatilities across a strip of strikes at a single maturity.

    This is a convenience function for building one row of the volatility
    surface — i.e. the vol smile at a fixed expiry T.

    Parameters
    ----------
    market_prices : array-like
        Observed option prices, one per strike.
    strikes : array-like
        Corresponding strike prices.
    S : float
        Spot price of the underlying.
    T : float
        Time to maturity in years (same for all strikes in this row).
    r : float
        Continuously compounded risk-free rate.
    option_type : str
        "call" or "put".

    Returns
    -------
    list of float or None
        Implied volatility for each strike. Returns None for any strike
        where the IV calculation fails (e.g. deep ITM/OTM with bad prices),
        so the caller can handle missing values gracefully rather than
        crashing the entire surface.

    Notes
    -----
    Call this once per maturity slice when constructing the full 3D
    volatility surface. The vol surface module will loop over maturities
    and stack these rows into a 2D grid.
    """
    ivs = []
    for price, K in zip(market_prices, strikes):
        try:
            iv = implied_volatility(price, S, K, T, r, option_type)
            ivs.append(iv)
        except (ValueError, RuntimeError):
            ivs.append(None)
    return ivs