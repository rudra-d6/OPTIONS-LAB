# src/greeks.py
import numpy as np
from scipy.stats import norm
from src.black_scholes import _validate_inputs


# ---------------------------------------------------------------------------
# Internal helper: compute d1, d2 and the standard normal PDF/CDF values
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma):
    """
    Compute d1 and d2 — the core probability arguments of Black-Scholes.

    d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    These are not probabilities themselves but arguments to N(.) and n(.),
    where:
      N(d2) = risk-neutral probability the call expires in-the-money
      N(d1) = delta of the call (share of the replicating portfolio)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _greeks_inputs(S, K, T, r, sigma, option_type):
    """
    Cast, validate, and reject degenerate cases for Greek calculations.

    Returns
    -------
    tuple : (S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2)

    Raises
    ------
    ValueError
        If T <= 0 or sigma <= 0 (Greeks are undefined/degenerate at expiry
        or zero vol), or if any base parameter is invalid.
    """
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    _validate_inputs(S, K, T, r, sigma)

    if T <= 0:
        raise ValueError(
            f"Greeks are undefined at expiry (T={T}). Requires T > 0."
        )
    if sigma <= 0:
        raise ValueError(
            f"Greeks are degenerate at zero volatility (sigma={sigma}). "
            "Requires sigma > 0."
        )

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)    # Standard normal PDF at d1 (used by gamma, vega, theta)
    N_d1 = norm.cdf(d1)    # Standard normal CDF at d1 (used by delta, rho)
    N_d2 = norm.cdf(d2)    # Standard normal CDF at d2 (used by theta, rho)

    return S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2


# ---------------------------------------------------------------------------
# Individual Greek functions
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta: rate of change of option price with respect to spot price.

        Call delta = N(d1)          ∈ (0, 1)
        Put  delta = N(d1) - 1      ∈ (-1, 0)

    Interpretation: a delta of 0.60 means the option price moves ~$0.60
    for every $1 move in the underlying. It also approximates the
    risk-neutral probability that the option expires in-the-money
    (though strictly N(d2) is that probability, not N(d1)).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes inputs.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Delta of the option.
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )
    if option_type == "call":
        return float(N_d1)
    else:
        return float(N_d1 - 1.0)


def gamma(S, K, T, r, sigma, option_type="call"):
    """
    Gamma: rate of change of delta with respect to spot price (second
    derivative of option price with respect to S).

        Gamma = n(d1) / (S * sigma * sqrt(T))

    Gamma is identical for calls and puts (put-call parity implies this).

    Interpretation: high gamma means delta changes rapidly as S moves —
    the option is most sensitive near ATM and near expiry. Gamma risk is
    the primary concern for delta-hedged portfolios.

    Returns
    -------
    float
        Gamma of the option (same for call and put).
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )
    return float(n_d1 / (S * sigma * np.sqrt(T)))


def vega(S, K, T, r, sigma, option_type="call"):
    """
    Vega: rate of change of option price with respect to volatility.

        Vega = S * n(d1) * sqrt(T)

    Convention: vega is often quoted per 1% move in vol (divide by 100).
    This function returns the raw derivative (per unit of sigma).

    Vega is identical for calls and puts.

    Interpretation: vega is highest for ATM options with long maturities.
    A long option position (call or put) always has positive vega — you
    benefit from rising volatility.

    Returns
    -------
    float
        Vega per unit of volatility (e.g. multiply by 0.01 for vega per 1%).
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )
    return float(S * n_d1 * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta: rate of change of option price with respect to time (time decay).

        Call theta = -[S*n(d1)*sigma / (2*sqrt(T))] - r*K*exp(-rT)*N(d2)
        Put  theta = -[S*n(d1)*sigma / (2*sqrt(T))] + r*K*exp(-rT)*N(-d2)

    Convention: theta is the derivative with respect to T (time to maturity),
    so it is negative — option value declines as expiry approaches, all
    else equal.

    The value returned is per year. Divide by 365 for daily theta decay,
    which is how traders typically think about it ("how much do I lose
    overnight?").

    Returns
    -------
    float
        Theta per year (negative for long positions). Divide by 365 for
        daily decay.
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )
    common = -(S * n_d1 * sigma) / (2.0 * np.sqrt(T))
    discount = K * np.exp(-r * T)

    if option_type == "call":
        return float(common - r * discount * N_d2)
    else:
        return float(common + r * discount * norm.cdf(-d2))


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho: rate of change of option price with respect to the risk-free rate.

        Call rho =  K * T * exp(-rT) * N(d2)
        Put  rho = -K * T * exp(-rT) * N(-d2)

    Interpretation: rho is typically the smallest of the Greeks in
    magnitude for short-dated options. It becomes more meaningful for
    long-dated options (LEAPS) where the discount factor has a larger
    effect on present value.

    Returns
    -------
    float
        Rho per unit of r (e.g. multiply by 0.01 for rho per 1% rate move).
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )
    discount = K * T * np.exp(-r * T)

    if option_type == "call":
        return float(discount * N_d2)
    else:
        return float(-discount * norm.cdf(-d2))


# ---------------------------------------------------------------------------
# Convenience function: compute all five Greeks at once
# ---------------------------------------------------------------------------

def all_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all five first-order Greeks in a single pass.

    Rather than calling _greeks_inputs five times, this computes d1, d2
    and the normal values once and passes them through. Useful for the
    Streamlit dashboard which needs to display all Greeks simultaneously.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes inputs.
    option_type : str
        "call" or "put".

    Returns
    -------
    dict with keys: "delta", "gamma", "vega", "theta", "rho"
        Values are floats. Theta is per year; divide by 365 for daily.
        Vega and rho are per unit of sigma and r respectively.

    Examples
    --------
    >>> all_greeks(100, 100, 1.0, 0.05, 0.20, "call")
    {'delta': 0.6368, 'gamma': 0.0188, 'vega': 37.52, 'theta': -6.41, 'rho': 53.23}
    """
    S, K, T, r, sigma, d1, d2, n_d1, N_d1, N_d2 = _greeks_inputs(
        S, K, T, r, sigma, option_type
    )

    discount = K * np.exp(-r * T)
    sqrt_T   = np.sqrt(T)

    # Delta
    if option_type == "call":
        _delta = N_d1
    else:
        _delta = N_d1 - 1.0

    # Gamma (same for call and put)
    _gamma = n_d1 / (S * sigma * sqrt_T)

    # Vega (same for call and put)
    _vega = S * n_d1 * sqrt_T

    # Theta
    common = -(S * n_d1 * sigma) / (2.0 * sqrt_T)
    if option_type == "call":
        _theta = common - r * discount * N_d2
    else:
        _theta = common + r * discount * norm.cdf(-d2)

    # Rho
    rho_factor = K * T * np.exp(-r * T)
    if option_type == "call":
        _rho = rho_factor * N_d2
    else:
        _rho = -rho_factor * norm.cdf(-d2)

    return {
        "delta": float(_delta),
        "gamma": float(_gamma),
        "vega":  float(_vega),
        "theta": float(_theta),
        "rho":   float(_rho),
    }