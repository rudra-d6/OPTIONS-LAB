# src/black_scholes.py
import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, r, sigma):
    """
    Validate inputs shared across pricing functions.

    Raises
    ------
    ValueError
        If any parameter is outside its financially meaningful domain.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if T < 0:
        raise ValueError(f"Time to maturity T must be non-negative, got {T}")
    if sigma < 0:
        raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
    # r can be negative (negative rate environments are real), so no check


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes analytical pricing formula for European options.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years. T=0 returns intrinsic value.
    r : float
        Continuously compounded risk-free interest rate (e.g. 0.05 for 5%).
        May be negative.
    sigma : float
        Annualised volatility of the underlying (e.g. 0.20 for 20%).
        Must be non-negative.
    option_type : str
        "call" for a call option, "put" for a put option.

    Returns
    -------
    float
        Fair value of the option under the Black-Scholes model.

    Raises
    ------
    ValueError
        If S <= 0, K <= 0, T < 0, sigma < 0, or option_type is invalid.

    Notes
    -----
    d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Call price = S*N(d1) - K*exp(-rT)*N(d2)
    Put price  = K*exp(-rT)*N(-d2) - S*N(-d1)

    where N(.) is the standard normal CDF.
    """
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    _validate_inputs(S, K, T, r, sigma)

    if T == 0:
        # Expired: price equals intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if sigma == 0:
        # Zero volatility: asset grows deterministically at r under risk-neutral measure.
        # Forward price F = S * exp(rT), discounted back: call = max(S - K*exp(-rT), 0)
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S, 0.0)

    # Standard Black-Scholes: d1 and d2 are the risk-neutral probability arguments
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
