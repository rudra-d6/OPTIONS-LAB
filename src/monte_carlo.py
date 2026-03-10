# src/monte_carlo.py
import numpy as np
from src.black_scholes import _validate_inputs


def monte_carlo(S, K, T, r, sigma, n_sim=100_000, option_type="call", seed=42):
    """
    Monte Carlo pricing for European options under the Black-Scholes model.

    Simulates n_sim risk-neutral paths of the underlying using geometric
    Brownian motion and discounts the average terminal payoff:

        Price = exp(-rT) * E[max(S_T - K, 0)]   (call)
        Price = exp(-rT) * E[max(K - S_T, 0)]   (put)

    where the terminal asset price is:

        S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years. T=0 returns intrinsic value.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualised volatility of the underlying.
    n_sim : int
        Number of simulated paths. Default 100,000. More paths reduce
        variance but increase runtime.
    option_type : str
        "call" or "put".
    seed : int or None
        Random seed for the NumPy default_rng generator. Pass None for a
        non-deterministic run (e.g. convergence analysis). Default 42.

    Returns
    -------
    float
        Monte Carlo estimate of the option fair value.

    Raises
    ------
    ValueError
        If S <= 0, K <= 0, T < 0, sigma < 0, or option_type is invalid.

    Notes
    -----
    With a fixed seed and sufficient n_sim, this should agree with
    Black-Scholes to within a few cents for standard European options.
    The standard error of the MC estimate scales as 1/sqrt(n_sim).
    """
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    _validate_inputs(S, K, T, r, sigma)

    if T == 0 or sigma == 0:
        # Degenerate cases: no simulation needed, return closed-form result
        if option_type == "call":
            intrinsic = max(S - K, 0.0)
        else:
            intrinsic = max(K - S, 0.0)
        # If T > 0 (sigma == 0 case), discount the intrinsic forward value
        return np.exp(-r * T) * intrinsic if T > 0 else float(intrinsic)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sim)

    # Simulate terminal asset price under risk-neutral measure (GBM exact solution)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    return float(np.exp(-r * T) * payoff.mean())