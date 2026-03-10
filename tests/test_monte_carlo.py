# tests/test_monte_carlo.py
from src.black_scholes import black_scholes
from src.monte_carlo import monte_carlo

def test_mc_converges_to_bs():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    bs_call = black_scholes(S, K, T, r, sigma, "call")
    mc_call = monte_carlo(S, K, T, r, sigma, n_sim=300_000, option_type="call", seed=7)
    # Allow a small Monte Carlo error band
    assert abs(mc_call - bs_call) < 0.05
