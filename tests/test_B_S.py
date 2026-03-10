# tests/test_black_scholes.py
import numpy as np
import math
from src.black_scholes import black_scholes

def approx(a, b, tol=1e-6):
    return abs(a - b) < tol

def test_atm_one_year_5pct_20vol_call_put():
    # Known reference values (classic textbook case)
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    call_ref, put_ref = 10.450583, 5.573526
    call = black_scholes(S, K, T, r, sigma, "call")
    put = black_scholes(S, K, T, r, sigma, "put")
    assert approx(call, call_ref, 1e-5)
    assert approx(put, put_ref, 1e-5)

def test_put_call_parity():
    S, K, T, r, sigma = 120, 110, 0.5, 0.03, 0.25
    call = black_scholes(S, K, T, r, sigma, "call")
    put  = black_scholes(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K*math.exp(-r*T)
    assert abs(lhs - rhs) < 1e-6

def test_degenerate_cases_T0_and_sigma0():
    # ---- T = 0 (expired): price = intrinsic now
    S, K, T, r, sigma = 95, 100, 0.0, 0.05, 0.20
    assert black_scholes(S, K, T, r, sigma, "call") == 0.0
    assert black_scholes(S, K, T, r, sigma, "put")  == 5.0

    # ---- sigma = 0 (zero vol): deterministic forward under risk-neutral measure
    S, K, T, r, sigma = 105, 100, 1.0, 0.05, 0.0
    # call = max(S - K*e^{-rT}, 0)
    # put  = max(K*e^{-rT} - S, 0)
    expected_call = max(S - (K * np.exp(-r*T)), 0.0)
    expected_put  = max(K * np.exp(-r*T) - S, 0.0)
    assert abs(black_scholes(S, K, T, r, sigma, "call") - expected_call) < 1e-12
    assert abs(black_scholes(S, K, T, r, sigma, "put")  - expected_put)  < 1e-12
