# tests/test_greeks.py
import pytest
import numpy as np
from src.greeks import delta, gamma, vega, theta, rho, all_greeks
from src.black_scholes import black_scholes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard ATM reference case used throughout
REF = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

def bump_price(param, bump, option_type="call", **kwargs):
    """
    Return BS prices at (param + bump) and (param - bump) for finite
    difference validation.
    """
    params = {**REF, "option_type": option_type, **kwargs}
    up   = {**params, param: params[param] + bump}
    down = {**params, param: params[param] - bump}
    return black_scholes(**up), black_scholes(**down)


def finite_diff(param, bump, option_type="call", **kwargs):
    """Central finite difference: (f(x+h) - f(x-h)) / (2h)."""
    f_up, f_down = bump_price(param, bump, option_type, **kwargs)
    return (f_up - f_down) / (2 * bump)


# ---------------------------------------------------------------------------
# Reference value tests
#
# These use known textbook/calculator values for the ATM case.
# S=100, K=100, T=1, r=0.05, sigma=0.20, call.
# Values verified against standard BS Greek calculators.
# ---------------------------------------------------------------------------

class TestReferenceValues:

    def test_call_delta_reference(self):
        d = delta(**REF, option_type="call")
        assert abs(d - 0.6368) < 1e-3

    def test_put_delta_reference(self):
        d = delta(**REF, option_type="put")
        assert abs(d - (-0.3632)) < 1e-3

    def test_gamma_reference(self):
        # Gamma is same for call and put
        g = gamma(**REF, option_type="call")
        assert abs(g - 0.0188) < 1e-3

    def test_vega_reference(self):
        # Raw vega (per unit of sigma). Divide by 100 for per-1%-vol vega.
        v = vega(**REF, option_type="call")
        assert abs(v - 37.52) < 0.1

    def test_theta_reference(self):
        # Theta per year, should be negative for long call
        t = theta(**REF, option_type="call")
        assert t < 0
        assert abs(t - (-6.414)) < 0.05

    def test_rho_reference(self):
        r_val = rho(**REF, option_type="call")
        assert r_val > 0          # Call rho is always positive
        assert abs(r_val - 53.23) < 0.1


# ---------------------------------------------------------------------------
# Finite difference validation
#
# For every Greek we numerically approximate the derivative using a central
# finite difference and verify the analytical formula agrees. This is the
# standard way to check derivative formulas — it doesn't rely on any
# external reference, just calculus.
#
# The bump sizes are chosen to be small enough for accuracy but large enough
# to avoid floating-point cancellation error.
# ---------------------------------------------------------------------------

class TestFiniteDifference:

    def test_call_delta_vs_fd(self):
        """Delta = dPrice/dS"""
        analytic = delta(**REF, option_type="call")
        numerical = finite_diff("S", bump=0.01, option_type="call")
        assert abs(analytic - numerical) < 1e-4

    def test_put_delta_vs_fd(self):
        analytic = delta(**REF, option_type="put")
        numerical = finite_diff("S", bump=0.01, option_type="put")
        assert abs(analytic - numerical) < 1e-4

    def test_gamma_vs_fd(self):
        """Gamma = d²Price/dS² ≈ (Price(S+h) - 2*Price(S) + Price(S-h)) / h²"""
        h = 0.5
        price_up   = black_scholes(REF["S"] + h, REF["K"], REF["T"],
                                   REF["r"], REF["sigma"], "call")
        price_mid  = black_scholes(REF["S"],     REF["K"], REF["T"],
                                   REF["r"], REF["sigma"], "call")
        price_down = black_scholes(REF["S"] - h, REF["K"], REF["T"],
                                   REF["r"], REF["sigma"], "call")
        numerical = (price_up - 2 * price_mid + price_down) / (h ** 2)
        analytic  = gamma(**REF, option_type="call")
        assert abs(analytic - numerical) < 1e-4

    def test_vega_vs_fd(self):
        """Vega = dPrice/dSigma"""
        analytic  = vega(**REF, option_type="call")
        numerical = finite_diff("sigma", bump=0.0001, option_type="call")
        assert abs(analytic - numerical) < 0.01

    def test_theta_vs_fd(self):
        """
        Theta = dPrice/dT (derivative w.r.t. time to maturity).
        Note: increasing T increases value, so theta > 0 as dPrice/dT.
        We negate to get the conventional 'time decay' sign.
        """
        analytic  = theta(**REF, option_type="call")
        numerical = finite_diff("T", bump=0.0001, option_type="call")
        # Analytical theta is defined as -dPrice/dT in convention
        # So analytical == numerical (both are dPrice/dT here)
        assert abs(analytic + numerical) < 0.01

    def test_rho_vs_fd(self):
        """Rho = dPrice/dr"""
        analytic  = rho(**REF, option_type="call")
        numerical = finite_diff("r", bump=0.0001, option_type="call")
        assert abs(analytic - numerical) < 0.01


# ---------------------------------------------------------------------------
# Structural / mathematical identity tests
# ---------------------------------------------------------------------------

class TestMathematicalIdentities:

    def test_put_call_delta_relationship(self):
        """
        Put-call parity implies: call_delta - put_delta = 1.
        Derivation: C - P = S - K*exp(-rT), differentiate w.r.t. S.
        """
        call_d = delta(**REF, option_type="call")
        put_d  = delta(**REF, option_type="put")
        assert abs((call_d - put_d) - 1.0) < 1e-10

    def test_call_put_gamma_equal(self):
        """Gamma is identical for calls and puts with same parameters."""
        call_g = gamma(**REF, option_type="call")
        put_g  = gamma(**REF, option_type="put")
        assert abs(call_g - put_g) < 1e-12

    def test_call_put_vega_equal(self):
        """Vega is identical for calls and puts with same parameters."""
        call_v = vega(**REF, option_type="call")
        put_v  = vega(**REF, option_type="put")
        assert abs(call_v - put_v) < 1e-12

    def test_delta_bounds_call(self):
        """Call delta must lie strictly in (0, 1)."""
        d = delta(**REF, option_type="call")
        assert 0 < d < 1

    def test_delta_bounds_put(self):
        """Put delta must lie strictly in (-1, 0)."""
        d = delta(**REF, option_type="put")
        assert -1 < d < 0

    def test_gamma_positive(self):
        """Gamma is always positive (long options benefit from large moves)."""
        assert gamma(**REF, option_type="call") > 0
        assert gamma(**REF, option_type="put")  > 0

    def test_vega_positive(self):
        """Vega is always positive for long options."""
        assert vega(**REF, option_type="call") > 0
        assert vega(**REF, option_type="put")  > 0

    def test_call_theta_negative(self):
        """Long call theta is always negative (time decay)."""
        assert theta(**REF, option_type="call") < 0

    def test_call_rho_positive(self):
        """Call rho is always positive (higher rates → higher call value)."""
        assert rho(**REF, option_type="call") > 0

    def test_put_rho_negative(self):
        """Put rho is always negative (higher rates → lower put value)."""
        assert rho(**REF, option_type="put") < 0

    def test_deep_itm_call_delta_approaches_one(self):
        """Deep ITM call: delta → 1 (acts like holding the stock)."""
        d = delta(S=200, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert d > 0.99

    def test_deep_otm_call_delta_approaches_zero(self):
        """Deep OTM call: delta → 0 (unlikely to expire ITM)."""
        d = delta(S=50, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert d < 0.01


# ---------------------------------------------------------------------------
# all_greeks convenience function
# ---------------------------------------------------------------------------

class TestAllGreeks:

    def test_all_greeks_matches_individuals(self):
        """all_greeks() must return values identical to calling each function."""
        g = all_greeks(**REF, option_type="call")
        assert abs(g["delta"] - delta(**REF, option_type="call")) < 1e-12
        assert abs(g["gamma"] - gamma(**REF, option_type="call")) < 1e-12
        assert abs(g["vega"]  - vega( **REF, option_type="call")) < 1e-12
        assert abs(g["theta"] - theta(**REF, option_type="call")) < 1e-12
        assert abs(g["rho"]   - rho(  **REF, option_type="call")) < 1e-12

    def test_all_greeks_returns_dict_with_correct_keys(self):
        g = all_greeks(**REF, option_type="put")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_all_greeks_values_are_floats(self):
        g = all_greeks(**REF, option_type="call")
        for key, val in g.items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_expired_raises(self):
        with pytest.raises(ValueError, match="expiry"):
            delta(S=100, K=100, T=0.0, r=0.05, sigma=0.20, option_type="call")

    def test_zero_vol_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            gamma(S=100, K=100, T=1.0, r=0.05, sigma=0.0, option_type="call")

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="forward")

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError):
            rho(S=-10, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")