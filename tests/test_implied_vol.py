# tests/test_implied_vol.py
import pytest
import numpy as np
from src.black_scholes import black_scholes
from src.implied_vol import implied_volatility, iv_surface_row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_price(sigma, S=100, K=100, T=1.0, r=0.05, option_type="call"):
    """Convenience wrapper to generate a market price from a known sigma."""
    return black_scholes(S, K, T, r, sigma, option_type)


# ---------------------------------------------------------------------------
# Core round-trip tests: BS_price(sigma) → IV solver → recovered sigma
#
# These are the most important tests. We generate a market price from a known
# sigma, pass it to the IV solver, and check we recover the original sigma.
# If round-trip accuracy fails, everything downstream (vol surface, Greeks
# with market-calibrated vol) will be wrong.
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_atm_call_round_trip(self):
        """ATM call: recover sigma to 8 decimal places."""
        sigma_true = 0.20
        price = bs_price(sigma_true, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-7

    def test_atm_put_round_trip(self):
        """ATM put: recover sigma to 8 decimal places."""
        sigma_true = 0.20
        price = bs_price(sigma_true, option_type="put")
        iv = implied_volatility(price, S=100, K=100, T=1.0, r=0.05, option_type="put")
        assert abs(iv - sigma_true) < 1e-7

    def test_itm_call_round_trip(self):
        """In-the-money call (S > K): higher intrinsic, solver must still converge."""
        sigma_true = 0.25
        price = bs_price(sigma_true, S=110, K=100, option_type="call")
        iv = implied_volatility(price, S=110, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-7

    def test_otm_call_round_trip(self):
        """Out-of-the-money call (S < K): mostly time value, tests flat region."""
        sigma_true = 0.30
        price = bs_price(sigma_true, S=90, K=100, option_type="call")
        iv = implied_volatility(price, S=90, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-7

    def test_otm_put_round_trip(self):
        """Out-of-the-money put."""
        sigma_true = 0.18
        price = bs_price(sigma_true, S=110, K=100, option_type="put")
        iv = implied_volatility(price, S=110, K=100, T=1.0, r=0.05, option_type="put")
        assert abs(iv - sigma_true) < 1e-7

    def test_short_maturity_round_trip(self):
        """Short maturity (T=0.1): tests near-expiry where vega gets small."""
        sigma_true = 0.20
        price = bs_price(sigma_true, T=0.1, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=0.1, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-6

    def test_long_maturity_round_trip(self):
        """Long maturity (T=5): LEAPS-style option."""
        sigma_true = 0.20
        price = bs_price(sigma_true, T=5.0, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=5.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-7

    def test_high_vol_round_trip(self):
        """High volatility regime (sigma=0.80, e.g. small-cap or crypto proxy)."""
        sigma_true = 0.80
        price = bs_price(sigma_true, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-6

    def test_low_vol_round_trip(self):
        """Low volatility regime (sigma=0.05, e.g. FX major pair)."""
        sigma_true = 0.05
        price = bs_price(sigma_true, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - sigma_true) < 1e-7

    def test_negative_rate_round_trip(self):
        """Negative risk-free rate (e.g. EUR rates 2014-2022)."""
        sigma_true = 0.20
        price = bs_price(sigma_true, r=-0.01, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=1.0, r=-0.01, option_type="call")
        assert abs(iv - sigma_true) < 1e-7


# ---------------------------------------------------------------------------
# No-arbitrage bound tests
#
# These verify that the solver correctly rejects prices that violate
# no-arbitrage conditions. Such prices have no theoretical IV — they imply
# free money and the BS model has no solution for them.
# ---------------------------------------------------------------------------

class TestNoArbitrageBounds:

    def test_price_below_intrinsic_raises(self):
        """
        A call price below max(S - K*exp(-rT), 0) implies arbitrage.
        For S=100, K=90, T=1, r=0.05: intrinsic ~ 14.27. Price of 1.0 is
        far below this.
        """
        with pytest.raises(ValueError, match="intrinsic lower bound"):
            implied_volatility(1.0, S=100, K=90, T=1.0, r=0.05, option_type="call")

    def test_price_above_spot_raises(self):
        """A call can never be worth more than the spot price."""
        with pytest.raises(ValueError, match="upper bound"):
            implied_volatility(105.0, S=100, K=100, T=1.0, r=0.05, option_type="call")

    def test_zero_price_raises(self):
        """Zero price implies zero IV — but IV is undefined for a zero-price option."""
        with pytest.raises(ValueError, match="positive"):
            implied_volatility(0.0, S=100, K=100, T=1.0, r=0.05, option_type="call")

    def test_negative_price_raises(self):
        """Negative option price is nonsensical."""
        with pytest.raises(ValueError, match="positive"):
            implied_volatility(-5.0, S=100, K=100, T=1.0, r=0.05, option_type="call")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_expired_option_raises(self):
        """T=0 means the option has expired. IV is undefined."""
        with pytest.raises(ValueError, match="expiry"):
            implied_volatility(5.0, S=100, K=100, T=0.0, r=0.05, option_type="call")

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            implied_volatility(5.0, S=100, K=100, T=-1.0, r=0.05, option_type="call")

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            implied_volatility(5.0, S=100, K=100, T=1.0, r=0.05, option_type="binary")

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError):
            implied_volatility(5.0, S=-100, K=100, T=1.0, r=0.05, option_type="call")

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError):
            implied_volatility(5.0, S=100, K=0, T=1.0, r=0.05, option_type="call")


# ---------------------------------------------------------------------------
# iv_surface_row tests
# ---------------------------------------------------------------------------

class TestIVSurfaceRow:

    def test_surface_row_round_trip(self):
        """
        Generate prices across a strike strip from a flat vol of 0.20,
        then recover IVs. All should be ~0.20 (flat vol surface).
        """
        S, T, r, sigma = 100, 1.0, 0.05, 0.20
        strikes = [80, 90, 95, 100, 105, 110, 120]
        prices = [bs_price(sigma, S=S, K=K, T=T, r=r, option_type="call")
                  for K in strikes]

        ivs = iv_surface_row(prices, strikes, S=S, T=T, r=r, option_type="call")

        for K, iv in zip(strikes, ivs):
            assert iv is not None, f"IV returned None for K={K}"
            assert abs(iv - sigma) < 1e-6, f"IV mismatch at K={K}: got {iv:.6f}"

    def test_surface_row_bad_price_returns_none(self):
        """
        If one price in the strip is invalid (e.g. below intrinsic),
        that entry should return None rather than crashing the whole row.
        """
        S, T, r = 100, 1.0, 0.05
        strikes = [90, 100, 110]
        prices  = [0.001, 10.45, 3.50]   # first price is below intrinsic for K=90

        ivs = iv_surface_row(prices, strikes, S=S, T=T, r=r, option_type="call")

        assert ivs[0] is None        # bad price → None
        assert ivs[1] is not None    # valid prices → solved
        assert ivs[2] is not None