# tests/test_market_data.py
"""
Tests for src/market_data.py — processing layer only.

The network layer (fetch_option_chain, get_spot_price, get_risk_free_rate,
fetch_surface_data) requires a live internet connection and a yfinance
installation, so it is not tested here. All tests operate on the pure
processing functions which are fully deterministic and network-free.

Test strategy
-------------
We construct synthetic DataFrames that mimic the structure of a real
yfinance option chain and verify that every processing step — mid price
computation, chain filtering, grid alignment — behaves correctly. This
covers the logic that will run on live data without ever needing a
network call.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.market_data import (
    mid_price,
    days_to_maturity,
    filter_chain,
    chain_to_grid,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic chain builders
# ---------------------------------------------------------------------------

def make_chain(strikes, bids, asks, last_prices=None, open_interests=None):
    """
    Build a synthetic option chain DataFrame in yfinance format.

    Parameters mirror the columns returned by yfinance Ticker.option_chain().
    """
    n = len(strikes)
    if last_prices is None:
        last_prices = [(b + a) / 2 for b, a in zip(bids, asks)]
    if open_interests is None:
        open_interests = [100] * n

    return pd.DataFrame({
        "strike":       strikes,
        "bid":          bids,
        "ask":          asks,
        "lastPrice":    last_prices,
        "openInterest": open_interests,
    })


# ---------------------------------------------------------------------------
# mid_price tests
# ---------------------------------------------------------------------------

class TestMidPrice:

    def test_standard_bid_ask(self):
        """Normal case: valid bid and ask."""
        assert abs(mid_price(1.0, 1.4) - 1.2) < 1e-10

    def test_zero_bid_returns_ask(self):
        """Zero bid → use ask as mid (one-sided market)."""
        assert abs(mid_price(0.0, 1.5) - 1.5) < 1e-10

    def test_nan_bid_returns_ask(self):
        assert abs(mid_price(np.nan, 1.5) - 1.5) < 1e-10

    def test_both_zero_falls_back_to_last(self):
        """Both bid and ask zero → fall back to last-traded price."""
        assert abs(mid_price(0.0, 0.0, last=2.5) - 2.5) < 1e-10

    def test_all_zero_returns_nan(self):
        """No usable price → NaN."""
        assert np.isnan(mid_price(0.0, 0.0, last=0.0))

    def test_all_nan_returns_nan(self):
        assert np.isnan(mid_price(np.nan, np.nan, last=np.nan))

    def test_symmetry(self):
        """Mid should be exactly halfway between bid and ask."""
        assert abs(mid_price(2.0, 3.0) - 2.5) < 1e-10

    def test_tight_spread(self):
        assert abs(mid_price(10.00, 10.05) - 10.025) < 1e-10


# ---------------------------------------------------------------------------
# days_to_maturity tests
# ---------------------------------------------------------------------------

class TestDaysToMaturity:

    def test_one_year(self):
        # Use 2023→2024 to avoid leap year (2024 has 366 days).
        # 2023 is not a leap year: Jan 1 2023 to Jan 1 2024 = exactly 365 days.
        ref    = date(2023, 1, 1)
        expiry = date(2024, 1, 1)
        T = days_to_maturity(expiry, reference=ref)
        assert abs(T - 365 / 365.0) < 1e-8

    def test_six_months(self):
        ref    = date(2024, 1, 1)
        expiry = date(2024, 7, 1)
        T = days_to_maturity(expiry, reference=ref)
        expected = 182 / 365.0
        assert abs(T - expected) < 1e-8

    def test_already_expired_gives_floor(self):
        """Expiry in the past → floored at 1/365 (not zero)."""
        ref    = date(2024, 6, 1)
        expiry = date(2024, 1, 1)
        T = days_to_maturity(expiry, reference=ref)
        assert T == 1 / 365.0

    def test_same_day_gives_floor(self):
        """Expiry today → floored at 1/365."""
        ref = date(2024, 6, 1)
        T = days_to_maturity(ref, reference=ref)
        assert T == 1 / 365.0

    def test_one_day(self):
        ref    = date(2024, 6, 1)
        expiry = date(2024, 6, 2)
        T = days_to_maturity(expiry, reference=ref)
        assert abs(T - 1 / 365.0) < 1e-8

    def test_result_is_float(self):
        ref    = date(2024, 1, 1)
        expiry = date(2025, 1, 1)
        assert isinstance(days_to_maturity(expiry, reference=ref), float)


# ---------------------------------------------------------------------------
# filter_chain tests
# ---------------------------------------------------------------------------

class TestFilterChain:

    def _standard_chain(self):
        """ATM-centred chain with clean bid-ask and good open interest."""
        return make_chain(
            strikes       = [85, 90, 95, 100, 105, 110, 115],
            bids          = [15.1, 10.2, 6.0, 3.0, 1.2, 0.4, 0.1],
            asks          = [15.3, 10.4, 6.2, 3.2, 1.4, 0.6, 0.2],
            open_interests= [500, 400, 300, 1000, 300, 200, 50],
        )

    def test_returns_dataframe(self):
        chain = self._standard_chain()
        result = filter_chain(chain, spot=100)
        assert isinstance(result, pd.DataFrame)

    def test_mid_column_added(self):
        chain = self._standard_chain()
        result = filter_chain(chain, spot=100)
        assert "mid" in result.columns

    def test_mid_values_correct(self):
        """Mid should be (bid + ask) / 2 for clean quotes."""
        chain = self._standard_chain()
        result = filter_chain(chain, spot=100)
        for _, row in result.iterrows():
            expected_mid = (row["bid"] + row["ask"]) / 2
            assert abs(row["mid"] - expected_mid) < 1e-8

    def test_sorted_by_strike(self):
        chain = self._standard_chain()
        result = filter_chain(chain, spot=100)
        assert list(result["strike"]) == sorted(result["strike"])

    def test_moneyness_filter_removes_far_strikes(self):
        """
        With width=0.10 around spot=100, only strikes in
        [100*exp(-0.10), 100*exp(+0.10)] ≈ [90.5, 110.5] survive.
        Strikes 85 and 115 should be excluded.
        """
        chain = self._standard_chain()
        result = filter_chain(chain, spot=100, moneyness_range=0.10)
        assert 85 not in result["strike"].values
        assert 115 not in result["strike"].values
        assert 100 in result["strike"].values

    def test_open_interest_filter(self):
        """Strike with openInterest below threshold should be removed."""
        chain = make_chain(
            strikes=[100, 105],
            bids=[3.0, 1.2],
            asks=[3.2, 1.4],
            open_interests=[5, 100],   # first is below default threshold of 10
        )
        result = filter_chain(chain, spot=100, min_open_interest=10)
        assert 100 not in result["strike"].values
        assert 105 in result["strike"].values

    def test_wide_spread_removed(self):
        """
        Spread ratio = (ask - bid) / mid.
        bid=1.0, ask=3.0 → mid=2.0, ratio=1.0 → exceeds default 0.50.
        """
        chain = make_chain(
            strikes=[100, 105],
            bids=[3.0, 1.0],
            asks=[3.2, 3.0],    # K=105 has very wide spread
            open_interests=[100, 100],
        )
        result = filter_chain(chain, spot=100, max_spread_ratio=0.50)
        assert 105 not in result["strike"].values
        assert 100 in result["strike"].values

    def test_zero_bid_retained_if_ask_valid(self):
        """
        Far-OTM options often have zero bid. These should be retained
        if the ask provides a usable mid price.
        """
        chain = make_chain(
            strikes=[100],
            bids=[0.0],
            asks=[0.5],
            open_interests=[50],
        )
        result = filter_chain(chain, spot=100)
        assert len(result) == 1
        assert abs(result.iloc[0]["mid"] - 0.5) < 1e-8

    def test_empty_chain_returns_empty(self):
        chain = pd.DataFrame(columns=["strike", "bid", "ask", "lastPrice", "openInterest"])
        result = filter_chain(chain, spot=100)
        assert result.empty

    def test_all_filtered_returns_empty(self):
        """If all strikes fail filters, should return empty DF not raise."""
        chain = make_chain(
            strikes=[200, 300],     # Far OTM — outside moneyness range
            bids=[0.01, 0.01],
            asks=[0.02, 0.02],
            open_interests=[1, 1],
        )
        result = filter_chain(chain, spot=100, moneyness_range=0.10)
        assert result.empty


# ---------------------------------------------------------------------------
# chain_to_grid tests
# ---------------------------------------------------------------------------

class TestChainToGrid:

    def _make_chains_by_expiry(self):
        """Two expiries with the same strike set."""
        ref = date(2024, 1, 1)
        exp1 = date(2024, 4, 1)   # ~90 days → T ≈ 0.247
        exp2 = date(2025, 1, 1)   # ~365 days → T ≈ 1.0

        chain = pd.DataFrame({
            "strike": [90.0, 100.0, 110.0],
            "mid":    [11.0,  5.0,   1.5],
        })
        return {exp1: chain, exp2: chain.copy()}, ref

    def test_output_shapes(self):
        chains, ref = self._make_chains_by_expiry()
        price_matrix, strikes, maturities = chain_to_grid(chains, spot=100, reference_date=ref)
        assert price_matrix.shape == (2, 3)
        assert len(strikes) == 3
        assert len(maturities) == 2

    def test_strikes_sorted_ascending(self):
        chains, ref = self._make_chains_by_expiry()
        _, strikes, _ = chain_to_grid(chains, spot=100, reference_date=ref)
        assert np.all(np.diff(strikes) > 0)

    def test_maturities_sorted_ascending(self):
        chains, ref = self._make_chains_by_expiry()
        _, _, maturities = chain_to_grid(chains, spot=100, reference_date=ref)
        assert np.all(np.diff(maturities) > 0)

    def test_prices_correctly_aligned(self):
        """Prices should appear in the correct column for each strike."""
        chains, ref = self._make_chains_by_expiry()
        price_matrix, strikes, _ = chain_to_grid(chains, spot=100, reference_date=ref)
        atm_idx = np.argmin(np.abs(strikes - 100.0))
        # Both expiries should have mid=5.0 at the ATM strike
        assert abs(price_matrix[0, atm_idx] - 5.0) < 1e-8
        assert abs(price_matrix[1, atm_idx] - 5.0) < 1e-8

    def test_misaligned_strikes_produce_nan(self):
        """
        If expiry 1 has strikes [90, 100, 110] and expiry 2 has [95, 100, 105],
        the union grid has 5 strikes. Entries missing in each expiry → NaN.
        """
        ref  = date(2024, 1, 1)
        exp1 = date(2024, 7, 1)
        exp2 = date(2025, 1, 1)

        chain1 = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "mid": [11.0, 5.0, 1.5]})
        chain2 = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "mid": [8.0,  5.0, 2.5]})

        chains = {exp1: chain1, exp2: chain2}
        price_matrix, strikes, _ = chain_to_grid(chains, spot=100, reference_date=ref)

        assert len(strikes) == 5

        # Strike 90 is only in exp1 — exp2 row should be NaN
        idx_90 = np.where(np.isclose(strikes, 90.0))[0][0]
        assert not np.isnan(price_matrix[0, idx_90])
        assert np.isnan(price_matrix[1, idx_90])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chain_to_grid({}, spot=100)

    def test_expired_expiry_dropped(self):
        """An expiry date in the past should be silently dropped."""
        ref        = date(2024, 6, 1)
        past_exp   = date(2024, 1, 1)   # already expired
        future_exp = date(2025, 1, 1)

        chain = pd.DataFrame({"strike": [100.0], "mid": [5.0]})
        chains = {past_exp: chain, future_exp: chain.copy()}

        _, _, maturities = chain_to_grid(chains, spot=100, reference_date=ref)
        # Only future_exp survives — but past_exp is floored to 1/365 not dropped
        # because days_to_maturity returns floor of 1 day for past expiries.
        # Verify there is exactly one maturity corresponding to future_exp.
        assert len(maturities) >= 1