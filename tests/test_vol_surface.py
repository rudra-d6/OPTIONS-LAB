# tests/test_vol_surface.py
import pytest
import numpy as np
from src.black_scholes import black_scholes
from src.vol_surface import (
    strike_grid,
    maturity_grid,
    synthetic_surface,
    market_surface,
    fill_nans,
    surface_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REF = dict(S=100.0, r=0.05)


def _bs_price_matrix(strikes, maturities, S, r, sigma, option_type="call"):
    """
    Generate a price matrix from flat-vol Black-Scholes prices.
    Used to test market_surface() round-trip in a controlled way.
    """
    n_T = len(maturities)
    n_K = len(strikes)
    matrix = np.zeros((n_T, n_K))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            matrix[i, j] = black_scholes(S, K, T, r, sigma, option_type)
    return matrix


# ---------------------------------------------------------------------------
# strike_grid tests
# ---------------------------------------------------------------------------

class TestStrikeGrid:

    def test_shape(self):
        strikes = strike_grid(100, n_strikes=21, width=0.40)
        assert strikes.shape == (21,)

    def test_atm_included(self):
        """With odd n_strikes, the middle element should be exactly spot."""
        strikes = strike_grid(100, n_strikes=21, width=0.40)
        mid = strikes[len(strikes) // 2]
        assert abs(mid - 100.0) < 1e-8

    def test_sorted_ascending(self):
        strikes = strike_grid(100, n_strikes=15, width=0.30)
        assert np.all(np.diff(strikes) > 0)

    def test_all_positive(self):
        strikes = strike_grid(50, n_strikes=11, width=0.50)
        assert np.all(strikes > 0)

    def test_width_bounds(self):
        """Outermost strikes should be S*exp(-width) and S*exp(+width)."""
        S, width = 100, 0.30
        strikes = strike_grid(S, n_strikes=11, width=width)
        assert abs(strikes[0]  - S * np.exp(-width)) < 1e-8
        assert abs(strikes[-1] - S * np.exp(+width)) < 1e-8

    def test_invalid_spot_raises(self):
        with pytest.raises(ValueError):
            strike_grid(S=0, n_strikes=11)

    def test_too_few_strikes_raises(self):
        with pytest.raises(ValueError):
            strike_grid(S=100, n_strikes=2)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            strike_grid(S=100, n_strikes=11, width=-0.1)


# ---------------------------------------------------------------------------
# maturity_grid tests
# ---------------------------------------------------------------------------

class TestMaturityGrid:

    def test_default_length(self):
        """Default schedule should return 8 tenors."""
        mats = maturity_grid()
        assert len(mats) == 8

    def test_default_sorted(self):
        mats = maturity_grid()
        assert np.all(np.diff(mats) > 0)

    def test_custom_tenors(self):
        custom = [0.5, 1.0, 2.0]
        mats = maturity_grid(tenors=custom)
        np.testing.assert_array_almost_equal(mats, [0.5, 1.0, 2.0])

    def test_custom_tenors_are_sorted(self):
        """Even if passed unsorted, output should be sorted."""
        mats = maturity_grid(tenors=[2.0, 0.5, 1.0])
        assert np.all(np.diff(mats) > 0)

    def test_zero_maturity_raises(self):
        with pytest.raises(ValueError):
            maturity_grid(tenors=[0.0, 1.0])

    def test_negative_maturity_raises(self):
        with pytest.raises(ValueError):
            maturity_grid(tenors=[-0.5, 1.0])


# ---------------------------------------------------------------------------
# synthetic_surface tests
# ---------------------------------------------------------------------------

class TestSyntheticSurface:

    def setup_method(self):
        self.result = synthetic_surface(**REF, atm_vol=0.20, n_strikes=11,
                                        tenors=[0.25, 0.5, 1.0])

    def test_result_has_required_keys(self):
        keys = {"strikes", "maturities", "moneyness", "iv_grid", "spot"}
        assert keys == set(self.result.keys())

    def test_iv_grid_shape(self):
        assert self.result["iv_grid"].shape == (3, 11)

    def test_no_nans_in_synthetic(self):
        """Parametric model always produces valid vols — no NaN expected."""
        assert not np.any(np.isnan(self.result["iv_grid"]))

    def test_all_ivs_positive(self):
        assert np.all(self.result["iv_grid"] > 0)

    def test_spot_stored_correctly(self):
        assert self.result["spot"] == REF["S"]

    def test_moneyness_atm_is_zero(self):
        """Middle strike in an odd-length grid should have moneyness ≈ 0."""
        moneyness = self.result["moneyness"]
        mid = moneyness[len(moneyness) // 2]
        assert abs(mid) < 1e-8

    def test_atm_vol_approximately_correct(self):
        """
        ATM vol at the shortest maturity should be close to atm_vol + term*sqrt(T).
        With default term=0.02 and T=0.25: expected ≈ 0.20 + 0.02*0.5 = 0.21.
        """
        result = synthetic_surface(**REF, atm_vol=0.20, skew=0.0, smile=0.0,
                                   term=0.02, n_strikes=11, tenors=[0.25])
        atm_idx = 11 // 2
        expected = 0.20 + 0.02 * np.sqrt(0.25)
        assert abs(result["iv_grid"][0, atm_idx] - expected) < 1e-8

    def test_skew_direction(self):
        """
        Negative skew → OTM puts (m < 0) have higher IV than OTM calls (m > 0).
        """
        result = synthetic_surface(**REF, atm_vol=0.20, skew=-0.10, smile=0.0,
                                   term=0.0, n_strikes=11, tenors=[1.0])
        iv_row  = result["iv_grid"][0, :]
        otm_put  = iv_row[0]    # leftmost = lowest strike = OTM put
        otm_call = iv_row[-1]   # rightmost = highest strike = OTM call
        assert otm_put > otm_call

    def test_smile_symmetry(self):
        """
        Zero skew, positive smile → left wing ≈ right wing (symmetric smile).
        """
        result = synthetic_surface(**REF, atm_vol=0.20, skew=0.0, smile=0.50,
                                   term=0.0, n_strikes=11, tenors=[1.0])
        iv_row = result["iv_grid"][0, :]
        # Symmetric: iv[0] ≈ iv[-1], iv[1] ≈ iv[-2], etc.
        assert abs(iv_row[0] - iv_row[-1]) < 1e-8
        assert abs(iv_row[1] - iv_row[-2]) < 1e-8

    def test_invalid_spot_raises(self):
        with pytest.raises(ValueError):
            synthetic_surface(S=0, r=0.05)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            synthetic_surface(S=100, r=0.05, option_type="binary")


# ---------------------------------------------------------------------------
# market_surface tests
# ---------------------------------------------------------------------------

class TestMarketSurface:

    def _setup(self, sigma=0.20, option_type="call"):
        """Generate a flat-vol BS price matrix and corresponding grid."""
        self.strikes    = strike_grid(REF["S"], n_strikes=11, width=0.30)
        self.maturities = maturity_grid(tenors=[0.25, 0.5, 1.0])
        self.prices     = _bs_price_matrix(
            self.strikes, self.maturities, REF["S"], REF["r"], sigma, option_type
        )

    def test_round_trip_flat_vol(self):
        """
        If the market prices come from flat-vol BS at sigma=0.20, the
        extracted IV surface should be uniformly ~0.20.
        """
        self._setup(sigma=0.20)
        result = market_surface(
            self.prices, self.strikes, self.maturities,
            S=REF["S"], r=REF["r"], option_type="call"
        )
        iv = result["iv_grid"]
        valid = iv[~np.isnan(iv)]
        assert len(valid) > 0
        assert np.all(np.abs(valid - 0.20) < 1e-5)

    def test_result_shape(self):
        self._setup()
        result = market_surface(
            self.prices, self.strikes, self.maturities,
            S=REF["S"], r=REF["r"]
        )
        assert result["iv_grid"].shape == (3, 11)

    def test_moneyness_computed_correctly(self):
        self._setup()
        result = market_surface(
            self.prices, self.strikes, self.maturities,
            S=REF["S"], r=REF["r"]
        )
        expected = np.log(self.strikes / REF["S"])
        np.testing.assert_array_almost_equal(result["moneyness"], expected)

    def test_bad_price_gives_nan(self):
        """
        A price of 0.001 for a near-ATM option is below intrinsic and should
        cause the IV solver to fail, giving NaN in the grid.
        """
        self._setup()
        self.prices[1, 5] = 0.001    # Corrupt one ATM price mid-surface
        result = market_surface(
            self.prices, self.strikes, self.maturities,
            S=REF["S"], r=REF["r"]
        )
        assert np.isnan(result["iv_grid"][1, 5])

    def test_shape_mismatch_raises(self):
        self._setup()
        with pytest.raises(ValueError, match="shape"):
            market_surface(
                self.prices[:2, :],    # Wrong: 2 rows but 3 maturities
                self.strikes, self.maturities,
                S=REF["S"], r=REF["r"]
            )

    def test_invalid_option_type_raises(self):
        self._setup()
        with pytest.raises(ValueError, match="option_type"):
            market_surface(
                self.prices, self.strikes, self.maturities,
                S=REF["S"], r=REF["r"], option_type="straddle"
            )


# ---------------------------------------------------------------------------
# fill_nans tests
# ---------------------------------------------------------------------------

class TestFillNans:

    def test_no_nans_unchanged(self):
        grid = np.array([[0.20, 0.21, 0.22], [0.19, 0.20, 0.21]])
        filled = fill_nans(grid)
        np.testing.assert_array_equal(grid, filled)

    def test_interior_nan_interpolated(self):
        """NaN between two valid values should be linearly interpolated."""
        grid = np.array([[0.20, np.nan, 0.24]])
        filled = fill_nans(grid)
        assert abs(filled[0, 1] - 0.22) < 1e-8

    def test_edge_nan_forward_filled(self):
        """NaN at the left edge should be backward-filled from first valid."""
        grid = np.array([[np.nan, 0.20, 0.22]])
        filled = fill_nans(grid)
        assert not np.isnan(filled[0, 0])

    def test_all_nan_row_untouched(self):
        """A row that is entirely NaN cannot be interpolated — leave as NaN."""
        grid = np.array([[np.nan, np.nan, np.nan], [0.20, 0.21, 0.22]])
        filled = fill_nans(grid)
        assert np.all(np.isnan(filled[0, :]))
        assert not np.any(np.isnan(filled[1, :]))

    def test_original_not_mutated(self):
        """fill_nans should return a copy, not modify in place."""
        grid = np.array([[0.20, np.nan, 0.22]])
        original = grid.copy()
        fill_nans(grid)
        np.testing.assert_array_equal(grid, original)


# ---------------------------------------------------------------------------
# surface_stats tests
# ---------------------------------------------------------------------------

class TestSurfaceStats:

    def setup_method(self):
        self.result = synthetic_surface(
            **REF, atm_vol=0.20, skew=-0.10, smile=0.50,
            term=0.02, n_strikes=21, tenors=[1/12, 0.5, 1.0, 2.0]
        )
        self.stats = surface_stats(self.result)

    def test_required_keys(self):
        keys = {"atm_vols", "min_iv", "max_iv", "mean_iv", "skew_1m", "term_slope"}
        assert keys == set(self.stats.keys())

    def test_atm_vols_shape(self):
        assert self.stats["atm_vols"].shape == (4,)

    def test_min_less_than_max(self):
        assert self.stats["min_iv"] < self.stats["max_iv"]

    def test_mean_between_min_max(self):
        assert self.stats["min_iv"] <= self.stats["mean_iv"] <= self.stats["max_iv"]

    def test_negative_skew_gives_negative_skew_stat(self):
        """
        With negative skew parameter, OTM puts should be richer than ATM →
        skew_1m = IV(90% strike) - IV(ATM) should be positive (puts are richer).
        """
        assert self.stats["skew_1m"] > 0

    def test_positive_term_gives_positive_slope(self):
        """Positive term parameter → ATM vol rises with maturity."""
        assert self.stats["term_slope"] > 0