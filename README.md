# ⬡ Options Lab

> **An interactive options pricing and volatility analysis tool built in Python and deployed via Streamlit.**
> Black-Scholes · Monte Carlo · Implied Volatility · Greeks · 3D Vol Surface · Live Market Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-3b8eea?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-f0c040?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-129%20passing-3dd68c?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/license-MIT-8090a8?style=flat-square)](LICENSE)

---

## Overview

Options Lab is a professional-grade quantitative finance tool that bridges the gap between academic pricing theory and practical market intuition. It started as a simple Black-Scholes vs Monte Carlo comparison tool and was extended into a full five-tab analytical platform covering pricing, risk sensitivities, implied volatility, the volatility surface, and live market data integration.


---

## Live Demo

🔗 **[options-lab.streamlit.app](https://options-lab-rudradubey.streamlit.app/)** 

---

## Features

| Tab | What it does |
|-----|-------------|
| **01 · Pricer** | Black-Scholes analytical price vs Monte Carlo simulation side by side. Sensitivity charts (price vs spot, price vs vol). Live payoff diagram with breakeven marked. |
| **02 · Greeks** | All five Greeks (Δ, Γ, ν, Θ, ρ) in trader-conventional units. Delta/Gamma vs spot, Vega vs spot, theta decay curves, delta heatmap across a spot × volatility grid. |
| **03 · Vol Surface** | Interactive 3D implied volatility surface with adjustable skew, smile, and term structure. Vol smile slices across all maturities. ATM term structure chart. |
| **04 · Convergence** | MC convergence visualiser across multiple seeds with ±1σ envelope. Log-log error vs N chart with theoretical 1/√N reference line. |
| **05 · Market Data** | Live option chain fetch via yfinance. Four-stage liquidity filter. Brentq IV extraction across the full strike-maturity grid. Real 3D vol surface from market prices. |

---

## Project Structure

```
options-lab/
│
├── src/
│   ├── black_scholes.py      # Analytical pricer — BS formula with edge case handling
│   ├── monte_carlo.py        # GBM simulation pricer with fixed-seed reproducibility
│   ├── implied_vol.py        # IV solver via scipy Brentq — no-arbitrage bounds checked
│   ├── greeks.py             # Analytical Greeks: Delta, Gamma, Vega, Theta, Rho
│   ├── vol_surface.py        # Synthetic + market surface construction, NaN filling
│   └── market_data.py        # yfinance integration — fetch, filter, align, extract IV
│
├── tests/
│   ├── test_black_scholes.py # 4 tests — reference values, put-call parity, edge cases
│   ├── test_monte_carlo.py   # 1 test  — convergence to BS price
│   ├── test_implied_vol.py   # 13 tests — round-trip, no-arbitrage bounds, validation
│   ├── test_greeks.py        # 27 tests — reference values, finite difference, identities
│   ├── test_vol_surface.py   # 33 tests — grid, synthetic, market, fill_nans, stats
│   └── test_market_data.py   # 29 tests — processing layer (network-free)
│
├── app.py                    # Streamlit app — five-tab UI wiring all src/ modules
├── conftest.py               # pytest path fix — adds project root to sys.path
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/rudra-d6/options-lab.git
cd options-lab
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Run the test suite

```bash
pytest tests/ -v
```

Expected output: **129 passed** across 6 test modules.

---

## Requirements

```
streamlit
numpy
scipy
pandas
plotly
yfinance
pytest
```

See `requirements.txt` for pinned versions.

---

## Technical Design

### Pricing Models

**Black-Scholes** (`src/black_scholes.py`)

The closed-form analytical solution for European option prices under the risk-neutral measure:

```
Call = S·N(d1) − K·e^{-rT}·N(d2)
Put  = K·e^{-rT}·N(−d2) − S·N(−d1)

d1 = [ln(S/K) + (r + 0.5σ²)T] / (σ√T)
d2 = d1 − σ√T
```

Edge cases handled: T=0 (returns intrinsic value), σ=0 (deterministic forward pricing), negative rates (supported — ECB/BoJ precedent).

**Monte Carlo** (`src/monte_carlo.py`)

Simulates n risk-neutral GBM paths and discounts the average terminal payoff:

```
S_T = S · exp((r − 0.5σ²)T + σ√T · Z),   Z ~ N(0,1)
Price = e^{-rT} · E[payoff(S_T)]
```

Uses `numpy.random.default_rng` with an exposed seed parameter for reproducible tests. Standard error scales as 1/√n.

### Implied Volatility (`src/implied_vol.py`)

IV is the σ that solves `BS(σ) − market_price = 0`. No closed-form inverse exists so the root is found numerically using **scipy Brentq** — a bracketed method combining bisection (guaranteed convergence) with secant steps (superlinear speed).

Before calling the solver, no-arbitrage bounds are checked:
- Lower bound: `max(S − K·e^{-rT}, 0)` — violation implies arbitrage
- Upper bound: `S` for calls, `K·e^{-rT}` for puts

### Greeks (`src/greeks.py`)

All five first-order Greeks computed analytically as partial derivatives of the BS price:

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta Δ | `N(d1)` (call), `N(d1)−1` (put) | Price change per $1 spot move |
| Gamma Γ | `n(d1) / (S·σ·√T)` | Delta change per $1 spot move |
| Vega ν | `S·n(d1)·√T` | Price change per 1% vol move (÷100) |
| Theta Θ | `−[S·n(d1)·σ/(2√T)] − r·K·e^{-rT}·N(d2)` | Daily price decay (÷365) |
| Rho ρ | `K·T·e^{-rT}·N(d2)` (call) | Price change per 1% rate move (÷100) |

Gamma and Vega are identical for calls and puts — provable directly from put-call parity.

### Volatility Surface (`src/vol_surface.py`)

**Synthetic mode** — parametric model for demo and testing:

```
IV(m, T) = (atm_vol + term·√T) + skew·m + smile·m²
```

where m = ln(K/S) is log-moneyness. Captures the three stylised facts of real vol surfaces: smile curvature, put skew, and term structure.

**Market mode** — calls the Brentq IV solver at every (strike, maturity) grid point on a real option chain. NaN values (solver failures at illiquid strikes) are filled by linear interpolation along the strike axis before Plotly renders the surface.

### Market Data (`src/market_data.py`)

The module is split into two explicit layers to preserve testability:

- **Network layer** — `get_spot_price()`, `get_risk_free_rate()`, `fetch_option_chain()`. Touches yfinance. Not unit-tested.
- **Processing layer** — `mid_price()`, `days_to_maturity()`, `filter_chain()`, `chain_to_grid()`. Pure functions. Fully tested without network access.

**filter_chain() applies four filters in sequence:**
1. Moneyness range (±40% log-moneyness) — removes near-zero-vega strikes
2. Open interest ≥ 10 — removes market-maker placeholder quotes
3. Mid-price validity — drops rows with no computable price
4. Spread ratio ≤ 0.50 — drops stale/crossed quotes (zero-bid rows exempt)

**chain_to_grid()** solves the union-strike problem: different expiries list different strike sets, so the function builds a common grid from the union of all strikes and aligns each chain onto it, filling gaps with NaN.

---

## Testing

The test suite covers correctness (reference values), numerical stability (finite difference validation), mathematical identities (put-call parity, Greek symmetries), edge cases (T=0, σ=0, deep ITM/OTM), and failure modes (invalid inputs, no-arbitrage violations).

```
tests/test_black_scholes.py   4 tests
tests/test_monte_carlo.py     1 test
tests/test_implied_vol.py    13 tests
tests/test_greeks.py         27 tests
tests/test_vol_surface.py    33 tests
tests/test_market_data.py    29 tests
─────────────────────────────────────
Total                       129 tests   0 failures
```

---

## Deployment

The app is deployed on Streamlit Cloud. To deploy your own instance:

```bash
# 1. Push to GitHub
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect your repo
# 4. Set main file: app.py
# 5. Deploy
```

Add a `.streamlit/config.toml` to lock the dark theme:

```toml
[theme]
base = "dark"
backgroundColor = "#0a0c0f"
secondaryBackgroundColor = "#0d1017"
textColor = "#d4d8dd"
font = "monospace"
```

---

## Stretch Goals

- [ ] Variance reduction — antithetic variates and control variates
- [ ] American option pricing via binomial tree
- [ ] P&L scenario analysis and heatmaps
- [ ] Export results to PDF/CSV
- [ ] Stochastic vol models — Heston model calibration
- [ ] Live price streaming via WebSockets

---

## Portfolio Context

**Target roles:** Quantitative Analyst · Quant Developer · Data Scientist (Financial Services)

**CV line:**
> Built an interactive options pricing and volatility analysis tool in Python (Streamlit), implementing Black-Scholes and Monte Carlo models, implied volatility extraction, a full Greeks dashboard, and a 3D volatility surface visualiser with real market data integration.

**Relevant modules at KCL:** Derivatives pricing theory · Optimisation Methods (scipy Brentq) · Stats for Finance (vol surface construction) · Data Visualisation (Plotly) · Computer Programming (modular architecture, pytest)

---

## Author

**Rudra Dubey**
MSc Data Science · King's College London
BSc Physics (2:1) · University of Exeter

[github.com/rudra-d6](https://github.com/rudra-d6) · [LinkedIn](www.linkedin.com/in/rudra-dubey-306465294)

---

 *Options Pricing & Volatility Surface Tool*