# app.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.black_scholes import black_scholes
from src.monte_carlo import monte_carlo
from src.greeks import all_greeks
from src.implied_vol import implied_volatility
from src.vol_surface import synthetic_surface, fill_nans, surface_stats

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Options Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — Bloomberg-terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root & background ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0c0f;
    color: #d4d8dd;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0d1017;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }

/* ── Header ─────────────────────────────────────────────────────── */
.app-header {
    padding: 1.2rem 0 0.6rem 0;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 1.6rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #f0c040;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.app-subtitle {
    font-size: 0.78rem;
    color: #5a6478;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* ── Metric cards ───────────────────────────────────────────────── */
.metric-card {
    background: #0d1017;
    border: 1px solid #1e2530;
    border-left: 3px solid #f0c040;
    border-radius: 3px;
    padding: 0.9rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-label {
    font-size: 0.68rem;
    color: #5a6478;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 600;
    color: #f0c040;
    line-height: 1;
}
.metric-delta {
    font-size: 0.72rem;
    margin-top: 0.25rem;
}
.metric-delta.pos { color: #3dd68c; }
.metric-delta.neg { color: #f05050; }
.metric-delta.neu { color: #5a6478; }

/* ── Section labels ─────────────────────────────────────────────── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #5a6478;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 0.35rem;
    margin-bottom: 1rem;
    margin-top: 1.4rem;
}

/* ── Greek cards ────────────────────────────────────────────────── */
.greek-card {
    background: #0d1017;
    border: 1px solid #1e2530;
    border-radius: 3px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
}
.greek-symbol {
    font-size: 1.4rem;
    color: #3b8eea;
    margin-bottom: 0.2rem;
}
.greek-name {
    font-size: 0.62rem;
    color: #5a6478;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.greek-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #d4d8dd;
    margin-top: 0.25rem;
}

/* ── Info box ───────────────────────────────────────────────────── */
.info-box {
    background: #0d1017;
    border: 1px solid #1e2530;
    border-left: 3px solid #3b8eea;
    border-radius: 3px;
    padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8090a8;
    line-height: 1.6;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5a6478;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f0c040;
    border-bottom-color: #f0c040 !important;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    background-color: #0d1017;
    border-color: #1e2530;
    color: #d4d8dd;
}

/* ── Streamlit overrides ────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #0d1017;
    border: 1px solid #1e2530;
    border-radius: 3px;
    padding: 0.8rem 1rem;
}
.stButton button {
    background-color: #f0c040;
    color: #0a0c0f;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 2px;
}
.stButton button:hover { background-color: #f5d060; }
hr { border-color: #1e2530; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly dark theme shared config
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0c0f",
    plot_bgcolor="#0d1017",
    font=dict(family="IBM Plex Mono", color="#8090a8", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", zerolinecolor="#1e2530"),
    yaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", zerolinecolor="#1e2530"),
)
AMBER  = "#f0c040"
GREEN  = "#3dd68c"
RED    = "#f05050"
BLUE   = "#3b8eea"
PURPLE = "#a67cec"

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <div class="app-title">⬡ Options Lab</div>
  <div class="app-subtitle">Black-Scholes · Monte Carlo · Implied Volatility · Greeks · Vol Surface</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — global parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Parameters")
    st.markdown("<div class='section-label'>Underlying</div>", unsafe_allow_html=True)
    S = st.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T = st.number_input("Maturity T (years)", value=1.0, min_value=0.01, max_value=10.0,
                        step=0.05, format="%.3f")

    st.markdown("<div class='section-label'>Market</div>", unsafe_allow_html=True)
    r     = st.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", value=0.20, min_value=0.001, max_value=5.0,
                             step=0.01, format="%.4f")
    option_type = st.selectbox("Option type", ["call", "put"])

    st.markdown("<div class='section-label'>Monte Carlo</div>", unsafe_allow_html=True)
    n_sim = st.select_slider(
        "Simulations",
        options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
        value=100_000,
    )
    mc_seed = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#3a4458;'>"
        "Rudra Dubey · MSc Data Science · KCL<br>"
        "Options Lab v1.0 · 2025"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 · Pricer",
    "02 · Greeks",
    "03 · Vol Surface",
    "04 · Convergence",
    "05 · Market Data",
])

@st.cache_data
def compute_delta_heatmap(K, T, r, option_type, s_min, s_max):
    s_grid   = np.linspace(s_min, s_max, 40)
    sig_grid = np.linspace(0.05, 0.80, 40)
    return np.array([
        [all_greeks(sv, K, T, r, sv2, option_type)["delta"]
         for sv in s_grid]
        for sv2 in sig_grid
    ]), s_grid, sig_grid

@st.cache_data
def compute_convergence(S, K, T, r, sigma, option_type, max_sims, n_seeds, base_seed):
    sim_steps = np.unique(np.round(
        np.logspace(np.log10(500), np.log10(max_sims), 60)
    ).astype(int))
    all_prices = []
    for seed_i in range(n_seeds):
        seed_val = base_seed + seed_i * 97
        prices = [monte_carlo(S, K, T, r, sigma, n_sim=int(n),
                              option_type=option_type, seed=seed_val)
                  for n in sim_steps]
        all_prices.append(prices)
    return sim_steps, np.array(all_prices)







# ============================================================================
# TAB 1 — PRICER
# ============================================================================
with tab1:
    st.markdown("<div class='section-label'>Black-Scholes vs Monte Carlo</div>",
                unsafe_allow_html=True)

    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    mc_price = monte_carlo(S, K, T, r, sigma, n_sim=n_sim,
                           option_type=option_type, seed=int(mc_seed))
    diff     = mc_price - bs_price
    diff_pct = (diff / bs_price * 100) if bs_price > 0 else 0.0

    # ── Metric cards ──────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Black-Scholes</div>
            <div class='metric-value'>${bs_price:.4f}</div>
            <div class='metric-delta neu'>Analytical</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        sign  = "pos" if diff >= 0 else "neg"
        arrow = "▲" if diff >= 0 else "▼"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Monte Carlo</div>
            <div class='metric-value'>${mc_price:.4f}</div>
            <div class='metric-delta {sign}'>{arrow} {abs(diff):.4f} ({diff_pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        time_val  = bs_price - intrinsic
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Intrinsic Value</div>
            <div class='metric-value'>${intrinsic:.4f}</div>
            <div class='metric-delta neu'>Time val: ${time_val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        moneyness_label = "ATM" if abs(S - K) < 0.01 * S else ("ITM" if (
            (option_type == "call" and S > K) or (option_type == "put" and S < K)
        ) else "OTM")
        log_m = np.log(S / K)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Moneyness</div>
            <div class='metric-value'>{moneyness_label}</div>
            <div class='metric-delta neu'>ln(S/K) = {log_m:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:2rem;'>Price Sensitivity</div>",
                unsafe_allow_html=True)

    # ── Price vs Spot chart ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        spot_range = np.linspace(max(S * 0.5, 1), S * 1.5, 120)
        bs_prices  = [black_scholes(s, K, T, r, sigma, option_type) for s in spot_range]
        intrinsics = [max(s - K, 0) if option_type == "call" else max(K - s, 0)
                      for s in spot_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spot_range, y=bs_prices,
            name="BS Price", line=dict(color=AMBER, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=spot_range, y=intrinsics,
            name="Intrinsic", line=dict(color=BLUE, width=1.5, dash="dash"),
        ))
        fig.add_vline(x=S, line=dict(color="#3a4458", width=1, dash="dot"))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Price vs Spot",
            xaxis_title="Spot (S)",
            yaxis_title="Option Price",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        vol_range  = np.linspace(0.01, 1.0, 120)
        bs_by_vol  = [black_scholes(S, K, T, r, v, option_type) for v in vol_range]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=vol_range * 100, y=bs_by_vol,
            name="BS Price", line=dict(color=GREEN, width=2),
        ))
        fig2.add_vline(x=sigma * 100, line=dict(color="#3a4458", width=1, dash="dot"))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title="Price vs Volatility",
            xaxis_title="Volatility σ (%)",
            yaxis_title="Option Price",
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Payoff diagram ─────────────────────────────────────────────
    st.markdown("<div class='section-label'>Expiry Payoff Diagram</div>",
                unsafe_allow_html=True)

    payoff_range  = np.linspace(max(S * 0.5, 1), S * 1.5, 200)
    payoff_values = [max(s - K, 0) if option_type == "call" else max(K - s, 0)
                     for s in payoff_range]
    net_pnl       = [p - bs_price for p in payoff_values]
    breakeven     = K + bs_price if option_type == "call" else K - bs_price

    fig3 = go.Figure()
    fig3.add_hline(y=0, line=dict(color="#3a4458", width=1))
    fig3.add_trace(go.Scatter(
        x=payoff_range, y=net_pnl,
        fill='tozeroy',
        fillcolor='rgba(61,214,140,0.08)',
        line=dict(color=GREEN, width=2),
        name="Net P&L at Expiry",
    ))
    fig3.add_vline(x=K, line=dict(color=AMBER, width=1, dash="dot"),
                   annotation_text="Strike", annotation_font_color=AMBER)
    fig3.add_vline(x=breakeven, line=dict(color=BLUE, width=1, dash="dot"),
                   annotation_text="Breakeven", annotation_font_color=BLUE)
    fig3.add_vline(x=S, line=dict(color="#5a6478", width=1, dash="dot"),
                   annotation_text="Spot", annotation_font_color="#5a6478")
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Long {option_type.capitalize()} P&L at Expiry (cost = ${bs_price:.4f})",
        xaxis_title="Spot at Expiry",
        yaxis_title="P&L",
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# TAB 2 — GREEKS
# ============================================================================
with tab2:
    st.markdown("<div class='section-label'>Greeks Dashboard</div>",
                unsafe_allow_html=True)

    greeks = all_greeks(S, K, T, r, sigma, option_type)

    # ── Greek summary cards ────────────────────────────────────────
    g_cols = st.columns(5)
    greek_meta = [
        ("Δ", "Delta",  greeks["delta"],  1,    "Price / $1 spot move"),
        ("Γ", "Gamma",  greeks["gamma"],  1,    "Delta / $1 spot move"),
        ("ν", "Vega",   greeks["vega"],   100,  "Price / 1% vol move"),
        ("Θ", "Theta",  greeks["theta"],  365,  "Price / 1 day"),
        ("ρ", "Rho",    greeks["rho"],    100,  "Price / 1% rate move"),
    ]
    for col, (sym, name, val, div, hint) in zip(g_cols, greek_meta):
        with col:
            display = val / div
            color = GREEN if display > 0 else (RED if display < 0 else "#5a6478")
            st.markdown(f"""
            <div class='greek-card'>
                <div class='greek-symbol'>{sym}</div>
                <div class='greek-name'>{name}</div>
                <div class='greek-value' style='color:{color}'>{display:.4f}</div>
                <div style='font-size:0.60rem;color:#3a4458;margin-top:0.3rem;'>{hint}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:2rem;'>Sensitivity Profiles</div>",
                unsafe_allow_html=True)

    # ── Greeks vs Spot ─────────────────────────────────────────────
    spot_scan = np.linspace(max(S * 0.5, 1), S * 1.55, 100)
    delta_s = [all_greeks(s, K, T, r, sigma, option_type)["delta"] for s in spot_scan]
    gamma_s = [all_greeks(s, K, T, r, sigma, option_type)["gamma"] for s in spot_scan]
    vega_s  = [all_greeks(s, K, T, r, sigma, option_type)["vega"]  for s in spot_scan]

    col1, col2 = st.columns(2)

    with col1:
        fig_dg = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dg.add_trace(go.Scatter(
            x=spot_scan, y=delta_s, name="Delta",
            line=dict(color=AMBER, width=2)), secondary_y=False)
        fig_dg.add_trace(go.Scatter(
            x=spot_scan, y=gamma_s, name="Gamma",
            line=dict(color=BLUE, width=2, dash="dash")), secondary_y=True)
        fig_dg.add_vline(x=S, line=dict(color="#3a4458", width=1, dash="dot"))
        fig_dg.update_layout(
            **PLOTLY_LAYOUT,
            title="Delta & Gamma vs Spot",
            height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig_dg.update_yaxes(title_text="Delta", secondary_y=False,
                             gridcolor="#1e2530", linecolor="#1e2530")
        fig_dg.update_yaxes(title_text="Gamma", secondary_y=True,
                             gridcolor="#1e2530", linecolor="#1e2530")
        fig_dg.update_xaxes(title_text="Spot (S)", gridcolor="#1e2530")
        st.plotly_chart(fig_dg, use_container_width=True)

    with col2:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=spot_scan, y=vega_s, name="Vega",
            line=dict(color=GREEN, width=2),
            fill='tozeroy', fillcolor='rgba(61,214,140,0.06)',
        ))
        fig_v.add_vline(x=S, line=dict(color="#3a4458", width=1, dash="dot"))
        fig_v.update_layout(
            **PLOTLY_LAYOUT,
            title="Vega vs Spot",
            xaxis_title="Spot (S)",
            yaxis_title="Vega",
            height=300,
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # ── Greeks vs Time ─────────────────────────────────────────────
    st.markdown("<div class='section-label'>Time Decay</div>", unsafe_allow_html=True)

    time_scan  = np.linspace(0.01, T, 100)
    theta_t    = [all_greeks(S, K, t, r, sigma, option_type)["theta"] / 365
                  for t in time_scan]
    bs_over_t  = [black_scholes(S, K, t, r, sigma, option_type) for t in time_scan]

    col3, col4 = st.columns(2)
    with col3:
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(
            x=time_scan, y=theta_t,
            line=dict(color=RED, width=2), name="Daily Theta",
            fill='tozeroy', fillcolor='rgba(240,80,80,0.06)',
        ))
        fig_th.update_layout(
            **PLOTLY_LAYOUT,
            title="Daily Theta Decay vs Time to Maturity",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="Theta (per day)",
            height=280,
        )
        st.plotly_chart(fig_th, use_container_width=True)

    with col4:
        fig_tv = go.Figure()
        fig_tv.add_trace(go.Scatter(
            x=time_scan, y=bs_over_t,
            line=dict(color=PURPLE, width=2), name="Option Price",
            fill='tozeroy', fillcolor='rgba(166,124,236,0.06)',
        ))
        fig_tv.update_layout(
            **PLOTLY_LAYOUT,
            title="Option Price vs Time to Maturity",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="Option Price",
            height=280,
        )
        st.plotly_chart(fig_tv, use_container_width=True)

    # ── Greeks heatmap: Delta over S × σ ──────────────────────────
    st.markdown("<div class='section-label'>Delta Heatmap — Spot × Volatility</div>",
                unsafe_allow_html=True)

    s_grid   = np.linspace(max(S * 0.6, 1), S * 1.4, 40)
    sig_grid = np.linspace(0.05, 0.80, 40)
    delta_heat = np.array([
        [all_greeks(sv, K, T, r, sv2, option_type)["delta"]
         for sv in s_grid]
        for sv2 in sig_grid
    ])

    fig_heat = go.Figure(data=go.Heatmap(
        z=delta_heat,
        x=np.round(s_grid, 1),
        y=np.round(sig_grid * 100, 1),
        colorscale=[[0, "#0d1017"], [0.5, BLUE], [1, AMBER]],
        colorbar=dict(
            title="Delta", tickfont=dict(family="IBM Plex Mono", color="#8090a8"),
            bgcolor="#0a0c0f", outlinecolor="#1e2530",
        ),
        hovertemplate="S=%{x}<br>σ=%{y}%<br>Delta=%{z:.4f}<extra></extra>",
    ))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title="Delta — Spot × Volatility grid",
        xaxis_title="Spot (S)",
        yaxis_title="Volatility (%)",
        height=340,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ============================================================================
# TAB 3 — VOL SURFACE
# ============================================================================
with tab3:
    st.markdown("<div class='section-label'>Volatility Surface — Synthetic Mode</div>",
                unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        atm_vol = st.slider("ATM Vol", 0.05, 0.80, 0.20, 0.01, format="%.2f",
                             key="surf_atm")
    with ctrl2:
        skew = st.slider("Skew", -0.50, 0.50, -0.10, 0.01, format="%.2f",
                          key="surf_skew")
    with ctrl3:
        smile = st.slider("Smile", 0.0, 2.0, 0.50, 0.05, format="%.2f",
                           key="surf_smile")
    with ctrl4:
        term = st.slider("Term Structure", -0.10, 0.10, 0.02, 0.005, format="%.3f",
                          key="surf_term")

    result = synthetic_surface(
        S=S, r=r,
        atm_vol=atm_vol, skew=skew, smile=smile, term=term,
        n_strikes=31, width=0.45,
        tenors=[1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0],
    )
    iv_grid    = fill_nans(result["iv_grid"])
    strikes    = result["strikes"]
    maturities = result["maturities"]
    moneyness  = result["moneyness"]
    stats      = surface_stats(result)

    # ── Stats strip ────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4)
    stat_cards = [
        ("ATM Vol (1M)",  f"{stats['atm_vols'][0]*100:.1f}%"),
        ("Min IV",        f"{stats['min_iv']*100:.1f}%"),
        ("Max IV",        f"{stats['max_iv']*100:.1f}%"),
        ("1M Skew",       f"{stats['skew_1m']*100:+.1f}%"),
    ]
    for col, (label, value) in zip([sc1, sc2, sc3, sc4], stat_cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='font-size:1.2rem'>{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:1.5rem;'>3D Surface</div>",
                unsafe_allow_html=True)

    # ── 3D Surface ─────────────────────────────────────────────────
    mat_labels = [f"{m*12:.0f}M" for m in maturities]

    fig_surf = go.Figure(data=[go.Surface(
        z=iv_grid * 100,
        x=strikes,
        y=maturities * 12,
        colorscale=[
            [0.0,  "#0d1017"],
            [0.25, "#1a3a6e"],
            [0.5,  BLUE],
            [0.75, AMBER],
            [1.0,  RED],
        ],
        colorbar=dict(
            title=dict(text="IV %", font=dict(family="IBM Plex Mono", color="#8090a8")),
            tickfont=dict(family="IBM Plex Mono", color="#8090a8"),
            bgcolor="#0a0c0f", outlinecolor="#1e2530",
        ),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
        hovertemplate="Strike=%{x:.1f}<br>Maturity=%{y:.1f}M<br>IV=%{z:.2f}%<extra></extra>",
        opacity=0.95,
    )])
    fig_surf.add_scatter3d(
        x=[S] * len(maturities),
        y=maturities * 12,
        z=stats["atm_vols"] * 100,
        mode="lines",
        line=dict(color=AMBER, width=4),
        name="ATM term structure",
    )
    fig_surf.update_layout(
        paper_bgcolor="#0a0c0f",
        font=dict(family="IBM Plex Mono", color="#8090a8", size=10),
        scene=dict(
            xaxis=dict(title="Strike", backgroundcolor="#0a0c0f",
                       gridcolor="#1e2530", linecolor="#1e2530"),
            yaxis=dict(title="Maturity (months)", backgroundcolor="#0a0c0f",
                       gridcolor="#1e2530", linecolor="#1e2530"),
            zaxis=dict(title="IV (%)", backgroundcolor="#0a0c0f",
                       gridcolor="#1e2530", linecolor="#1e2530"),
            bgcolor="#0a0c0f",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        title="Implied Volatility Surface",
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    # ── Vol smile slices ───────────────────────────────────────────
    st.markdown("<div class='section-label'>Vol Smile — Maturity Slices</div>",
                unsafe_allow_html=True)

    smile_colors = [AMBER, GREEN, BLUE, PURPLE, RED,
                    "#ff9040", "#40d0d0", "#d040d0"]
    fig_smile = go.Figure()
    for i, (T_val, mat_label) in enumerate(zip(maturities, mat_labels)):
        fig_smile.add_trace(go.Scatter(
            x=moneyness, y=iv_grid[i] * 100,
            name=mat_label,
            line=dict(color=smile_colors[i % len(smile_colors)], width=1.8),
            mode="lines",
            hovertemplate=f"{mat_label}: IV=%{{y:.2f}}%<extra></extra>",
        ))
    fig_smile.add_vline(x=0, line=dict(color="#3a4458", width=1, dash="dot"),
                        annotation_text="ATM", annotation_font_color="#5a6478")
    fig_smile.update_layout(
        **PLOTLY_LAYOUT,
        title="Vol Smile by Maturity",
        xaxis_title="Log-Moneyness ln(K/S)",
        yaxis_title="Implied Volatility (%)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=340,
    )
    st.plotly_chart(fig_smile, use_container_width=True)

    # ── ATM term structure ─────────────────────────────────────────
    st.markdown("<div class='section-label'>ATM Term Structure</div>",
                unsafe_allow_html=True)

    fig_term = go.Figure()
    fig_term.add_trace(go.Scatter(
        x=maturities * 12,
        y=stats["atm_vols"] * 100,
        mode="lines+markers",
        line=dict(color=AMBER, width=2),
        marker=dict(color=AMBER, size=7, symbol="diamond"),
        name="ATM IV",
        hovertemplate="Maturity=%{x:.1f}M<br>ATM IV=%{y:.2f}%<extra></extra>",
    ))
    fig_term.update_layout(
        **PLOTLY_LAYOUT,
        title="ATM Implied Volatility — Term Structure",
        xaxis_title="Maturity (months)",
        yaxis_title="ATM IV (%)",
        height=280,
    )
    st.plotly_chart(fig_term, use_container_width=True)


# ============================================================================
# TAB 4 — MC CONVERGENCE
# ============================================================================
with tab4:
    st.markdown("<div class='section-label'>Monte Carlo Convergence Visualiser</div>",
                unsafe_allow_html=True)

    bs_ref = black_scholes(S, K, T, r, sigma, option_type)

    c1, c2 = st.columns(2)
    with c1:
        max_sims = st.select_slider(
            "Max simulations",
            options=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
            value=100_000,
            key="conv_max",
        )
    with c2:
        n_seeds = st.slider("Number of independent runs", 1, 10, 5, key="conv_seeds")

    # ── Build convergence data ──────────────────────────────────────
    sim_steps = np.unique(np.round(
        np.logspace(np.log10(500), np.log10(max_sims), 60)
    ).astype(int))

    conv_colors = [AMBER, GREEN, BLUE, PURPLE, RED,
                   "#ff9040", "#40d0d0", "#d040d0", "#d0d040", "#40d080"]

    fig_conv = go.Figure()
    fig_conv.add_hline(
        y=bs_ref, line=dict(color=AMBER, width=2, dash="dot"),
        annotation_text=f"BS = {bs_ref:.4f}",
        annotation_font_color=AMBER,
        annotation_position="right",
    )

    all_prices = []
    for seed_i in range(n_seeds):
        seed_val = int(mc_seed) + seed_i * 97
        prices   = []
        for n in sim_steps:
            p = monte_carlo(S, K, T, r, sigma, n_sim=int(n),
                            option_type=option_type, seed=seed_val)
            prices.append(p)
        all_prices.append(prices)
        opacity = 0.9 if seed_i == 0 else 0.45
        fig_conv.add_trace(go.Scatter(
            x=sim_steps, y=prices,
            mode="lines",
            name=f"Seed {seed_val}",
            line=dict(color=conv_colors[seed_i % len(conv_colors)],
                      width=1.5 if seed_i > 0 else 2),
            opacity=opacity,
            hovertemplate="N=%{x:,.0f}<br>Price=%{y:.5f}<extra></extra>",
        ))

    # ── Standard error envelope ────────────────────────────────────
    prices_arr = np.array(all_prices)
    mean_conv  = prices_arr.mean(axis=0)
    std_conv   = prices_arr.std(axis=0)
    fig_conv.add_trace(go.Scatter(
        x=np.concatenate([sim_steps, sim_steps[::-1]]),
        y=np.concatenate([mean_conv + std_conv, (mean_conv - std_conv)[::-1]]),
        fill='toself',
        fillcolor='rgba(61,142,234,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name="±1 std dev",
        showlegend=True,
    ))

    fig_conv.update_layout(
        **PLOTLY_LAYOUT,
        title=f"MC Convergence — {option_type.capitalize()} Price vs N Simulations",
        xaxis_title="Number of Simulations (log scale)",
        yaxis_title="Option Price",
        xaxis_type="log",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=420,
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # ── Error vs N ─────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Absolute Error vs Black-Scholes</div>",
                unsafe_allow_html=True)

    errors    = np.abs(mean_conv - bs_ref)
    theory_se = bs_ref * sigma * np.sqrt(1 / sim_steps)  # approximate 1/sqrt(N) envelope

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=sim_steps, y=errors,
        mode="lines", name="|MC - BS|",
        line=dict(color=RED, width=2),
    ))
    fig_err.add_trace(go.Scatter(
        x=sim_steps, y=theory_se,
        mode="lines", name="~1/√N envelope",
        line=dict(color="#5a6478", width=1.5, dash="dash"),
    ))
    fig_err.update_layout(
        **PLOTLY_LAYOUT,
        title="Absolute Error vs N (log-log)",
        xaxis_title="Number of Simulations",
        yaxis_title="Absolute Error",
        xaxis_type="log",
        yaxis_type="log",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=300,
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Final MC stats ──────────────────────────────────────────────
    final_mc  = monte_carlo(S, K, T, r, sigma, n_sim=max_sims,
                            option_type=option_type, seed=int(mc_seed))
    final_err = abs(final_mc - bs_ref)
    se_theory = bs_ref * sigma / np.sqrt(max_sims)

    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value) in zip([m1, m2, m3, m4], [
        ("BS Reference",    f"${bs_ref:.5f}"),
        (f"MC ({max_sims:,})", f"${final_mc:.5f}"),
        ("Abs. Error",      f"${final_err:.5f}"),
        ("Theoretical SE",  f"${se_theory:.5f}"),
    ]):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='font-size:1.1rem'>{value}</div>
            </div>""", unsafe_allow_html=True)


# ============================================================================
# TAB 5 — MARKET DATA
# ============================================================================
with tab5:
    st.markdown("<div class='section-label'>Live Market Data — Implied Volatility Surface</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Fetches a live option chain from Yahoo Finance, cleans and filters it, then
    extracts implied volatility at each (strike, maturity) grid point using the
    Brentq solver. Requires an active internet connection. Use the synthetic
    surface in Tab 03 for offline exploration.
    </div>
    """, unsafe_allow_html=True)

    md_col1, md_col2, md_col3, md_col4 = st.columns(4)
    with md_col1:
        ticker = st.text_input("Ticker", value="AAPL", key="md_ticker").upper()
    with md_col2:
        md_option_type = st.selectbox("Option type", ["call", "put"], key="md_type")
    with md_col3:
        max_expiries = st.slider("Max expiries", 3, 12, 6, key="md_expiries")
    with md_col4:
        md_moneyness = st.slider("Moneyness range", 0.10, 0.60, 0.35, 0.05,
                                  key="md_moneyness", format="%.2f")

    fetch_btn = st.button("Fetch Live Surface", key="md_fetch")

    if fetch_btn:
        try:
            from src.market_data import fetch_surface_data

            with st.spinner(f"Fetching {ticker} option chain…"):
                md_result = fetch_surface_data(
                    ticker=ticker,
                    option_type=md_option_type,
                    max_expiries=max_expiries,
                    moneyness_range=md_moneyness,
                )

            md_iv    = md_result["iv_grid"]
            md_K     = md_result["strikes"]
            md_T     = md_result["maturities"]
            md_m     = md_result["moneyness"]
            md_spot  = md_result["spot"]
            md_stats = surface_stats(md_result)

            st.success(
                f"✓  {ticker} · Spot ${md_spot:.2f} · "
                f"{len(md_T)} expiries · {len(md_K)} strikes"
            )

            # ── Stats strip ────────────────────────────────────────
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, (label, value) in zip([sc1, sc2, sc3, sc4], [
                ("Spot",         f"${md_spot:.2f}"),
                ("ATM Vol (1M)", f"{md_stats['atm_vols'][0]*100:.1f}%"
                                 if not np.isnan(md_stats['atm_vols'][0]) else "N/A"),
                ("IV Range",     f"{md_stats['min_iv']*100:.1f}% – "
                                 f"{md_stats['max_iv']*100:.1f}%"),
                ("Term Slope",   f"{md_stats['term_slope']*100:+.1f}%"
                                 if not np.isnan(md_stats['term_slope']) else "N/A"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value' style='font-size:1.1rem'>{value}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(
                "<div class='section-label' style='margin-top:1.5rem;'>3D Surface</div>",
                unsafe_allow_html=True,
            )

            # ── 3D surface ──────────────────────────────────────────
            fig_md = go.Figure(data=[go.Surface(
                z=md_iv * 100,
                x=md_K,
                y=md_T * 12,
                colorscale=[
                    [0.0,  "#0d1017"],
                    [0.25, "#1a3a6e"],
                    [0.5,  BLUE],
                    [0.75, AMBER],
                    [1.0,  RED],
                ],
                colorbar=dict(
                    title=dict(text="IV %",
                               font=dict(family="IBM Plex Mono", color="#8090a8")),
                    tickfont=dict(family="IBM Plex Mono", color="#8090a8"),
                    bgcolor="#0a0c0f", outlinecolor="#1e2530",
                ),
                hovertemplate="K=%{x:.1f}<br>T=%{y:.1f}M<br>IV=%{z:.2f}%<extra></extra>",
            )])
            fig_md.update_layout(
                paper_bgcolor="#0a0c0f",
                font=dict(family="IBM Plex Mono", color="#8090a8", size=10),
                scene=dict(
                    xaxis=dict(title="Strike", backgroundcolor="#0a0c0f",
                               gridcolor="#1e2530"),
                    yaxis=dict(title="Maturity (months)", backgroundcolor="#0a0c0f",
                               gridcolor="#1e2530"),
                    zaxis=dict(title="IV (%)", backgroundcolor="#0a0c0f",
                               gridcolor="#1e2530"),
                    bgcolor="#0a0c0f",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=500,
                title=f"{ticker} Implied Volatility Surface",
            )
            st.plotly_chart(fig_md, use_container_width=True)

            # ── Vol smile slices ────────────────────────────────────
            st.markdown(
                "<div class='section-label'>Vol Smile — Maturity Slices</div>",
                unsafe_allow_html=True,
            )
            fig_md_smile = go.Figure()
            smile_cols   = [AMBER, GREEN, BLUE, PURPLE, RED,
                            "#ff9040", "#40d0d0", "#d040d0",
                            "#d0d040", "#40d080", "#d08040", "#80d040"]
            for i, T_val in enumerate(md_T):
                label = f"{T_val*12:.1f}M"
                fig_md_smile.add_trace(go.Scatter(
                    x=md_m, y=md_iv[i] * 100,
                    name=label,
                    line=dict(color=smile_cols[i % len(smile_cols)], width=1.8),
                    mode="lines",
                    hovertemplate=f"{label}: IV=%{{y:.2f}}%<extra></extra>",
                ))
            fig_md_smile.add_vline(
                x=0, line=dict(color="#3a4458", width=1, dash="dot"),
                annotation_text="ATM", annotation_font_color="#5a6478",
            )
            fig_md_smile.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{ticker} Vol Smile by Maturity",
                xaxis_title="Log-Moneyness ln(K/S)",
                yaxis_title="Implied Volatility (%)",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                height=360,
            )
            st.plotly_chart(fig_md_smile, use_container_width=True)

            # ── Raw IV grid table ───────────────────────────────────
            with st.expander("Raw IV Grid (%)"):
                df_iv = pd.DataFrame(
                    md_iv * 100,
                    index=[f"{t*12:.1f}M" for t in md_T],
                    columns=[f"{k:.0f}" for k in md_K],
                ).round(2)
                st.dataframe(df_iv, use_container_width=True)

        except ImportError:
            st.error("yfinance is not installed. Run: `pip install yfinance`")
        except ValueError as e:
            st.error(f"Data error: {e}")
        except Exception as e:
            st.error(f"Failed to fetch {ticker}: {e}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;font-family:IBM Plex Mono;
                    font-size:0.78rem;color:#3a4458;'>
            Enter a ticker and click Fetch Live Surface
        </div>""", unsafe_allow_html=True)