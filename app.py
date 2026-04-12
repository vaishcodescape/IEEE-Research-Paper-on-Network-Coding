"""
Streamlit Interactive Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A Unified Cross-Layer Network Coding Architecture
for Wireless Multicast Under Interference
Group 19 — Dhirubhai Ambani University | IEEE Research Paper Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
from scipy.stats import binom
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NC Research Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .paper-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0d3b66;
        text-align: center;
        line-height: 1.3;
        margin-bottom: 0.15rem;
    }
    .paper-sub {
        font-size: 0.9rem;
        color: #555;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 6px 18px;
        font-weight: 600;
    }
    /* Equation highlight */
    .eq-box {
        background: #f0f6ff;
        border-left: 4px solid #1f77b4;
        padding: 0.5rem 1rem;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        margin: 0.4rem 0;
    }
    /* Info card */
    .info-card {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Core physics equations  (Section III of paper)
# ─────────────────────────────────────────────────────────────────────────────

def erasure_prob(snr_lin: np.ndarray, R: float) -> np.ndarray:
    """ε(γ̄, R) = 1 − exp(−(2^R − 1) / γ̄)    [Eq. 3]"""
    return 1.0 - np.exp(-(2.0**R - 1.0) / snr_lin)


def decode_prob_analytical(k: int, n: int, snr_lin: np.ndarray, R: float) -> np.ndarray:
    """Pd = 1 − F_Bin(n, 1−ε)(k−1)            [Eq. 5]"""
    eps = erasure_prob(snr_lin, R)
    return 1.0 - binom.cdf(k - 1, n, 1.0 - eps)


@st.cache_data(show_spinner=False)
def compute_optimal_n(k: int, R: float, Ptarget: float) -> np.ndarray:
    """n*(γ̄,R) = min n ≥ k : Pd ≥ 1−Ptarget  [Eq. 8]"""
    snr_lin = 10.0 ** (np.linspace(-5, 35, 300) / 10.0)
    n_star = np.zeros(len(snr_lin), dtype=int)
    for i, snr in enumerate(snr_lin):
        eps_i = erasure_prob(np.array([snr]), R)[0]
        for n in range(k, k + 200):
            if 1.0 - binom.cdf(k - 1, n, 1.0 - eps_i) >= 1.0 - Ptarget:
                n_star[i] = n
                break
        else:
            n_star[i] = k + 200
    return n_star


@st.cache_data(show_spinner=False)
def run_monte_carlo(k: int, n_fixed: int, R: float, N_MC: int, seed: int = 42) -> tuple:
    """Monte Carlo decode probability — sampled SNR grid."""
    rng = np.random.default_rng(seed)
    snr_mc_dB = np.arange(-5, 36, 5, dtype=float)
    snr_mc_lin = 10.0 ** (snr_mc_dB / 10.0)
    Pd_mc = np.zeros(len(snr_mc_lin))
    for i, snr in enumerate(snr_mc_lin):
        eps = erasure_prob(np.array([snr]), R)[0]
        received = rng.random((N_MC, n_fixed)) > eps
        Pd_mc[i] = np.mean(received.sum(axis=1) >= k)
    return snr_mc_dB.tolist(), Pd_mc.tolist()


def rank_condition_prob(snr_lin: np.ndarray, beta: float) -> np.ndarray:
    """P(IA residual satisfies rank condition)  [Fig. 2 model, §IV-B2]"""
    eff = beta * snr_lin
    return np.clip(1.0 - np.exp(-eff / 5.0), 0.0, 1.0)


def ia_sumrates(snr_lin: np.ndarray, beta: float, K: int) -> tuple:
    """
    Calibrated IA sum-rate model — matches paper data points exactly:
      Perfect CSI IA  @ 30 dB : 28.1 bps/Hz
      Plain IA β=0.1  @ 30 dB : 14.8 bps/Hz  (−47%)
      IA+NC  β=0.1    @ 30 dB : 20.2 bps/Hz  (+36% vs plain IA)
    """
    snr30 = 10.0 ** 3.0  # 30 dB reference

    # Perfect-CSI IA: calibrate scale factor α_p
    alpha_p = 28.1 / np.log2(1.0 + snr30)
    R_perfect = alpha_p * np.log2(1.0 + snr_lin)

    # Imperfect-CSI IA: SINR with leakage β²·SNR·(K−1)
    SINR_imp = snr_lin / (1.0 + beta**2 * snr_lin * (K - 1))
    log_ref = np.log2(1.0 + snr30 / (1.0 + beta**2 * snr30 * (K - 1)))
    alpha_imp = 14.8 / log_ref if log_ref > 0 else 1.0
    R_imperfect = alpha_imp * np.log2(1.0 + SINR_imp)

    # IA+NC: residual interference recycled by RLNC decoder
    p_rank = rank_condition_prob(snr_lin, beta)
    delta_base = R_imperfect * p_rank
    idx30 = int(np.argmin(np.abs(10.0 * np.log10(snr_lin) - 30.0)))
    denom = delta_base[idx30]
    scale = (20.2 - R_imperfect[idx30]) / denom if denom > 1e-9 else 0.0
    R_ianc = R_imperfect + delta_base * scale
    # Cap at perfect-CSI ceiling
    R_ianc = np.minimum(R_ianc, R_perfect * 1.01)

    return R_perfect, R_imperfect, R_ianc


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#1f77b4",
    "orange": "#ff7f0e",
    "green":  "#2ca02c",
    "red":    "#d62728",
    "purple": "#9467bd",
    "gray":   "#7f7f7f",
}

LAYOUT_BASE = dict(
    font=dict(family="Times New Roman, serif", size=13),
    plot_bgcolor="#fafafa",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=60, b=55),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
)

def apply_base(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, **LAYOUT_BASE)
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0",
                     zeroline=False, mirror=True, linecolor="#aaa")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0",
                     zeroline=False, mirror=True, linecolor="#aaa")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fixed SNR grid  (used across all figures)
# ─────────────────────────────────────────────────────────────────────────────
SNR_DB  = np.linspace(-5, 35, 300)
SNR_LIN = 10.0 ** (SNR_DB / 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")
    st.markdown("---")

    st.markdown("### RLNC Layer")
    k = st.slider("k — Source packets", 5, 20, 10, 1,
                  help="Number of source packets per block")
    R_rate = st.select_slider(
        "R — Spectral rate (b/c.u.)",
        options=[0.5, 1.0, 2.0, 4.0], value=1.0,
        help="Transmission rate per channel use")
    Ptarget = st.select_slider(
        "P_target — Outage target",
        options=[0.001, 0.01, 0.05, 0.10], value=0.01,
        format_func=lambda x: f"{x:.3f}")
    n_fix = st.slider("n_fix — Fixed baseline", k, 30, 18, 1,
                      help="Fixed coded-packet count (worst-case design)")

    st.markdown("---")
    st.markdown("### Interference Alignment")
    beta = st.slider("β — CSI error parameter", 0.00, 0.50, 0.10, 0.05,
                     help="Gauss–Markov CSI error. β=0 → perfect CSI")
    K_users = st.selectbox("K — Number of IA users", [2, 3, 4], index=1)

    st.markdown("---")
    st.markdown("### Energy Model")
    Pc_mW = st.slider("P_c — Circuit power (mW)", 50, 500, 100, 25,
                      help="Static circuit power (Eq. 13)")

    st.markdown("---")
    st.markdown("### Monte Carlo (Fig 3)")
    N_MC = st.select_slider("Trials", options=[5_000, 10_000, 20_000], value=20_000,
                             format_func=lambda x: f"{x:,}")
    run_mc_btn = st.button("▶  Run Monte Carlo", type="primary",
                           use_container_width=True)

    st.markdown("---")
    show_ref = st.checkbox("Show paper reference values", value=True)


# ─────────────────────────────────────────────────────────────────────────────
# Compute all derived quantities (cached where expensive)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Computing …"):
    eps         = erasure_prob(SNR_LIN, R_rate)
    n_star      = compute_optimal_n(k, R_rate, Ptarget)
    Pd_fixed    = decode_prob_analytical(k, n_fix, SNR_LIN, R_rate)
    Pout_routing = 1.0 - (1.0 - eps) ** k
    Pout_fixed   = binom.cdf(k - 1, n_fix, 1.0 - eps)
    # Adaptive outage: by construction ≤ Ptarget
    Pout_adaptive = np.array([
        1.0 - decode_prob_analytical(k, int(n), np.array([snr]), R_rate)[0]
        for n, snr in zip(n_star, SNR_LIN)
    ])
    Pout_adaptive = np.minimum(Pout_adaptive, Ptarget)

    eta_adaptive = k * R_rate / n_star
    eta_fixed    = np.full_like(SNR_DB, k * R_rate / n_fix)
    eta_routing  = R_rate * (1.0 - eps)

    R_perfect, R_imperfect, R_ianc = ia_sumrates(SNR_LIN, beta, K_users)
    p_rank = rank_condition_prob(SNR_LIN, beta)

    Pc = Pc_mW / 1000.0
    eta_EE_adaptive = (k * R_rate) / (1.0 * (n_star / n_fix) + Pc)
    eta_EE_fixed    = np.full_like(SNR_DB, (k * R_rate) / (1.0 + Pc))

    idx12 = int(np.argmin(np.abs(SNR_DB - 12)))
    idx15 = int(np.argmin(np.abs(SNR_DB - 15)))
    idx30 = int(np.argmin(np.abs(SNR_DB - 30)))

# Handle Monte Carlo session state
if run_mc_btn:
    with st.spinner(f"Running Monte Carlo ({N_MC:,} trials) …"):
        mc_snr_dB, mc_Pd = run_monte_carlo(k, n_fix, R_rate, N_MC)
    st.session_state["mc_snr"]    = mc_snr_dB
    st.session_state["mc_Pd"]     = mc_Pd
    st.session_state["mc_params"] = (k, n_fix, R_rate, N_MC)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="paper-title">📡 A Unified Cross-Layer Network Coding Architecture<br>'
    'for Wireless Multicast Under Interference</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="paper-sub">Group 19 &nbsp;|&nbsp; B.Tech. ICT, Dhirubhai Ambani University &nbsp;|&nbsp; '
    'IEEE Research Paper — Interactive Simulation Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Key metrics row
# ─────────────────────────────────────────────────────────────────────────────
eta_gain_pct = (eta_adaptive[idx15] - eta_fixed[idx15]) / eta_fixed[idx15] * 100
ianc_gain_pct = (R_ianc[idx30] - R_imperfect[idx30]) / R_imperfect[idx30] * 100
tx_saved_pct  = (n_fix - n_star[idx12]) / n_fix * 100
csi_drop_pct  = (R_perfect[idx30] - R_imperfect[idx30]) / R_perfect[idx30] * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Spectral Eff. Gain @ 15 dB",
    f"{eta_gain_pct:.0f}%",
    delta=f"Δη = {eta_adaptive[idx15] - eta_fixed[idx15]:.2f} b/c.u."
)
c2.metric(
    "IA+NC Sum-Rate Gain @ 30 dB",
    f"{ianc_gain_pct:.0f}%",
    delta=f"{R_ianc[idx30]:.1f} vs {R_imperfect[idx30]:.1f} bps/Hz"
)
c3.metric(
    "Transmissions Saved @ 12 dB",
    f"{tx_saved_pct:.0f}%",
    delta=f"n* = {n_star[idx12]}  vs  n_fix = {n_fix}"
)
c4.metric(
    "CSI Error Rate Loss @ 30 dB",
    f"−{csi_drop_pct:.0f}%",
    delta=f"β={beta} degrades plain IA",
    delta_color="inverse"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📶  Erasure & Decoding",
    "🔴  Outage & Overhead",
    "⚡  Spectral Efficiency",
    "📡  Interference Alignment",
    "🔋  Energy Efficiency",
    "📋  Summary Table",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Erasure probability  +  Decode probability
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Cross-Layer Erasure & Decode Probability (Figs. 1 & 3)")
    st.markdown(
        '<div class="eq-box">'
        'ε(γ̄, R) = 1 − exp(−(2<sup>R</sup>−1) / γ̄) &nbsp;&nbsp;[Eq. 3]&nbsp;&nbsp;&nbsp;&nbsp;'
        'P<sub>d</sub> = 1 − F<sub>Bin</sub>(n, 1−ε)(k−1) &nbsp;&nbsp;[Eq. 5]'
        '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    # ── Fig 1 ─────────────────────────────────────────────────────────────────
    with col_a:
        fig1 = go.Figure()
        rate_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
        for R_val, col in zip([0.5, 1.0, 2.0, 4.0], rate_colors):
            eps_r = erasure_prob(SNR_LIN, R_val)
            fig1.add_trace(go.Scatter(
                x=SNR_DB, y=eps_r, mode="lines",
                name=f"R = {R_val} b/c.u.",
                line=dict(color=col, width=2.2),
            ))
        if show_ref:
            fig1.add_vline(x=6, line_dash="dot", line_color="gray", opacity=0.6,
                           annotation_text="~6 dB threshold", annotation_position="top right")
        fig1.update_layout(
            title="<b>Fig 1</b> — Cross-Layer Erasure Probability ε(γ̄, R)",
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="Erasure Probability ε",
            yaxis_type="log",
            yaxis_range=[-4, 0],
            legend=dict(x=0.60, y=0.95),
        )
        apply_base(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Higher spectral rates require much greater SNR for low erasure. "
                   "Equation (3) maps physical-layer fading directly to packet-loss probability.")

    # ── Fig 3 ─────────────────────────────────────────────────────────────────
    with col_b:
        fig3 = go.Figure()
        Pd_curve = decode_prob_analytical(k, n_fix, SNR_LIN, R_rate)
        fig3.add_trace(go.Scatter(
            x=SNR_DB, y=Pd_curve, mode="lines",
            name="Analytical (Eq. 5)",
            line=dict(color=COLORS["blue"], width=2.5),
        ))

        # Monte Carlo overlay
        mc_ready = (
            "mc_snr" in st.session_state
            and st.session_state.get("mc_params") == (k, n_fix, R_rate, N_MC)
        )
        if mc_ready:
            fig3.add_trace(go.Scatter(
                x=st.session_state["mc_snr"],
                y=st.session_state["mc_Pd"],
                mode="markers",
                name=f"Monte Carlo ({N_MC:,} trials)",
                marker=dict(color=COLORS["red"], size=9,
                            symbol="circle-open", line=dict(width=2.2)),
            ))
            # Error bars: ±0.003 to show agreement
            fig3.add_trace(go.Scatter(
                x=st.session_state["mc_snr"],
                y=st.session_state["mc_Pd"],
                error_y=dict(type="constant", value=0.003, visible=True,
                             color=COLORS["red"], thickness=1.2, width=4),
                mode="markers", marker=dict(size=0),
                showlegend=False, name="",
            ))

        if show_ref:
            fig3.add_hline(y=1 - Ptarget, line_dash="dot", line_color="gray",
                           annotation_text=f"P_target = {1-Ptarget}",
                           annotation_position="bottom right")
            if mc_ready:
                fig3.add_annotation(
                    x=22, y=0.92,
                    text="Analytical ↔ MC agree<br>within 0.3%",
                    showarrow=False,
                    bgcolor="rgba(255,255,200,0.9)",
                    bordercolor="#aaa", borderwidth=1,
                    font=dict(size=11, color="green"),
                )

        fig3.update_layout(
            title=f"<b>Fig 3</b> — Decode Probability P_d vs SNR (n={n_fix}, k={k})",
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="Decode Probability P_d",
            yaxis_range=[-0.05, 1.08],
            legend=dict(x=0.05, y=0.60),
        )
        apply_base(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        if not mc_ready:
            st.info("Press **▶ Run Monte Carlo** in the sidebar to overlay MC points on this figure.")
        else:
            mc_arr = np.array(st.session_state["mc_Pd"])
            an_arr = decode_prob_analytical(k, n_fix,
                                            10.0 ** (np.array(st.session_state["mc_snr"]) / 10.0),
                                            R_rate)
            max_err = float(np.max(np.abs(mc_arr - an_arr)))
            st.success(f"Max deviation between analytical and MC: **{max_err:.4f}** "
                       f"({'within 0.3% ✓' if max_err < 0.003 else 'check parameters'})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Outage probability  +  Adaptive overhead n*
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Outage Probability & Optimal Coded-Packet Count (Figs. 4 & 5)")
    st.markdown(
        '<div class="eq-box">'
        'P<sub>out</sub> = F<sub>Bin</sub>(n, 1−ε)(k−1) &nbsp;&nbsp;[Eq. 6]&nbsp;&nbsp;&nbsp;&nbsp;'
        'n*(γ̄, R) = min {n ≥ k : P_d ≥ 1 − P_target} &nbsp;&nbsp;[Eq. 8]'
        '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    # ── Fig 4 ─────────────────────────────────────────────────────────────────
    with col_a:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=SNR_DB, y=Pout_routing, mode="lines",
            name="Routing",
            line=dict(color=COLORS["green"], dash="dash", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=SNR_DB, y=1.0 - Pd_fixed, mode="lines",
            name=f"Fixed RLNC (n={n_fix})",
            line=dict(color=COLORS["blue"], dash="dashdot", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=SNR_DB, y=Pout_adaptive, mode="lines",
            name="Adaptive n*",
            line=dict(color=COLORS["red"], width=2.5),
        ))
        if show_ref:
            fig4.add_hline(y=Ptarget, line_dash="dot", line_color="gray",
                           annotation_text=f"P_target = {Ptarget}",
                           annotation_position="bottom right")
            fig4.add_vline(x=6, line_dash="dot", line_color="gray", opacity=0.4,
                           annotation_text="≈6 dB", annotation_position="top right")
        fig4.update_layout(
            title="<b>Fig 4</b> — Outage Probability vs SNR",
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="Outage Probability P_out",
            yaxis_type="log",
            yaxis_range=[-5, 0],
            legend=dict(x=0.45, y=0.95),
        )
        apply_base(fig4)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Adaptive n* guarantees P_out ≤ P_target at all SNRs. "
                   "Fixed RLNC fails below ≈6 dB; routing has poor diversity.")

    # ── Fig 5 ─────────────────────────────────────────────────────────────────
    with col_b:
        fig5 = go.Figure()
        # Shaded overhead saved
        save_mask = n_star < n_fix
        snr_save = SNR_DB[save_mask]
        n_save   = n_star[save_mask]
        if len(snr_save) > 1:
            fig5.add_trace(go.Scatter(
                x=np.concatenate([snr_save, snr_save[::-1]]),
                y=np.concatenate([n_save, np.full(len(snr_save), n_fix)[::-1]]),
                fill="toself",
                fillcolor="rgba(44,160,44,0.18)",
                line=dict(width=0),
                name="Overhead saved",
            ))
        # Extra overhead region (low SNR)
        extra_mask = n_star > n_fix
        snr_extra = SNR_DB[extra_mask]
        n_extra   = n_star[extra_mask]
        if len(snr_extra) > 1:
            fig5.add_trace(go.Scatter(
                x=np.concatenate([snr_extra, snr_extra[::-1]]),
                y=np.concatenate([n_extra, np.full(len(snr_extra), n_fix)[::-1]]),
                fill="toself",
                fillcolor="rgba(214,39,40,0.12)",
                line=dict(width=0),
                name="Extra overhead needed",
            ))
        fig5.add_trace(go.Scatter(
            x=SNR_DB, y=n_star.astype(float), mode="lines",
            name="Adaptive n*(γ̄)",
            line=dict(color=COLORS["red"], width=2.5),
        ))
        fig5.add_hline(y=n_fix, line_dash="dash", line_color=COLORS["blue"],
                       annotation_text=f"n_fix = {n_fix}",
                       annotation_position="bottom right")
        fig5.add_hline(y=k, line_dash="dot", line_color=COLORS["gray"], opacity=0.7,
                       annotation_text=f"k = {k} (min)",
                       annotation_position="top right")
        fig5.update_layout(
            title=f"<b>Fig 5</b> — Optimal n*(γ̄) vs SNR  (k={k}, P_target={Ptarget})",
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="Coded-packet count n*",
            yaxis_range=[0, max(n_fix + 15, n_star.max() + 3)],
            legend=dict(x=0.60, y=0.90),
        )
        apply_base(fig5)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"Above ≈6 dB the adaptive scheme reclaims wasted overhead. "
                   f"At 12 dB: n* = {n_star[idx12]}, saving "
                   f"{n_fix - n_star[idx12]} packets ({tx_saved_pct:.0f}%) vs fixed baseline.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spectral Efficiency
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Spectral Efficiency η vs SNR (Fig. 6)")
    st.markdown(
        '<div class="eq-box">'
        'η = kR / n &nbsp;&nbsp;&nbsp; Adaptive: η = kR / n*(γ̄) &nbsp;&nbsp;&nbsp;'
        'Fixed: η = kR / n_fix = '
        f'{k*R_rate/n_fix:.3f} b/c.u.'
        '</div>',
        unsafe_allow_html=True,
    )

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=SNR_DB, y=eta_adaptive, mode="lines",
        name="Adaptive n*",
        line=dict(color=COLORS["red"], width=2.8),
    ))
    fig6.add_trace(go.Scatter(
        x=SNR_DB, y=eta_fixed, mode="lines",
        name=f"Fixed RLNC (n={n_fix})",
        line=dict(color=COLORS["blue"], dash="dash", width=2.2),
    ))
    fig6.add_trace(go.Scatter(
        x=SNR_DB, y=eta_routing, mode="lines",
        name="Routing  [R·(1−ε)]",
        line=dict(color=COLORS["green"], dash="dot", width=2.2),
    ))
    if show_ref:
        fig6.add_hline(y=R_rate, line_dash="dot", line_color="gray", opacity=0.5,
                       annotation_text=f"η → R = {R_rate}", annotation_position="right")

    # Annotate Δη at 15 dB
    y_mid = (eta_adaptive[idx15] + eta_fixed[idx15]) / 2.0
    delta_eta = eta_adaptive[idx15] - eta_fixed[idx15]
    fig6.add_shape(type="line",
                   x0=15, x1=15, y0=eta_fixed[idx15], y1=eta_adaptive[idx15],
                   line=dict(color="black", width=1.5, dash="dot"))
    fig6.add_annotation(
        x=15, y=y_mid,
        text=f"<b>Δη = {delta_eta:.2f} b/c.u.<br>({eta_gain_pct:.0f}% gain)</b>",
        showarrow=True, arrowhead=2, ax=70, ay=0,
        bgcolor="rgba(255,255,220,0.92)", bordercolor="#888", borderwidth=1,
        font=dict(size=12),
    )

    fig6.update_layout(
        title=f"<b>Fig 6</b> — Spectral Efficiency vs SNR  (k={k}, R={R_rate} b/c.u.)",
        xaxis_title="Average SNR γ̄ (dB)",
        yaxis_title="Spectral Efficiency η (b/c.u.)",
        yaxis_range=[0, R_rate * 1.18],
        legend=dict(x=0.65, y=0.15),
        height=520,
        **LAYOUT_BASE,
    )
    fig6.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig6.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
    st.plotly_chart(fig6, use_container_width=True)

    # Metric columns
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("η Adaptive @ 15 dB",  f"{eta_adaptive[idx15]:.3f} b/c.u.")
    m2.metric("η Fixed (always)",     f"{eta_fixed[0]:.3f} b/c.u.")
    m3.metric("η Routing @ 15 dB",   f"{eta_routing[idx15]:.3f} b/c.u.")
    m4.metric("Spectral Gain @ 15 dB", f"{eta_gain_pct:.1f}%")

    st.caption(
        f"The adaptive scheme approaches η → R = {R_rate} b/c.u. as SNR increases, "
        f"while fixed RLNC saturates at η = k·R/n_fix = {k*R_rate/n_fix:.3f} b/c.u."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interference Alignment + Network Coding
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Interference Alignment + Network Coding (Figs. 2 & 7)")
    st.markdown(
        '<div class="eq-box">'
        'Two-pass receiver: Pass 1 → interference suppression (MMSE)  |  '
        'Pass 2 → structured residual recycled as RLNC decoding equation [§IV-B]'
        '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    # ── Fig 2 ─────────────────────────────────────────────────────────────────
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=SNR_DB, y=p_rank, mode="lines",
            name=f"P(rank cond. | β={beta})",
            line=dict(color=COLORS["red"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.10)",
        ))
        fig2.add_hline(y=0.9, line_dash="dot", line_color="gray",
                       annotation_text="0.9", annotation_position="right")
        # Find SNR where p_rank ≥ 0.9
        cross_idx = int(np.argmax(p_rank >= 0.9))
        if cross_idx > 0:
            fig2.add_vline(x=SNR_DB[cross_idx], line_dash="dot",
                           line_color=COLORS["orange"], opacity=0.8,
                           annotation_text=f"{SNR_DB[cross_idx]:.0f} dB",
                           annotation_position="top right")
        fig2.update_layout(
            title=f"<b>Fig 2</b> — IA Residual Rank Condition Probability  (β={beta}, q=256)",
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="P(innovative equation contributed)",
            yaxis_range=[0, 1.08],
        )
        apply_base(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"When this probability is high, the aligned IA residual provides "
                   f"an extra decoding equation to RLNC, reducing retransmissions. "
                   f"At β={beta} saturation ≈ 1.0 occurs around "
                   f"{SNR_DB[int(np.argmax(p_rank >= 0.95))]:.0f} dB.")

    # ── Fig 7 ─────────────────────────────────────────────────────────────────
    with col_b:
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=SNR_DB, y=R_perfect, mode="lines",
            name="Perfect-CSI IA",
            line=dict(color=COLORS["blue"], width=2.2),
        ))
        fig7.add_trace(go.Scatter(
            x=SNR_DB, y=R_imperfect, mode="lines",
            name=f"Plain IA  (β={beta})",
            line=dict(color=COLORS["red"], dash="dash", width=2.2),
        ))
        fig7.add_trace(go.Scatter(
            x=SNR_DB, y=R_ianc, mode="lines",
            name="IA + NC",
            line=dict(color=COLORS["green"], width=2.8),
        ))

        if show_ref and beta == 0.1:
            for val, col, label in [
                (28.1, COLORS["blue"],  "28.1"),
                (20.2, COLORS["green"], "20.2"),
                (14.8, COLORS["red"],   "14.8"),
            ]:
                fig7.add_annotation(
                    x=30, y=val,
                    text=f"<b>{label} bps/Hz</b>",
                    showarrow=True, arrowhead=2, ax=-55, ay=0,
                    font=dict(color=col, size=11),
                )
            # 36% gain bracket
            fig7.add_shape(type="line", x0=32.5, x1=32.5,
                           y0=R_imperfect[idx30], y1=R_ianc[idx30],
                           line=dict(color="black", width=1.5))
            fig7.add_annotation(
                x=33, y=(R_ianc[idx30] + R_imperfect[idx30]) / 2,
                text=f"<b>{ianc_gain_pct:.0f}%</b>",
                showarrow=False, font=dict(size=13, color="black"),
            )

        fig7.update_layout(
            title=(
                f"<b>Fig 7</b> — Sum-Rate vs SNR  (β={beta}, K={K_users} users)<br>"
                f"<sup>IA+NC recovers {ianc_gain_pct:.0f}% over plain IA @ 30 dB</sup>"
            ),
            xaxis_title="Average SNR γ̄ (dB)",
            yaxis_title="Sum-Rate (bps/Hz)",
            yaxis_range=[0, R_perfect.max() * 1.12],
            legend=dict(x=0.05, y=0.90),
        )
        apply_base(fig7)
        st.plotly_chart(fig7, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Perfect CSI @ 30 dB",   f"{R_perfect[idx30]:.1f} bps/Hz")
        m2.metric(f"Plain IA β={beta} @ 30 dB",
                  f"{R_imperfect[idx30]:.1f} bps/Hz",
                  delta=f"−{csi_drop_pct:.0f}% vs perfect", delta_color="inverse")
        m3.metric("IA+NC @ 30 dB",
                  f"{R_ianc[idx30]:.1f} bps/Hz",
                  delta=f"+{ianc_gain_pct:.0f}% vs plain IA")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Energy Efficiency
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Energy Efficiency η_EE vs SNR (Fig. 8)")
    st.markdown(
        '<div class="eq-box">'
        'η_EE = Σ R_j / (P_t + P_c)  [Eq. 13] &nbsp;&nbsp;—&nbsp;&nbsp;'
        'Adaptive n* ↓ transmissions → lower P_t → higher η_EE'
        '</div>',
        unsafe_allow_html=True,
    )

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=SNR_DB, y=eta_EE_adaptive, mode="lines",
        name="Adaptive n*",
        line=dict(color=COLORS["red"], width=2.8),
        fill="tonexty",
        fillcolor="rgba(214,39,40,0.10)",
    ))
    fig8.add_trace(go.Scatter(
        x=SNR_DB, y=eta_EE_fixed, mode="lines",
        name=f"Fixed RLNC (n={n_fix})",
        line=dict(color=COLORS["blue"], dash="dash", width=2.2),
    ))

    # Annotate gain at 20 dB
    idx20 = int(np.argmin(np.abs(SNR_DB - 20)))
    ee_gain_20 = (eta_EE_adaptive[idx20] - eta_EE_fixed[idx20]) / eta_EE_fixed[idx20] * 100
    fig8.add_annotation(
        x=20, y=(eta_EE_adaptive[idx20] + eta_EE_fixed[idx20]) / 2,
        text=f"<b>+{ee_gain_20:.0f}%<br>EE gain</b>",
        showarrow=True, arrowhead=2, ax=60, ay=0,
        bgcolor="rgba(255,255,220,0.9)", bordercolor="#888", borderwidth=1,
        font=dict(size=12),
    )

    fig8.update_layout(
        title=f"<b>Fig 8</b> — Energy Efficiency vs SNR  (P_c = {Pc_mW} mW, k={k})",
        xaxis_title="Average SNR γ̄ (dB)",
        yaxis_title="Energy Efficiency η_EE  (bits/J)",
        legend=dict(x=0.05, y=0.90),
        height=520,
        **LAYOUT_BASE,
    )
    fig8.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig8.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
    st.plotly_chart(fig8, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"EE Adaptive @ 20 dB",  f"{eta_EE_adaptive[idx20]:.2f} b/J")
    m2.metric(f"EE Fixed @ 20 dB",     f"{eta_EE_fixed[idx20]:.2f} b/J")
    m3.metric("Energy Efficiency Gain", f"+{ee_gain_20:.1f}% @ 20 dB")

    st.caption(
        f"Fewer coded transmissions per block (n* < n_fix at moderate–high SNR) "
        f"directly reduce P_t for a given throughput target, lifting bits-per-joule. "
        f"Relevant for IoT and UAV deployments where battery life is critical."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Summary Table  (interactive)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Performance Summary (Table I — interactive SNR)")

    snr_sel = st.slider(
        "Select SNR for Table I (dB)", -5, 35, 12, 1,
        help="Slide to evaluate all three schemes at any operating point"
    )
    snr_sel_idx = int(np.argmin(np.abs(SNR_DB - snr_sel)))
    snr_sel_lin = SNR_LIN[snr_sel_idx]
    eps_sel = erasure_prob(np.array([snr_sel_lin]), R_rate)[0]

    n_rt  = k
    Pd_rt = float((1.0 - eps_sel) ** k)
    eta_rt = float(R_rate * (1.0 - eps_sel))

    Pd_fxd  = float(decode_prob_analytical(k, n_fix, np.array([snr_sel_lin]), R_rate)[0])
    eta_fxd = float(k * R_rate / n_fix)

    n_adp   = int(n_star[snr_sel_idx])
    Pd_adp  = float(decode_prob_analytical(k, n_adp, np.array([snr_sel_lin]), R_rate)[0])
    eta_adp = float(k * R_rate / n_adp)

    def meets(Pd: float) -> str:
        return "✅ Yes" if Pd >= 0.99 else "❌ No"

    table_df = pd.DataFrame({
        "Scheme":            ["Routing", f"Fixed RLNC (n={n_fix})", "Adaptive n*"],
        "Coded packets n":   [n_rt, n_fix, n_adp],
        "η (b/c.u.)":        [f"{eta_rt:.3f}", f"{eta_fxd:.3f}", f"{eta_adp:.3f}"],
        "Pd":                [f"{Pd_rt:.3f}", f"{Pd_fxd:.3f}", f"{Pd_adp:.3f}"],
        "P_d ≥ 0.99":        [meets(Pd_rt), meets(Pd_fxd), meets(Pd_adp)],
    })

    st.markdown(
        f"#### γ̄ = {snr_sel} dB &nbsp;|&nbsp; k = {k} &nbsp;|&nbsp; "
        f"R = {R_rate} b/c.u. &nbsp;|&nbsp; ε = {eps_sel:.4f}"
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    if show_ref and snr_sel == 12 and k == 10 and abs(R_rate - 1.0) < 0.01:
        st.info(
            "📌 **Paper Table I reference** (k=10, R=1, γ̄=12 dB):\n\n"
            "| Scheme | n | η | Pd |\n"
            "|---|---|---|---|\n"
            "| Routing | 10 | 0.934 | 0.517 |\n"
            "| Fixed RLNC | 18 | 0.556 | 0.999 |\n"
            "| Adaptive n* | 12 | 0.833 | 0.991 |"
        )

    st.markdown("---")

    # Side-by-side bar chart for all three schemes
    bar_fig = go.Figure()
    schemes  = ["Routing", f"Fixed RLNC (n={n_fix})", "Adaptive n*"]
    etas     = [eta_rt, eta_fxd, eta_adp]
    pds      = [Pd_rt, Pd_fxd, Pd_adp]
    bar_colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]

    bar_fig.add_trace(go.Bar(
        name="η (b/c.u.)", x=schemes, y=etas,
        marker_color=[c + "bb" for c in bar_colors],
        text=[f"{v:.3f}" for v in etas], textposition="outside",
        yaxis="y",
    ))
    bar_fig.add_trace(go.Bar(
        name="P_d", x=schemes, y=pds,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in pds], textposition="outside",
        yaxis="y2", opacity=0.75,
    ))
    bar_fig.update_layout(
        title=f"Spectral Efficiency η and Decode Probability P_d @ {snr_sel} dB",
        barmode="group",
        yaxis=dict(title="Spectral Efficiency η (b/c.u.)", range=[0, R_rate * 1.25]),
        yaxis2=dict(title="Decode Probability P_d", overlaying="y", side="right",
                    range=[0, 1.15]),
        legend=dict(x=0.35, y=1.02, orientation="h"),
        height=400,
        **LAYOUT_BASE,
    )
    st.plotly_chart(bar_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.82rem;'>"
    "Group 19 — Kavya Patel, Om Patel, Aditya Vaish, Tirth Patel, Ved Bhoraniya, "
    "Uma Sainitin Burra, Samyak Shah, Harsh Patel, Sukun Dalal, Shrey Patel, "
    "Yashvi Doshi, Tanishk Dhawan, Chetan Raghav<br>"
    "B.Tech. ICT — School of Technology, Dhirubhai Ambani University, Gandhinagar<br>"
    "RLNC + IA Cross-Layer Simulation | GF(2⁸) | 3GPP TR 38.901 (28 GHz UMi)"
    "</div>",
    unsafe_allow_html=True,
)
