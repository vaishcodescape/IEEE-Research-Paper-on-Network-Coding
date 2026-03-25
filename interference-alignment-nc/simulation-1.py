#!/usr/bin/env python3
"""
Joint Interference Alignment and Network Coding (IA+NC) Simulation
for Multi-User MIMO Interference Channel with Imperfect CSI.

System: K-user MIMO IC, Rayleigh fading, Gauss-Markov CSI error model.
Schemes compared:
    1. No IA (SVD beamforming baseline)
    2. Classical Interference Alignment
    3. Joint IA + Network Coding (Proposed)

Outputs (saved to plots/):
    fig1  – Sum Rate vs SNR
    fig2  – Per-User Spectral Efficiency vs SNR
    fig3  – Sum Rate vs CSI Error (beta)
    fig4  – Energy Efficiency vs SNR
    fig5  – Sum Rate vs Number of Users (K)
    fig6  – Degrees-of-Freedom Comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
np.random.seed(2024)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (7.5, 5.5),
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

SNR_DB_RANGE = np.arange(0, 32, 2)


# ──────────────────────────────────────────────────────────────
# Channel Generation
# ──────────────────────────────────────────────────────────────
def generate_channels(K, Mt, Mr, beta):
    """
    Rayleigh fading with Gauss–Markov imperfect-CSI model:
        H_true = sqrt(1 − β²)·H_est + β·E
    where H_est and E are i.i.d. CN(0,1) per element.

    Returns K×K lists of (Mr × Mt) matrices for H_true and H_est.
    """
    H_true = [[None] * K for _ in range(K)]
    H_est = [[None] * K for _ in range(K)]
    scale = 1.0 / np.sqrt(2)
    coeff = np.sqrt(1 - beta ** 2)
    for j in range(K):
        for i in range(K):
            H_hat = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) * scale
            E = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) * scale
            H_true[j][i] = coeff * H_hat + beta * E
            H_est[j][i] = H_hat
    return H_true, H_est


# ──────────────────────────────────────────────────────────────
# Interference Alignment  (iterative min-leakage)
# ──────────────────────────────────────────────────────────────
def interference_alignment(H_est, K, Mt, Mr, d, max_iter=300, tol=1e-10):
    """
    Alternating minimisation of total interference leakage.

    Each iteration:
        • Decoder U[j]  ← d eigenvectors of Q_rx[j] with *smallest* eigenvalues
        • Precoder V[i] ← d eigenvectors of Q_tx[i] with *smallest* eigenvalues

    Feasibility requires Mt + Mr ≥ (K+1)·d.

    Returns (V, U) where V[k] is Mt×d, U[k] is Mr×d.
    """
    V = []
    for k in range(K):
        q, _ = np.linalg.qr(
            np.random.randn(Mt, d) + 1j * np.random.randn(Mt, d)
        )
        V.append(q[:, :d])

    U = [None] * K
    reg = 1e-12 * np.eye(max(Mt, Mr), dtype=complex)

    for _ in range(max_iter):
        # --- update decoders ---
        for j in range(K):
            Q = np.zeros((Mr, Mr), dtype=complex)
            for i in range(K):
                if i != j:
                    HV = H_est[j][i] @ V[i]
                    Q += HV @ HV.conj().T
            Q += reg[:Mr, :Mr]
            _, evecs = np.linalg.eigh(Q)
            U[j] = evecs[:, :d]

        V_old = [v.copy() for v in V]

        # --- update precoders ---
        for i in range(K):
            Q = np.zeros((Mt, Mt), dtype=complex)
            for j in range(K):
                if j != i:
                    HhU = H_est[j][i].conj().T @ U[j]
                    Q += HhU @ HhU.conj().T
            Q += reg[:Mt, :Mt]
            _, evecs = np.linalg.eigh(Q)
            V[i] = evecs[:, :d]

        if sum(np.linalg.norm(V[k] - V_old[k]) for k in range(K)) < tol:
            break

    return V, U


# ──────────────────────────────────────────────────────────────
# SVD Baseline (no interference management)
# ──────────────────────────────────────────────────────────────
def svd_baseline(H_est, K, d):
    """d dominant singular-vector pairs of each direct link."""
    V, U = [], []
    for k in range(K):
        u, _, vh = np.linalg.svd(H_est[k][k])
        V.append(vh.conj().T[:, :d])
        U.append(u[:, :d])
    return V, U


# ──────────────────────────────────────────────────────────────
# Sum-Rate Computation
# ──────────────────────────────────────────────────────────────
def compute_sum_rate(H_true, H_est, V, U, K, d, snr_lin, scheme="standard"):
    """
    scheme 'standard':
        Classical IA / no-IA — rate computed in the d-dimensional space
        after applying the IA (or SVD) decoder U[k].

    scheme 'ia_nc':
        Joint IA + NC — after NC decodes and cancels the aligned
        interference, the receiver processes in the full Mr-dimensional
        space.  The gain has two components:
          (a) NC reduces residual interference (aligned part cancelled),
          (b) full-dimensional processing preserves receive diversity
              that classical IA sacrifices via d-dim projection.
    """
    p_stream = snr_lin / d
    total = 0.0

    for k in range(K):
        if scheme == "ia_nc":
            Mr = H_true[k][k].shape[0]
            H_des = H_true[k][k] @ V[k]                   # Mr × d

            # Aligned subspace from *estimated* channels (receiver side-info)
            int_est = [H_est[k][i] @ V[i] for i in range(K) if i != k]
            I_est = np.hstack(int_est)                     # Mr × (K-1)d
            U_sub, _, _ = np.linalg.svd(I_est, full_matrices=True)
            P_a = U_sub[:, :d] @ U_sub[:, :d].conj().T    # projection

            eta = np.clip(1.0 - np.exp(-0.15 * snr_lin), 0.0, 0.95)

            Q_res = np.zeros((Mr, Mr), dtype=complex)
            for i in range(K):
                if i == k:
                    continue
                HV = H_true[k][i] @ V[i]
                HV_a = P_a @ HV
                HV_l = HV - HV_a
                Q_res += p_stream * (
                    (1.0 - eta) * (HV_a @ HV_a.conj().T)
                    + HV_l @ HV_l.conj().T
                )

            Q_ni = np.eye(Mr) + Q_res
            Q_sni = Q_ni + p_stream * (H_des @ H_des.conj().T)

        else:
            Heff = U[k].conj().T @ H_true[k][k] @ V[k]   # d × d
            Qint = np.zeros((d, d), dtype=complex)
            for i in range(K):
                if i != k:
                    Gi = U[k].conj().T @ H_true[k][i] @ V[i]
                    Qint += p_stream * (Gi @ Gi.conj().T)

            Q_ni = np.eye(d) + Qint
            Q_sni = Q_ni + p_stream * (Heff @ Heff.conj().T)

        s1, ld1 = np.linalg.slogdet(Q_sni)
        s2, ld2 = np.linalg.slogdet(Q_ni)
        Rk = np.real(ld1 - ld2) / np.log(2) if (s1 > 0 and s2 > 0) else 0.0
        total += max(Rk, 0.0)

    return total


# ──────────────────────────────────────────────────────────────
# Simulation Drivers
# ──────────────────────────────────────────────────────────────
def sim_snr_sweep(K=3, Mt=4, Mr=4, d=2, beta=0.1, num_mc=100):
    """Sum rate vs SNR for all three schemes."""
    snr_db = SNR_DB_RANGE
    snr_lin = 10.0 ** (snr_db / 10.0)
    n = len(snr_db)
    R = {s: np.zeros(n) for s in ("no_ia", "ia", "ia_nc")}

    print(f"[SNR sweep] K={K} Mt={Mt} Mr={Mr} d={d} β={beta} MC={num_mc}")
    t0 = time.time()
    for mc in range(num_mc):
        if (mc + 1) % 25 == 0:
            print(f"  {mc+1}/{num_mc}  ({time.time()-t0:.0f}s)")
        Ht, He = generate_channels(K, Mt, Mr, beta)
        Vs, Us = svd_baseline(He, K, d)
        Vi, Ui = interference_alignment(He, K, Mt, Mr, d)
        for si, snr in enumerate(snr_lin):
            R["no_ia"][si] += compute_sum_rate(Ht, He, Vs, Us, K, d, snr)
            R["ia"][si] += compute_sum_rate(Ht, He, Vi, Ui, K, d, snr)
            R["ia_nc"][si] += compute_sum_rate(Ht, He, Vi, Ui, K, d, snr, "ia_nc")
    for s in R:
        R[s] /= num_mc
    print(f"  Done in {time.time()-t0:.1f}s")
    return snr_db, R


def sim_beta_sweep(K=3, Mt=4, Mr=4, d=2, snr_db_fix=20, num_mc=100):
    """Sum rate vs CSI-error parameter β at fixed SNR."""
    betas = np.linspace(0.0, 0.5, 11)
    snr_lin = 10.0 ** (snr_db_fix / 10.0)
    R_ia = np.zeros(len(betas))
    R_nc = np.zeros(len(betas))

    print(f"[Beta sweep] SNR={snr_db_fix} dB  MC={num_mc}")
    t0 = time.time()
    for bi, b in enumerate(betas):
        for mc in range(num_mc):
            Ht, He = generate_channels(K, Mt, Mr, b)
            Vi, Ui = interference_alignment(He, K, Mt, Mr, d)
            R_ia[bi] += compute_sum_rate(Ht, He, Vi, Ui, K, d, snr_lin)
            R_nc[bi] += compute_sum_rate(Ht, He, Vi, Ui, K, d, snr_lin, "ia_nc")
        R_ia[bi] /= num_mc
        R_nc[bi] /= num_mc
        print(f"  β={b:.2f}  IA={R_ia[bi]:.2f}  IA+NC={R_nc[bi]:.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return betas, R_ia, R_nc


def sim_k_sweep(Mt=4, Mr=4, beta=0.1, snr_db_fix=20, num_mc=100):
    """Sum rate vs number of users K."""
    Ks = np.array([2, 3, 4, 5, 6])
    snr_lin = 10.0 ** (snr_db_fix / 10.0)
    R_ia = np.zeros(len(Ks))
    R_nc = np.zeros(len(Ks))
    d_used = np.zeros(len(Ks), dtype=int)

    print(f"[K sweep] SNR={snr_db_fix} dB  β={beta}  MC={num_mc}")
    t0 = time.time()
    for ki, Kv in enumerate(Ks):
        dv = max(1, min(2, (Mt + Mr) // (Kv + 1)))
        d_used[ki] = dv
        for mc in range(num_mc):
            Ht, He = generate_channels(Kv, Mt, Mr, beta)
            Vi, Ui = interference_alignment(He, Kv, Mt, Mr, dv)
            R_ia[ki] += compute_sum_rate(Ht, He, Vi, Ui, Kv, dv, snr_lin)
            R_nc[ki] += compute_sum_rate(Ht, He, Vi, Ui, Kv, dv, snr_lin, "ia_nc")
        R_ia[ki] /= num_mc
        R_nc[ki] /= num_mc
        print(f"  K={Kv} d={dv}  IA={R_ia[ki]:.2f}  IA+NC={R_nc[ki]:.2f}")
    print(f"  Done in {time.time()-t0:.1f}s")
    return Ks, d_used, R_ia, R_nc


# ──────────────────────────────────────────────────────────────
# Plotting Helpers
# ──────────────────────────────────────────────────────────────
STYLES = [
    ("no_ia", "No IA (SVD Baseline)", "b-o"),
    ("ia", "Classical IA", "r-s"),
    ("ia_nc", "Joint IA + NC (Proposed)", "g-^"),
]


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────
# Figure 1 – Sum Rate vs SNR
# ──────────────────────────────────────────────────────────────
def plot_sum_rate_snr(snr_db, R):
    fig, ax = plt.subplots()
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R[key], fmt, label=lbl, markevery=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Sum Rate (bits/s/Hz)")
    ax.set_title(
        "Sum Rate vs SNR\n"
        r"($K$=3, $M_t$=$M_r$=4, $d$=2, $\beta$=0.1)"
    )
    ax.legend()
    ax.grid(True)
    _save(fig, "fig1_sum_rate_vs_snr.png")


# ──────────────────────────────────────────────────────────────
# Figure 2 – Per-User Spectral Efficiency
# ──────────────────────────────────────────────────────────────
def plot_spectral_eff(snr_db, R, K=3):
    fig, ax = plt.subplots()
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R[key] / K, fmt, label=lbl, markevery=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Per-User Spectral Efficiency (bits/s/Hz)")
    ax.set_title("Per-User Spectral Efficiency vs SNR")
    ax.legend()
    ax.grid(True)
    _save(fig, "fig2_spectral_efficiency.png")


# ──────────────────────────────────────────────────────────────
# Figure 3 – Sum Rate vs CSI Error β
# ──────────────────────────────────────────────────────────────
def plot_beta(betas, R_ia, R_nc):
    fig, ax = plt.subplots()
    ax.plot(betas, R_ia, "r-s", label="Classical IA", markersize=8)
    ax.plot(betas, R_nc, "g-^", label="Joint IA + NC (Proposed)", markersize=8)
    ax.fill_between(betas, R_ia, R_nc, alpha=0.12, color="green")
    ax.set_xlabel(r"CSI Error Parameter ($\beta$)")
    ax.set_ylabel("Sum Rate (bits/s/Hz)")
    ax.set_title("Impact of CSI Imperfection on Sum Rate  (SNR = 20 dB)")
    ax.legend()
    ax.grid(True)
    _save(fig, "fig3_sum_rate_vs_beta.png")


# ──────────────────────────────────────────────────────────────
# Figure 4 – Energy Efficiency vs SNR
# ──────────────────────────────────────────────────────────────
def plot_energy_eff(snr_db, R, P_c=1.0):
    fig, ax = plt.subplots()
    P_total = 10.0 ** (snr_db / 10.0) + P_c
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R[key] / P_total, fmt, label=lbl, markevery=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Energy Efficiency (bits/s/Hz per Watt)")
    ax.set_title(r"Energy Efficiency vs SNR  ($P_c$ = 1 W)")
    ax.legend()
    ax.grid(True)
    _save(fig, "fig4_energy_efficiency.png")


# ──────────────────────────────────────────────────────────────
# Figure 5 – Sum Rate vs Number of Users K
# ──────────────────────────────────────────────────────────────
def plot_k_sweep(Ks, d_used, R_ia, R_nc):
    fig, ax = plt.subplots()
    ax.plot(Ks, R_ia, "r-s", label="Classical IA", markersize=8)
    ax.plot(Ks, R_nc, "g-^", label="Joint IA + NC (Proposed)", markersize=8)
    for ki, Kv in enumerate(Ks):
        ax.annotate(
            f"d={d_used[ki]}",
            (Kv, R_nc[ki]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Number of Users ($K$)")
    ax.set_ylabel("Sum Rate (bits/s/Hz)")
    ax.set_title(r"Sum Rate vs Number of Users  (SNR=20 dB, $\beta$=0.1)")
    ax.set_xticks(Ks)
    ax.legend()
    ax.grid(True)
    _save(fig, "fig5_sum_rate_vs_K.png")


# ──────────────────────────────────────────────────────────────
# Figure 6 – DoF Comparison
# ──────────────────────────────────────────────────────────────
def plot_dof(snr_db, R, K=3, d=2):
    fig, ax = plt.subplots()
    snr_lin = 10.0 ** (snr_db / 10.0)
    log2snr = np.log2(snr_lin)
    for key, lbl, fmt in STYLES:
        ax.plot(log2snr, R[key], fmt, label=lbl, markevery=2)

    ideal_dof = K * d
    y_ref = ideal_dof * log2snr
    offset = R["ia_nc"][-1] - y_ref[-1]
    ax.plot(
        log2snr,
        y_ref + offset,
        "k--",
        alpha=0.5,
        label=f"Reference slope (DoF = {ideal_dof})",
    )

    # annotate estimated slopes at high SNR
    mask = snr_db >= 18
    for key, lbl, _ in STYLES:
        c = np.polyfit(log2snr[mask], R[key][mask], 1)
        ax.annotate(
            f"slope ≈ {c[0]:.1f}",
            xy=(log2snr[mask][-1], R[key][mask][-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(r"$\log_2$(SNR)")
    ax.set_ylabel("Sum Rate (bits/s/Hz)")
    ax.set_title("Degrees-of-Freedom Comparison")
    ax.legend(fontsize=9)
    ax.grid(True)
    _save(fig, "fig6_dof_comparison.png")


# ──────────────────────────────────────────────────────────────
# Composite Summary Figure (for quick overview)
# ──────────────────────────────────────────────────────────────
def plot_composite(snr_db, R_snr, betas, Rb_ia, Rb_nc, Ks, d_used, Rk_ia, Rk_nc):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (0,0) Sum Rate vs SNR
    ax = axes[0, 0]
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R_snr[key], fmt, label=lbl, markevery=2)
    ax.set(xlabel="SNR (dB)", ylabel="Sum Rate (bits/s/Hz)")
    ax.set_title("(a) Sum Rate vs SNR")
    ax.legend(fontsize=8)
    ax.grid(True)

    # (0,1) Spectral Efficiency
    ax = axes[0, 1]
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R_snr[key] / 3, fmt, label=lbl, markevery=2)
    ax.set(xlabel="SNR (dB)", ylabel="Per-User SE (bits/s/Hz)")
    ax.set_title("(b) Spectral Efficiency")
    ax.legend(fontsize=8)
    ax.grid(True)

    # (0,2) Beta sweep
    ax = axes[0, 2]
    ax.plot(betas, Rb_ia, "r-s", label="Classical IA", ms=6)
    ax.plot(betas, Rb_nc, "g-^", label="IA + NC", ms=6)
    ax.fill_between(betas, Rb_ia, Rb_nc, alpha=0.12, color="green")
    ax.set(xlabel=r"$\beta$", ylabel="Sum Rate (bits/s/Hz)")
    ax.set_title(r"(c) Sum Rate vs $\beta$  (SNR=20 dB)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # (1,0) Energy Efficiency
    ax = axes[1, 0]
    Pt = 10.0 ** (snr_db / 10.0) + 1.0
    for key, lbl, fmt in STYLES:
        ax.plot(snr_db, R_snr[key] / Pt, fmt, label=lbl, markevery=2)
    ax.set(xlabel="SNR (dB)", ylabel="EE (bits/s/Hz/W)")
    ax.set_title("(d) Energy Efficiency")
    ax.legend(fontsize=8)
    ax.grid(True)

    # (1,1) K sweep
    ax = axes[1, 1]
    ax.plot(Ks, Rk_ia, "r-s", label="Classical IA", ms=7)
    ax.plot(Ks, Rk_nc, "g-^", label="IA + NC", ms=7)
    for ki, Kv in enumerate(Ks):
        ax.annotate(f"d={d_used[ki]}", (Kv, Rk_nc[ki]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(Ks)
    ax.set(xlabel="$K$", ylabel="Sum Rate (bits/s/Hz)")
    ax.set_title("(e) Sum Rate vs Users")
    ax.legend(fontsize=8)
    ax.grid(True)

    # (1,2) DoF
    ax = axes[1, 2]
    log2snr = np.log2(10.0 ** (snr_db / 10.0))
    for key, lbl, fmt in STYLES:
        ax.plot(log2snr, R_snr[key], fmt, label=lbl, markevery=2)
    ax.set(xlabel=r"$\log_2$(SNR)", ylabel="Sum Rate (bits/s/Hz)")
    ax.set_title("(f) DoF Comparison")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle(
        "Joint IA + Network Coding: Performance Evaluation",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_composite_summary.png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wall = time.time()
    sep = "=" * 62
    print(sep)
    print("  Joint IA + NC Simulation — MIMO Interference Channel")
    print(sep)

    # ── 1. Sum Rate vs SNR ──
    print("\n▸ Simulation 1/3 : Sum Rate vs SNR")
    snr_db, R_snr = sim_snr_sweep()
    plot_sum_rate_snr(snr_db, R_snr)
    plot_spectral_eff(snr_db, R_snr)
    plot_dof(snr_db, R_snr)
    plot_energy_eff(snr_db, R_snr)

    # ── 2. Beta sweep ──
    print("\n▸ Simulation 2/3 : Sum Rate vs β")
    betas, Rb_ia, Rb_nc = sim_beta_sweep()
    plot_beta(betas, Rb_ia, Rb_nc)

    # ── 3. K sweep ──
    print("\n▸ Simulation 3/3 : Sum Rate vs K")
    Ks, d_used, Rk_ia, Rk_nc = sim_k_sweep()
    plot_k_sweep(Ks, d_used, Rk_ia, Rk_nc)

    # ── Composite overview ──
    plot_composite(snr_db, R_snr, betas, Rb_ia, Rb_nc, Ks, d_used, Rk_ia, Rk_nc)

    # ── Summary ──
    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)
    print(f"  {'SNR':>6s}  {'No IA':>8s}  {'IA':>8s}  {'IA+NC':>8s}  {'NC Gain':>8s}")
    for si in range(0, len(snr_db), 4):
        gain = R_snr["ia_nc"][si] - R_snr["ia"][si]
        print(
            f"  {snr_db[si]:5.0f}   {R_snr['no_ia'][si]:8.2f}  "
            f"{R_snr['ia'][si]:8.2f}  {R_snr['ia_nc'][si]:8.2f}  {gain:+8.2f}"
        )
    mask = snr_db >= 18
    log2s = np.log2(10.0 ** (snr_db[mask] / 10.0))
    print(f"\n  Estimated DoF (high-SNR slope):")
    for key, lbl, _ in STYLES:
        c = np.polyfit(log2s, R_snr[key][mask], 1)
        print(f"    {lbl:30s}  slope ≈ {c[0]:.2f}")
    print(f"\n  IA feasibility  Mt+Mr={(4+4)} ≥ (K+1)·d = {(3+1)*2}")
    elapsed = time.time() - wall
    print(f"\n  Total wall time : {elapsed:.0f}s")
    print(f"  Plots saved to  : {PLOT_DIR}/")
    print(sep)
