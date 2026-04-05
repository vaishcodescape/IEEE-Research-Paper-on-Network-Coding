"""
Simulation 1 — PLNC BER & Throughput on the Two-Way Relay Channel (AWGN)
Produces: Fig 1 (BER vs SNR), Fig 2 (Throughput vs SNR), Fig 3 (Gain factor)

Setup (from paper Sec VIII-A):
  - 3×10^5 random bit pairs per SNR point
  - SNR swept from −4 dB to 21 dB
  - Real AWGN channel, σ² = 1/SNR
  - BPSK: 0 → +1, 1 → −1
  - Routing uses 4 slots; PLNC uses 2 slots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

NUM_TRIALS = 300_000
SNR_DB = np.arange(-4, 22, 1.0)
SNR_LIN = 10 ** (SNR_DB / 10)


def bpsk_map(bits):
    return 1.0 - 2.0 * bits


def simulate_routing(snr_lin, n_trials):
    """4-slot routing: A→R, R→B, B→R, R→A.  Each hop is independent BPSK/AWGN."""
    sigma = 1.0 / np.sqrt(snr_lin)
    a = np.random.randint(0, 2, n_trials)
    b = np.random.randint(0, 2, n_trials)

    # A→R  then  R→B  (relay decodes-and-forwards a)
    x_a = bpsk_map(a)
    y_ar = x_a + sigma * np.random.randn(n_trials)
    a_hat_relay = (y_ar < 0).astype(int)
    x_af = bpsk_map(a_hat_relay)
    y_rb = x_af + sigma * np.random.randn(n_trials)
    a_hat_b = (y_rb < 0).astype(int)

    # B→R  then  R→A  (relay decodes-and-forwards b)
    x_b = bpsk_map(b)
    y_br = x_b + sigma * np.random.randn(n_trials)
    b_hat_relay = (y_br < 0).astype(int)
    x_bf = bpsk_map(b_hat_relay)
    y_ra = x_bf + sigma * np.random.randn(n_trials)
    b_hat_a = (y_ra < 0).astype(int)

    err_a = np.sum(b_hat_a != b)
    err_b = np.sum(a_hat_b != a)
    ber = (err_a + err_b) / (2 * n_trials)
    throughput = (1.0 - ber) * 0.5          # 2 bits / 4 slots × (1−BER)
    return ber, throughput


def simulate_plnc(snr_lin, n_trials):
    """2-slot PLNC: MAC phase (A+B→R) then BC phase (R→A,B)."""
    sigma = 1.0 / np.sqrt(snr_lin)
    a = np.random.randint(0, 2, n_trials)
    b = np.random.randint(0, 2, n_trials)

    x_a = bpsk_map(a)
    x_b = bpsk_map(b)

    # Slot 1 — MAC phase: relay sees y_R = x_A + x_B + n
    y_r = x_a + x_b + sigma * np.random.randn(n_trials)
    xor_hat = (np.abs(y_r) <= 1.0).astype(int)     # |y|≤1 ⇒ XOR=1

    # Slot 2 — BC phase: relay broadcasts BPSK-mapped XOR
    x_xor = bpsk_map(xor_hat)
    y_bc_a = x_xor + sigma * np.random.randn(n_trials)
    xor_rx_a = (y_bc_a < 0).astype(int)
    y_bc_b = x_xor + sigma * np.random.randn(n_trials)
    xor_rx_b = (y_bc_b < 0).astype(int)

    b_hat = xor_rx_a ^ a
    a_hat = xor_rx_b ^ b

    xor_true = a ^ b
    relay_err = np.mean(xor_hat != xor_true)

    err_a = np.sum(a_hat != a)
    err_b = np.sum(b_hat != b)
    ber = (err_a + err_b) / (2 * n_trials)
    throughput = (1.0 - ber) * 1.0          # 2 bits / 2 slots × (1−BER)
    return ber, throughput, relay_err


def theoretical_routing_ber(snr_lin):
    """Two cascaded BPSK hops: P_e ≈ 2Q(√SNR)(1−Q(√SNR)) ≈ 2Q(√SNR)."""
    pe_hop = 0.5 * erfc(np.sqrt(snr_lin / 2))
    return 2 * pe_hop * (1 - pe_hop)


def theoretical_plnc_ber(snr_lin):
    """PLNC end-to-end BER: MAC-phase XOR detection + BC-phase BPSK.

    MAC phase: relay threshold detector on y_R = x_A + x_B + n.
      P_relay = 1.5·Q(1/σ) − 0.5·Q(3/σ)  with σ = 1/√SNR.
    BC phase: standard BPSK hop  P_bc = Q(1/σ).
    Cascade: P_e = P_relay + P_bc − 2·P_relay·P_bc.
    """
    p_relay = (0.75 * erfc(np.sqrt(snr_lin / 2))
               - 0.25 * erfc(np.sqrt(9.0 * snr_lin / 2)))
    p_bc = 0.5 * erfc(np.sqrt(snr_lin / 2))
    return p_relay + p_bc - 2.0 * p_relay * p_bc


# ── Run simulations ──────────────────────────────────────────────────────────
ber_rt = np.zeros_like(SNR_DB)
thr_rt = np.zeros_like(SNR_DB)
ber_plnc = np.zeros_like(SNR_DB)
thr_plnc = np.zeros_like(SNR_DB)
relay_xor_ber = np.zeros_like(SNR_DB)

for i, snr in enumerate(SNR_LIN):
    ber_rt[i], thr_rt[i] = simulate_routing(snr, NUM_TRIALS)
    ber_plnc[i], thr_plnc[i], relay_xor_ber[i] = simulate_plnc(snr, NUM_TRIALS)

ber_rt_theory = theoretical_routing_ber(SNR_LIN)
ber_plnc_theory = theoretical_plnc_ber(SNR_LIN)

# ── Fig 1: BER vs SNR ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.semilogy(SNR_DB, ber_rt, 'bo', ms=5, label='Routing (sim)')
ax1.semilogy(SNR_DB, ber_rt_theory, 'b-', lw=1.2, label='Routing (theory)')
ax1.semilogy(SNR_DB, ber_plnc, 'rs', ms=5, label='PLNC end-to-end (sim)')
ax1.semilogy(SNR_DB, ber_plnc_theory, 'r-', lw=1.2, label='PLNC (theory)')
ax1.semilogy(SNR_DB, relay_xor_ber, 'g^', ms=5, label='Relay XOR BER')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('BER')
ax1.set_title('Fig 1 — BER vs SNR (TWRC, AWGN)')
ax1.legend()
ax1.grid(True, which='both', ls=':', alpha=0.5)
ax1.set_ylim(bottom=1e-6)
fig1.tight_layout()
fig1.savefig('fig1_ber_vs_snr.png', dpi=200)

# ── Fig 2: Throughput vs SNR ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.fill_between(SNR_DB, thr_rt, thr_plnc,
                 alpha=0.15, color='green', label='PLNC gain region')
ax2.plot(SNR_DB, thr_rt, 'b-o', ms=4, lw=1.2, label='Routing (0.5 b/slot max)')
ax2.plot(SNR_DB, thr_plnc, 'r-s', ms=4, lw=1.2, label='PLNC (1.0 b/slot max)')
ax2.axhline(0.5, ls=':', color='blue', alpha=0.5)
ax2.axhline(1.0, ls=':', color='red', alpha=0.5)
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('Throughput (bits / time slot)')
ax2.set_title('Fig 2 — Effective Throughput vs SNR')
ax2.legend()
ax2.grid(True, ls=':', alpha=0.5)
fig2.tight_layout()
fig2.savefig('fig2_throughput_vs_snr.png', dpi=200)

# ── Fig 3: Gain factor vs SNR ────────────────────────────────────────────────
gain = np.where(thr_rt > 1e-9, thr_plnc / thr_rt, 0)
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(SNR_DB, gain, 'k-o', ms=4, lw=1.2)
ax3.axhline(2.0, ls=':', color='gray', label='Theoretical max 2.0×')
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('Throughput gain (PLNC / Routing)')
ax3.set_title('Fig 3 — Throughput Gain Factor vs SNR')
ax3.legend()
ax3.grid(True, ls=':', alpha=0.5)
fig3.tight_layout()
fig3.savefig('fig3_gain_factor.png', dpi=200)

plt.show()
print("Done — saved fig1, fig2, fig3.")
