"""
Simulation 2 — Interference Alignment + Network Coding under CSI Uncertainty
Produces: Fig 4 (Sum-rate vs SNR), Fig 5 (Sum-rate vs β), Table II

Setup (from paper Sec VIII-B):
  - (4×4, 2)^3 MIMO: K=3 users, Mt=Mr=4, d=2 streams each
  - Gauss-Markov CSI error: H = √(1−β²)·Hc + β·E
  - 500 channel realisations per SNR point
  - Iterative min-leakage IA precoder, max 200 iters
  - Schemes: ZF (β=0), Classical IA (β=0,0.1), Joint IA+NC (β=0,0.1)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = 3           # users
Mt = 4          # Tx antennas
Mr = 4          # Rx antennas
d = 2           # streams per user
N_REAL = 500    # channel realisations
MAX_ITER = 200
LEAK_TOL = 1e-4

SNR_DB = np.arange(0, 35, 5.0)
SNR_LIN = 10 ** (SNR_DB / 10)
BETA_RANGE = np.linspace(0, 0.5, 11)


def generate_channels(K, Mr, Mt, beta=0.0):
    """Draw true and estimated channels with Gauss-Markov CSI error."""
    Hc = {}
    H_true = {}
    for j in range(K):
        for i in range(K):
            hc = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
            E = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
            Hc[(j, i)] = hc
            H_true[(j, i)] = np.sqrt(1 - beta**2) * hc + beta * E
    return H_true, Hc


def init_precoders(K, Mt, d):
    V = {}
    for i in range(K):
        Q, _ = np.linalg.qr((np.random.randn(Mt, d) + 1j * np.random.randn(Mt, d)))
        V[i] = Q[:, :d]
    return V


def min_leakage_ia(H_est, K, Mr, Mt, d, max_iter=MAX_ITER, tol=LEAK_TOL):
    """Iterative min-leakage IA on estimated channels."""
    V = init_precoders(K, Mt, d)
    for iteration in range(max_iter):
        U = {}
        for j in range(K):
            Q_int = np.zeros((Mr, Mr), dtype=complex)
            for i in range(K):
                if i != j:
                    HV = H_est[(j, i)] @ V[i]
                    Q_int += HV @ HV.conj().T
            eigvals, eigvecs = np.linalg.eigh(Q_int)
            U[j] = eigvecs[:, :d]

        V_new = {}
        total_leak = 0.0
        for i in range(K):
            Q_int = np.zeros((Mt, Mt), dtype=complex)
            for j in range(K):
                if j != i:
                    HU = H_est[(j, i)].conj().T @ U[j]
                    Q_int += HU @ HU.conj().T
            eigvals, eigvecs = np.linalg.eigh(Q_int)
            V_new[i] = eigvecs[:, :d]
            total_leak += np.sum(eigvals[:d]).real

        prev_V = V
        V = V_new
        if total_leak < tol:
            break

    return V, U


def compute_sum_rate(H, V, U, K, d, sigma2):
    """Sum rate after IA: R_j = log2 det(I + (1/σ²) H_eff Q H_eff^H)."""
    total = 0.0
    for j in range(K):
        H_eff = U[j].conj().T @ H[(j, j)] @ V[j]
        S = np.eye(d) + (1.0 / sigma2) * (H_eff @ H_eff.conj().T)
        total += np.log2(np.abs(np.linalg.det(S)))
    return total


def compute_ia_nc_rate(H, V, U, K, d, sigma2):
    """IA+NC: desired signal rate + rank-gain from structured interference."""
    rate = compute_sum_rate(H, V, U, K, d, sigma2)
    for j in range(K):
        P_j = np.eye(Mr) - U[j] @ U[j].conj().T
        for i in range(K):
            if i != j:
                z = P_j @ H[(j, i)] @ V[i]
                extra_rank = np.linalg.matrix_rank(z, tol=np.sqrt(sigma2))
                if extra_rank > 0:
                    snr_int = np.linalg.norm(z, 'fro')**2 / (d * sigma2)
                    rate += min(extra_rank, 1) * np.log2(1 + snr_int / extra_rank)
    return rate


def compute_zf_rate(H, K, Mt, d, sigma2):
    """Zero-forcing rate (perfect CSI only)."""
    total = 0.0
    for j in range(K):
        interf_cols = []
        for i in range(K):
            if i != j:
                interf_cols.append(H[(j, i)] @ np.eye(Mt)[:, :d])
        B = np.hstack(interf_cols)
        Ub, Sb, _ = np.linalg.svd(B, full_matrices=True)
        null_start = min(B.shape[1], Mr)
        W = Ub[:, null_start:]
        if W.shape[1] == 0:
            continue
        H_eff = W.conj().T @ H[(j, j)]
        H_eff = H_eff[:d, :d]
        S = np.eye(min(d, H_eff.shape[0])) + (1.0 / sigma2) * (H_eff @ H_eff.conj().T)
        total += np.log2(np.abs(np.linalg.det(S)))
    return total


# ── Main simulation loop ─────────────────────────────────────────────────────
results = {
    'zf_b0': np.zeros(len(SNR_DB)),
    'ia_b0': np.zeros(len(SNR_DB)),
    'ianc_b0': np.zeros(len(SNR_DB)),
    'ia_b01': np.zeros(len(SNR_DB)),
    'ianc_b01': np.zeros(len(SNR_DB)),
}

print("Running IA/NC simulation (this takes a few minutes)...")
for si, snr in enumerate(SNR_LIN):
    sigma2 = 1.0 / snr
    accum = {k: 0.0 for k in results}
    for _ in range(N_REAL):
        # β = 0 (perfect CSI)
        H0, Hc0 = generate_channels(K, Mr, Mt, beta=0.0)
        V0, U0 = min_leakage_ia(Hc0, K, Mr, Mt, d)
        accum['zf_b0'] += compute_zf_rate(H0, K, Mt, d, sigma2)
        accum['ia_b0'] += compute_sum_rate(H0, V0, U0, K, d, sigma2)
        accum['ianc_b0'] += compute_ia_nc_rate(H0, V0, U0, K, d, sigma2)

        # β = 0.1
        H1, Hc1 = generate_channels(K, Mr, Mt, beta=0.1)
        V1, U1 = min_leakage_ia(Hc1, K, Mr, Mt, d)
        accum['ia_b01'] += compute_sum_rate(H1, V1, U1, K, d, sigma2)
        accum['ianc_b01'] += compute_ia_nc_rate(H1, V1, U1, K, d, sigma2)

    for k in results:
        results[k][si] = accum[k] / N_REAL
    print(f"  SNR={SNR_DB[si]:5.1f} dB done")

# ── Table II ──────────────────────────────────────────────────────────────────
print("\n=== Table II: System Sum-Rate (bps/Hz) ===")
header = f"{'Scheme':<30s} {'10 dB':>8s} {'20 dB':>8s} {'30 dB':>8s}"
print(header)
print("-" * len(header))
for label, key in [('Zero-Forcing (β=0)', 'zf_b0'),
                   ('Classical IA (β=0)', 'ia_b0'),
                   ('Joint IA+NC (β=0)', 'ianc_b0'),
                   ('Classical IA (β=0.1)', 'ia_b01'),
                   ('Joint IA+NC (β=0.1)', 'ianc_b01')]:
    vals = [results[key][np.argmin(np.abs(SNR_DB - s))] for s in [10, 20, 30]]
    print(f"{label:<30s} {vals[0]:8.1f} {vals[1]:8.1f} {vals[2]:8.1f}")

# ── Fig 4: Sum-rate vs SNR (β = 0.1) ─────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(SNR_DB, results['ia_b01'], 'b-o', ms=5, label='Classical IA (β=0.1)')
ax4.plot(SNR_DB, results['ianc_b01'], 'r-s', ms=5, label='Joint IA+NC (β=0.1)')
ax4.plot(SNR_DB, results['ia_b0'], 'b--^', ms=4, alpha=0.5, label='Classical IA (β=0)')
ax4.plot(SNR_DB, results['ianc_b0'], 'r--v', ms=4, alpha=0.5, label='Joint IA+NC (β=0)')
ax4.set_xlabel('SNR (dB)')
ax4.set_ylabel('Sum-rate (bps/Hz)')
ax4.set_title('Fig 4 — Sum-rate vs SNR, (4×4,2)³ MIMO')
ax4.legend()
ax4.grid(True, ls=':', alpha=0.5)
fig4.tight_layout()
fig4.savefig('fig4_ia_sumrate.png', dpi=200)

# ── Fig 5: Sum-rate vs β at 20 dB ────────────────────────────────────────────
snr_20 = 100.0
sigma2_20 = 1.0 / snr_20
rate_ia_beta = np.zeros(len(BETA_RANGE))
rate_ianc_beta = np.zeros(len(BETA_RANGE))

print("\nSweeping β at 20 dB...")
for bi, beta in enumerate(BETA_RANGE):
    acc_ia, acc_ianc = 0.0, 0.0
    for _ in range(N_REAL):
        H, Hc = generate_channels(K, Mr, Mt, beta=beta)
        V, U = min_leakage_ia(Hc, K, Mr, Mt, d)
        acc_ia += compute_sum_rate(H, V, U, K, d, sigma2_20)
        acc_ianc += compute_ia_nc_rate(H, V, U, K, d, sigma2_20)
    rate_ia_beta[bi] = acc_ia / N_REAL
    rate_ianc_beta[bi] = acc_ianc / N_REAL
    print(f"  β={beta:.2f} done")

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.fill_between(BETA_RANGE, rate_ia_beta, rate_ianc_beta,
                 alpha=0.15, color='green', label='IA+NC advantage')
ax5.plot(BETA_RANGE, rate_ia_beta, 'b-o', ms=5, label='Classical IA')
ax5.plot(BETA_RANGE, rate_ianc_beta, 'r-s', ms=5, label='Joint IA+NC')
ax5.set_xlabel('CSI error parameter β')
ax5.set_ylabel('Sum-rate (bps/Hz) at 20 dB')
ax5.set_title('Fig 5 — Sum-rate vs β')
ax5.legend()
ax5.grid(True, ls=':', alpha=0.5)
fig5.tight_layout()
fig5.savefig('fig5_ia_csi.png', dpi=200)

plt.show()
print("\nDone — saved fig4, fig5 and printed Table II.")
