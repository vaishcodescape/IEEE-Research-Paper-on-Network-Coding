"""
Simulation 6 — Outage Probability P_out vs Average SNR γ̄
Produces: Fig 9

Setup (from paper Sec VIII-C, Fig 9 caption):
  - k = 8, n = 12, R = 1 b/cu
  - Analytical: P_out = F_Bin(n, 1−ε)(k−1)
  - Routing:    P_out = 1 − (1−ε)^k
  - Monte Carlo: 2×10^4 trials per point
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.stats import binom
import matplotlib.pyplot as plt

k = 8
n = 12
R = 1.0
N_MC = 20_000
GAMMA_DB = np.linspace(0, 25, 200)
GAMMA_LIN = 10 ** (GAMMA_DB / 10)
MC_SNR_DB = np.arange(2, 26, 2)


def erasure(gamma_bar, R):
    return 1.0 - np.exp(-(2**R - 1) / gamma_bar)


# ── Analytical curves ─────────────────────────────────────────────────────────
eps_arr = erasure(GAMMA_LIN, R)
pout_nc = binom.cdf(k - 1, n, 1 - eps_arr)
pout_rt = 1 - (1 - eps_arr) ** k

# ── Monte Carlo ───────────────────────────────────────────────────────────────
mc_pout_nc = []
mc_pout_rt = []
for sdb in MC_SNR_DB:
    g = 10 ** (sdb / 10)
    eps = erasure(g, R)
    received = np.random.binomial(n, 1 - eps, size=N_MC)
    mc_pout_nc.append(np.mean(received < k))

    survived = np.random.binomial(k, 1 - eps, size=N_MC)
    mc_pout_rt.append(np.mean(survived < k))

# ── Fig 9 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(GAMMA_DB, pout_nc, 'b-', lw=1.5, label='RLNC ($n=12$, analytical)')
ax.semilogy(GAMMA_DB, pout_rt, 'r-', lw=1.5, label='Routing (analytical)')
ax.semilogy(MC_SNR_DB, mc_pout_nc, 'bs', ms=6, mfc='none', label='RLNC (Monte Carlo)')
ax.semilogy(MC_SNR_DB, mc_pout_rt, 'rs', ms=6, mfc='none', label='Routing (Monte Carlo)')
ax.set_xlabel('Average SNR $\\bar{\\gamma}$ (dB)')
ax.set_ylabel('Outage probability $P_{out}$')
ax.set_title('Fig 9 — Outage Probability vs SNR (k=8, n=12, R=1 b/cu)')
ax.legend()
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.set_ylim(1e-6, 1.5)
fig.tight_layout()
fig.savefig('fig9_outage_prob.png', dpi=200)
plt.show()
print("Done — saved fig9_outage_prob.png")
