"""
Simulation 4 — Decode Probability P_d vs Average SNR γ̄
Produces: Fig 7

Setup (from paper Sec VIII-C):
  - k = 10 source packets, GF(2^8)
  - R = 1 b/cu, target P_d = 0.99
  - Fixed baseline n = 18 (designed for ε_max = 0.30)
  - Adaptive n* from Eq (25)
  - 2×10^4 Rayleigh realisations per SNR point
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.stats import binom
import matplotlib.pyplot as plt

k = 10
R = 1.0
P_TARGET = 0.01
N_FIX = 18
N_MC = 20_000
GAMMA_DB = np.linspace(0, 25, 50)
GAMMA_LIN = 10 ** (GAMMA_DB / 10)


def erasure(gamma_bar, R):
    return 1.0 - np.exp(-(2**R - 1) / gamma_bar)


def decode_prob_analytical(k, n, eps):
    """P_d = 1 − F_Bin(n, 1−ε)(k−1)"""
    if eps >= 1.0:
        return 0.0
    return 1.0 - binom.cdf(k - 1, n, 1 - eps)


def find_n_star(k, gamma_bar, R, p_target):
    eps = erasure(gamma_bar, R)
    for n in range(k, k + 100):
        if decode_prob_analytical(k, n, eps) >= 1 - p_target:
            return n
    return k + 100


def monte_carlo_decode_prob(k, n, gamma_bar, R, n_mc):
    eps = erasure(gamma_bar, R)
    received = np.random.binomial(n, 1 - eps, size=n_mc)
    return np.mean(received >= k)


# ── Compute curves ────────────────────────────────────────────────────────────
pd_fixed = np.zeros(len(GAMMA_DB))
pd_adaptive = np.zeros(len(GAMMA_DB))
pd_mc = np.zeros(len(GAMMA_DB))
n_star_arr = np.zeros(len(GAMMA_DB), dtype=int)

for i, g in enumerate(GAMMA_LIN):
    eps = erasure(g, R)
    pd_fixed[i] = decode_prob_analytical(k, N_FIX, eps)

    ns = find_n_star(k, g, R, P_TARGET)
    n_star_arr[i] = ns
    pd_adaptive[i] = decode_prob_analytical(k, ns, eps)

    pd_mc[i] = monte_carlo_decode_prob(k, ns, g, R, N_MC)

# ── Fig 7 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(GAMMA_DB, pd_adaptive, 'b-', lw=1.5, label='Adaptive $n^*$ (analytical)')
ax.plot(GAMMA_DB, pd_fixed, '--', color='orange', lw=1.5, label='Fixed $n = 18$ (analytical)')
ax.plot(GAMMA_DB, pd_mc, 'ro', ms=4, alpha=0.6, label=f'Monte Carlo ($2\\times10^4$ trials)')
ax.axhline(0.99, ls=':', color='gray', alpha=0.6, label='Target $P_d = 0.99$')
ax.set_xlabel('Average SNR $\\bar{\\gamma}$ (dB)')
ax.set_ylabel('Decode probability $P_d$')
ax.set_title('Fig 7 — Decode Probability vs SNR (k=10, R=1 b/cu)')
ax.legend(loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig('fig7_decode_prob.png', dpi=200)
plt.show()
print("Done — saved fig7_decode_prob.png")
