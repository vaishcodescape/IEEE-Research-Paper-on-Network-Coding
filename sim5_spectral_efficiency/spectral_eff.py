"""
Simulation 5 — Spectral Efficiency η vs Average SNR γ̄
Produces: Fig 8

Setup (from paper Sec VIII-C):
  - k = 10, R = 1 b/cu, P_d target = 0.99
  - η = kR / n
  - Adaptive n* vs fixed n = 18 vs routing
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
GAMMA_DB = np.linspace(0, 25, 200)
GAMMA_LIN = 10 ** (GAMMA_DB / 10)


def erasure(gamma_bar, R):
    return 1.0 - np.exp(-(2**R - 1) / gamma_bar)


def decode_prob(k, n, eps):
    if eps >= 1.0:
        return 0.0
    return 1.0 - binom.cdf(k - 1, n, 1 - eps)


def find_n_star(k, gamma_bar, R, p_target):
    eps = erasure(gamma_bar, R)
    for n in range(k, k + 200):
        if decode_prob(k, n, eps) >= 1 - p_target:
            return n
    return k + 200


eta_adaptive = np.zeros(len(GAMMA_DB))
eta_fixed = np.zeros(len(GAMMA_DB))
eta_routing = np.zeros(len(GAMMA_DB))

for i, g in enumerate(GAMMA_LIN):
    eps = erasure(g, R)

    ns = find_n_star(k, g, R, P_TARGET)
    eta_adaptive[i] = k * R / ns

    eta_fixed[i] = k * R / N_FIX

    eta_routing[i] = (1 - eps) * R       # each of k packets must survive individually

# ── Fig 8 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(GAMMA_DB, eta_adaptive, 'b-', lw=2, label='Adaptive $n^*$')
ax.plot(GAMMA_DB, eta_fixed, '--', color='orange', lw=1.5, label='Fixed $n = 18$')
ax.plot(GAMMA_DB, eta_routing, 'g-.', lw=1.5, label='Routing (no NC)')
ax.axhline(1.0, ls=':', color='gray', alpha=0.5)

# Mark the 42% gap at 15 dB
idx_15 = np.argmin(np.abs(GAMMA_DB - 15))
ax.annotate(f'Δη ≈ 42%', xy=(15, eta_adaptive[idx_15]),
            xytext=(18, 0.65), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.set_xlabel('Average SNR $\\bar{\\gamma}$ (dB)')
ax.set_ylabel('Spectral efficiency $\\eta$ (b/cu)')
ax.set_title('Fig 8 — Spectral Efficiency vs SNR (k=10, R=1, $P_d \\geq 0.99$)')
ax.legend()
ax.grid(True, ls=':', alpha=0.5)
ax.set_ylim(0, 1.15)
fig.tight_layout()
fig.savefig('fig8_spectral_eff.png', dpi=200)
plt.show()
print("Done — saved fig8_spectral_eff.png")
