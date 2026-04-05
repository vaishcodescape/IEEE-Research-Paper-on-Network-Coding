"""
Simulation 3 — Erasure Probability ε vs Average SNR γ̄
Produces: Fig 6

From paper Eq (7):  ε(γ̄, R) = 1 − exp(−(2^R − 1)/γ̄)
Rates: R ∈ {0.5, 1.0, 1.5, 2.0, 2.5} b/cu
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA_DB = np.linspace(0, 30, 300)
GAMMA_LIN = 10 ** (GAMMA_DB / 10)
RATES = [0.5, 1.0, 1.5, 2.0, 2.5]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
N_MC = 50_000


def erasure_analytical(gamma_bar, R):
    return 1.0 - np.exp(-(2**R - 1) / gamma_bar)


def erasure_monte_carlo(gamma_bar, R, n_mc):
    """Draw Rayleigh SNR samples, check threshold."""
    gamma_samples = np.random.exponential(scale=gamma_bar, size=n_mc)
    threshold = 2**R - 1
    return np.mean(gamma_samples < threshold)


fig, ax = plt.subplots(figsize=(7, 5))
mc_snr_pts = np.arange(0, 31, 5)

for R, c in zip(RATES, COLORS):
    eps = erasure_analytical(GAMMA_LIN, R)
    ax.plot(GAMMA_DB, eps, color=c, lw=1.5, label=f'R = {R} b/cu')

    mc_eps = []
    for s in mc_snr_pts:
        g = 10 ** (s / 10)
        mc_eps.append(erasure_monte_carlo(g, R, N_MC))
    ax.plot(mc_snr_pts, mc_eps, 'o', color=c, ms=5, alpha=0.7)

ax.set_xlabel('Average SNR $\\bar{\\gamma}$ (dB)')
ax.set_ylabel('Erasure probability $\\varepsilon$')
ax.set_title('Fig 6 — Erasure Probability vs SNR (Rayleigh fading, q = 256)')
ax.legend()
ax.grid(True, ls=':', alpha=0.5)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig('fig6_erasure_prob.png', dpi=200)
plt.show()
print("Done — saved fig6_erasure_prob.png")
