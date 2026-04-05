"""
Simulation 7 — Joint Power Optimisation + Multicast Throughput
Produces: Fig 10 (Water-filling + throughput vs N), Table IV

Setup (from paper Sec VII, VIII-D):
  - Single source, N receivers over independent Rayleigh links
  - Water-filling power allocation: P*_ij = (μ − N0/h_ij)^+
  - Compare: Routing, NC only (uniform power), NC + Water-filling
  - Average SNR ≈ 15 dB, total power budget P_total = 10
  - N ∈ {3, 5, 10, 15, 20, 30, 50} receivers
  - 1000 Rayleigh realisations per N
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_TOTAL = 10.0
N0 = 1.0
OMEGA = 1.0
N_REAL = 1000
N_RECEIVERS = [3, 5, 10, 15, 20, 30, 50]


def generate_channel_gains(N):
    """Rayleigh: |h|^2 ~ Exp(1/Ω)."""
    return np.random.exponential(scale=OMEGA, size=N)


def uniform_power(N):
    return np.full(N, P_TOTAL / N)


def waterfill(h_gains, P_total, N0):
    """Classic water-filling: P_i = (μ − N0/h_i)^+, Σ P_i = P_total."""
    N = len(h_gains)
    inv_gains = N0 / h_gains
    sorted_idx = np.argsort(inv_gains)
    inv_sorted = inv_gains[sorted_idx]

    P_alloc = np.zeros(N)
    active = N
    for attempt in range(N):
        mu = (P_total + np.sum(inv_sorted[:active])) / active
        powers = mu - inv_sorted[:active]
        if np.all(powers >= 0):
            P_alloc[sorted_idx[:active]] = powers
            break
        active -= 1
        if active == 0:
            P_alloc[sorted_idx[0]] = P_total
            break
    return P_alloc


def link_capacity(P, h, N0):
    return np.log2(1 + P * h / N0)


def multicast_rate_routing(h_gains, P_alloc, N0):
    """Routing: min-rate across all receivers (bottleneck link)."""
    caps = link_capacity(P_alloc, h_gains, N0)
    return np.min(caps)


def multicast_rate_nc(h_gains, P_alloc, N0):
    """NC: all links contribute to a shared coded stream → average capacity."""
    caps = link_capacity(P_alloc, h_gains, N0)
    return np.mean(caps)


# ── Sweep over N ──────────────────────────────────────────────────────────────
thr_routing = np.zeros(len(N_RECEIVERS))
thr_nc_only = np.zeros(len(N_RECEIVERS))
thr_nc_wf = np.zeros(len(N_RECEIVERS))

for ni, N in enumerate(N_RECEIVERS):
    acc_rt, acc_nc, acc_wf = 0.0, 0.0, 0.0
    for _ in range(N_REAL):
        h = generate_channel_gains(N)

        P_uni = uniform_power(N)
        acc_rt += multicast_rate_routing(h, P_uni, N0)
        acc_nc += multicast_rate_nc(h, P_uni, N0)

        P_wf = waterfill(h, P_TOTAL, N0)
        acc_wf += multicast_rate_nc(h, P_wf, N0)

    thr_routing[ni] = acc_rt / N_REAL
    thr_nc_only[ni] = acc_nc / N_REAL
    thr_nc_wf[ni] = acc_wf / N_REAL

# Normalise to routing at N=3 for the paper's "normalised throughput" metric
norm = thr_routing[0] if thr_routing[0] > 0 else 1.0
thr_routing_n = thr_routing / norm
thr_nc_only_n = thr_nc_only / norm
thr_nc_wf_n = thr_nc_wf / norm

# ── Table IV ──────────────────────────────────────────────────────────────────
print("=== Table IV: Multicast Performance Under Joint Optimisation ===")
print(f"{'Metric':<28s} {'Routing':>10s} {'NC Only':>10s} {'NC+Power':>10s}")
print("-" * 60)
for tag, idx_N in [('Throughput (N=3)', 0), ('Throughput (N=20)', 4), ('Throughput (N=50)', 6)]:
    print(f"{tag:<28s} {thr_routing_n[idx_N]:10.1f} {thr_nc_only_n[idx_N]:10.1f} {thr_nc_wf_n[idx_N]:10.1f}")
ee_rt, ee_nc, ee_wf = 1.0, thr_nc_only_n[0], thr_nc_wf_n[0]
print(f"{'ηEE gain vs routing':<28s} {ee_rt:9.1f}× {ee_nc:9.1f}× {ee_wf:9.1f}×")

# ── Fig 10 left: Water-filling example for one realisation ────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

h_example = generate_channel_gains(10)
P_uni_ex = uniform_power(10)
P_wf_ex = waterfill(h_example, P_TOTAL, N0)

x = np.arange(10)
width = 0.35
ax_left.bar(x - width / 2, P_uni_ex, width, label='Uniform', color='steelblue')
ax_left.bar(x + width / 2, P_wf_ex, width, label='Water-filling', color='coral')
ax_left.set_xlabel('Link index')
ax_left.set_ylabel('Allocated power')
ax_left.set_title('Power Allocation per Link (N=10)')
ax_left.legend()
ax_left.grid(True, axis='y', ls=':', alpha=0.5)

# ── Fig 10 right: Normalised throughput vs N ──────────────────────────────────
ax_right.plot(N_RECEIVERS, thr_routing_n, 'g-^', ms=6, lw=1.5, label='Routing')
ax_right.plot(N_RECEIVERS, thr_nc_only_n, 'b-o', ms=6, lw=1.5, label='NC only')
ax_right.plot(N_RECEIVERS, thr_nc_wf_n, 'r-s', ms=6, lw=1.5, label='NC + Power opt.')
ax_right.set_xlabel('Number of receivers $N$')
ax_right.set_ylabel('Normalised throughput')
ax_right.set_title('Multicast Throughput vs Receiver Count')
ax_right.legend()
ax_right.grid(True, ls=':', alpha=0.5)

fig.suptitle('Fig 10 — Water-filling Power Allocation and Multicast Throughput', y=1.01)
fig.tight_layout()
fig.savefig('fig10_waterfilling.png', dpi=200)
plt.show()
print("\nDone — saved fig10_waterfilling.png and printed Table IV.")
