#!/usr/bin/env python
"""Analyze the per-pair dump written by `probe_velocity_gauge --spectral NK OUT`.

Scores the two velocity conventions on diamond LCAO against the
Thomas-Reiche-Kuhn f-sum rule (exact value: the valence electron count) and
measures the energy-resolved deviation between them. Results and the verdict
are recorded in doc/position_commutator.md section 7.

Usage: python analyze_velocity_convention.py spectral_pairs.txt NK
"""
import sys
import numpy as np

TRK = 3.80998212        # hbar^2 / 2 m_e, eV A^2
N_OCC = 4               # diamond: 8 valence electrons, spin-degenerate

d = np.loadtxt(sys.argv[1])
nk = int(sys.argv[2]) ** 3
d = d[d[:, 0] >= 1]     # drop kidx 0: SetupOptions default Gamma, not the grid
n, m = d[:, 1].astype(int), d[:, 2].astype(int)
w = d[:, 4] - d[:, 3]
va = d[:, 5] + 1j * d[:, 6]
vb = d[:, 7] + 1j * d[:, 8]
pairs = (n < N_OCC) & (m >= N_OCC)

for v, name in [(va, "route (a) explicit"), (vb, "route (b) commutator")]:
    f = np.abs(v[pairs]) ** 2 / (TRK * w[pairs])
    print("n_eff %-22s = %7.4f  (exact: %d)" % (name, 2.0 * f.sum() / nk, 2 * N_OCC))

print()
for lo, hi in [(0, 10), (10, 14), (14, 18), (18, 25), (25, 40), (40, 300)]:
    msk = pairs & (w >= lo) & (w < hi) & (np.abs(vb) > 1e-2)
    if msk.sum() < 5:
        continue
    r = np.abs(va[msk] / vb[msk])
    print("w in [%3d,%3d): median|va/vb| = %.3f  IQR [%.3f, %.3f]  (%d pairs)"
          % (lo, hi, np.median(r), np.percentile(r, 25), np.percentile(r, 75), msk.sum()))
