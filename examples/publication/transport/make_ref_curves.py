#!/usr/bin/env python3
"""Exact-diagonalization reference curves for the transport figure.

Implements, at ABSOLUTE scale, the trace limits of the three stochastic
estimators exactly as coded (constants and kernels transcribed from the
C++ -- see the spec in this session's audit; source lines cited inline):

  dc_spectrum  sigma_aa(E) = (2 pi/Omega) SUM_nm |V_nm|^2 dJ(E-e_n) dJ(E-e_m)
               x 2.4341348e6 -> S/m (pi of Kubo-Greenwood times spin 2;
               before 2026-09-08 the code omitted both); Jackson N =
               n_moments over the run's own padded window (recovered
               EXACTLY from the stored Chebyshev-node energy grid).
  ac           C(t) = (1/N) SUM_nm f_n (1-f_m) |V_nm|^2 e^{i w_nm t},
               f = the truncated Jackson-Chebyshev Fermi operator (not
               the analytic function), then the code's literal windowed
               sine transform, thermal factor, and driver optics chain
               (Hilbert -> epsilon -> alpha).
  msd          MSD(E,t) = [SUM_nm dJ(E-e_n)|V_nm|^2 g(de_nm,t)]
                          / SUM_n dJ(E-e_n),
               g(de,t) = 4 sin^2(de t/2hbar)/de^2 (limit t^2/hbar^2),
               delta on the bra index only; D = 0.5 d/dt MSD.

The kernel windows come from the production outputs themselves, so the
reference sits on each run's exact spectral rescaling.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, "results", *p)

SIG_SI = 2.4341348e6      # e^2/(hbar Angstrom) in S/m   (driver.py:377)
HBAR_FS = 0.6582119569    # eV fs                        (msd.hpp:269)
HBAR_S = 6.582119569e-16  # eV s                         (driver.py:370)
KB = 8.617333262e-5       # eV/K
EPS0 = 8.8541878128e-12
C_MS = 2.99792458e8
EV_CM1 = 8065.56


def jackson(N):
    D = N + 1.0
    th = np.pi / D
    cot = np.cos(th) / np.sin(th)
    n = np.arange(1, N)
    g = np.empty(N)
    g[0] = 1.0
    g[1:] = ((D - n) * np.cos(th * n) + cot * np.sin(th * n)) / D
    return g


def window_from_cheb_grid(E):
    """Recover (center, half_width) exactly from an ascending Chebyshev
    node grid E_k = c + a cos(pi (N-1-k+0.5)/N)."""
    N = E.size
    t0 = np.cos(np.pi * (N - 0.5) / N)   # k = 0
    t1 = np.cos(np.pi * 0.5 / N)         # k = N-1
    a = (E[-1] - E[0]) / (t1 - t0)
    c = E[0] - a * t0
    return c, a


def delta_matrix(E, eps, c, a, N_mom):
    """dJ[k, n] = Jackson-broadened delta_J^E(E_k - eps_n) in 1/eV,
    evaluated exactly as kpm.hpp does (series over T_n, /(pi sqrt(1-x^2)),
    then /a for energy units)."""
    g = jackson(N_mom)
    x = np.clip((E - c) / a, -1 + 1e-12, 1 - 1e-12)
    y = np.clip((eps - c) / a, -1.0, 1.0)
    tx = np.arccos(x)[:, None]           # (nE, 1)
    ty = np.arccos(y)[None, :]           # (1, N)
    out = np.full((E.size, eps.size), g[0])
    # sum over moments in chunks to bound memory
    for m0 in range(1, N_mom, 256):
        m = np.arange(m0, min(m0 + 256, N_mom))
        out += 2.0 * np.einsum("m,km,mn->kn",
                               g[m],
                               np.cos(tx * m[None, :]),
                               np.cos(ty.T * m[None, :]).T,
                               optimize=True)
    out /= np.pi * np.sqrt(np.maximum(1e-14, 1.0 - x[:, None] ** 2))
    return out / a


def fermi_kpm(eps, c, a, mu, temp, n_kpm):
    """f evaluated at eigenvalues through the truncated Jackson-Chebyshev
    Fermi operator (fermi_dirac.hpp): a_k by dense Chebyshev projection,
    Jackson-damped, degree n_kpm."""
    xs = np.clip((eps - c) / a, -1.0, 1.0)
    if temp <= 0:
        th_mu = np.arccos(np.clip((mu - c) / a, -1, 1))
        k = np.arange(1, n_kpm)
        ak = np.concatenate([[1 - th_mu / np.pi], -2 * np.sin(k * th_mu) / (np.pi * k)])
    else:
        beta_s = a / (KB * temp)
        mu_s = (mu - c) / a
        M = 1 << 16
        th = np.pi * (np.arange(M) + 0.5) / M
        fv = 1.0 / (1.0 + np.exp(np.clip(beta_s * (np.cos(th) - mu_s), -700, 700)))
        k = np.arange(n_kpm)
        ak = 2.0 / M * (np.cos(k[:, None] * th[None, :]) @ fv)
        ak[0] *= 0.5
    g = jackson(n_kpm)
    tt = np.arccos(xs)
    f = np.zeros_like(xs)
    for k0 in range(0, n_kpm, 512):
        k = np.arange(k0, min(k0 + 512, n_kpm))
        f += (g[k] * ak[k]) @ np.cos(np.outer(k, tt))
    return f


def ref_dc_spectrum(name):
    z = np.load(R(name, "dc_spectrum.npz"))
    E = z["energy"]
    N_mom = E.size
    c, a = window_from_cheb_grid(E)
    d = np.load(R(f"ref_{name}_eig.npz"))
    eps, W, cell = d["eps"], d["Wabc"], d["cell"]
    omega = abs(np.linalg.det(cell))
    dJ = delta_matrix(E, eps, c, a, N_mom)          # (nE, N)
    M = W @ dJ.T                                    # (N, nE)
    # Spin-degenerate Kubo-Greenwood, as the code reports since 2026-09-08.
    sig = 2.0 * np.pi * np.einsum("kn,nk->k", dJ, M) / omega * SIG_SI
    return E, sig


def ref_ac(name, mu, temp=300.0):
    z = np.load(R(name, "ac.npz"))
    w_ev = z["omega_cm1"] / EV_CM1
    nt = w_ev.size - 1
    a = w_ev[1] * (nt + 1)                          # omega_i = i a/(nt+1)
    # center from the dc_spectrum grid of the same system (same setup)
    zs = np.load(R(name, "dc_spectrum.npz"))
    c, a_dc = window_from_cheb_grid(zs["energy"])
    d = np.load(R(f"ref_{name}_eig.npz"))
    eps, W, cell = d["eps"], d["Wabc"], d["cell"]
    N = d["norb"]
    omega_cell = abs(np.linalg.det(cell))
    import json as _json
    n_kpm = 1000
    inp = os.path.join(HERE, name + ".json")
    if os.path.exists(inp):
        for t in _json.load(open(inp))["tasks"]:
            if t["type"] == "ac":
                n_kpm = int(t.get("n_kpm", 1000))
    f = fermi_kpm(eps, c, a, mu, temp, n_kpm)
    # pair sums: J(t) = (1/N) sum_nm f_n (1-f_m) W_nm e^{i (e_n - e_m) t / hbar}
    ns = nt * 2
    t_step = np.pi / (a * 2.0)                      # 1/eV
    t = np.arange(ns + 1) * t_step
    P = (f[:, None] * (1.0 - f[None, :])) * W       # (N, N)
    # bin pair frequencies for tractability (bin << window sigma)
    de = eps[:, None] - eps[None, :]
    lo, hi = de.min(), de.max()
    nbin = 60000
    idx = np.clip(((de - lo) / (hi - lo) * (nbin - 1)).astype(np.int64), 0, nbin - 1)
    Jw = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nbin)
    wbin = lo + (np.arange(nbin) + 0.5) * (hi - lo) / nbin
    keep = Jw != 0
    Jw, wbin = Jw[keep], wbin[keep]
    phase = np.exp(1j * np.outer(t, wbin))          # (ns+1, nb)
    cac = (phase @ Jw) / N                          # complex correlator
    # literal transform (observables.hpp:626-640)
    wgt = np.exp(-2.0 * ((np.arange(ns + 1) + 1.0) / (ns + 1.0)) ** 2)
    sig = np.zeros(nt + 1)
    beta = 1.0 / (KB * temp)
    for i in range(1, nt + 1):
        s = np.sum(2.0 * np.sin(w_ev[i] * np.arange(ns + 1) * t_step) * wgt * cac.imag)
        thermal = np.exp(-beta * w_ev[i]) - 1.0 if temp > 0 else -1.0
        sig[i] = (N / omega_cell) * t_step * s * thermal / w_ev[i]
    sig_si = 2.0 * sig * SIG_SI   # spin degeneracy, as the code reports since 2026-09-08
    # driver optics chain (driver.py:354-406)
    from scipy.signal import hilbert
    full = np.zeros(2 * (nt + 1))
    full[nt + 1:] = sig_si
    full[1:nt + 2] = sig_si[::-1]
    sig_c = sig_si + 1j * np.imag(hilbert(full))[nt + 1:2 * (nt + 1)]
    epsn = np.ones(nt + 1, dtype=complex)
    m = w_ev > 1e-10
    epsn[m] = 1 + 1j * sig_c[m] * HBAR_S / (EPS0 * w_ev[m])
    Nc = np.sqrt(epsn)
    alpha = 2 * (w_ev / HBAR_S) * Nc.imag / C_MS / 100.0
    return w_ev, sig_si, alpha, sig_c.imag


def ref_msd(name):
    z = np.load(R(name, "msd.npz"))
    E, t = z["energy"], z["time_fs"]
    n_kpm = E.size
    c, a = window_from_cheb_grid(E)
    d = np.load(R(f"ref_{name}_eig.npz"))
    eps, W = d["eps"], d["Wx"]
    de = eps[:, None] - eps[None, :]
    small = np.abs(de) < 1e-9
    dJ = delta_matrix(E, eps, c, a, n_kpm)          # (nE, N)
    dens = dJ.sum(axis=1)                           # trace-limit denominator
    msd = np.zeros((t.size, E.size))
    for it, tv in enumerate(t):
        if tv == 0:
            continue
        gk = np.where(small, (tv / HBAR_FS) ** 2,
                      4.0 * np.sin(de * tv / (2 * HBAR_FS)) ** 2 /
                      np.where(small, 1.0, de) ** 2)
        num_n = (W * gk).sum(axis=1)                # (N,)
        msd[it] = (dJ @ num_n) / dens
    diff = 0.5 * np.gradient(msd, t, axis=0, edge_order=1)
    return E, t, msd, diff


SYS = [("diamond", 15.418), ("silicon", 6.528), ("aluminum", 11.112081)]

if __name__ == "__main__":
    todo = sys.argv[1:] or [s for s, _ in SYS]
    for name, mu in SYS:
        if name not in todo:
            continue
        print(f"[{name}] dc_spectrum ...", flush=True)
        E, sig = ref_dc_spectrum(name)
        print(f"[{name}] ac ...", flush=True)
        w, sac, alpha, sac_im = ref_ac(name, mu)
        print(f"[{name}] msd ...", flush=True)
        Em, t, msd, diff = ref_msd(name)
        np.savez(R(f"ref_{name}_curves.npz"),
                 energy=E, sigma_e=sig,
                 omega_ev=w, sigma_w=sac, sigma_w_imag=sac_im, alpha=alpha,
                 msd_energy=Em, time_fs=t, msd=msd, diffusion=diff,
                 prefactor_fixed=np.array(1))  # already in the fixed units: rescale_prefactor_fix.py skips it
        print(f"[{name}] wrote ref_{name}_curves.npz", flush=True)
