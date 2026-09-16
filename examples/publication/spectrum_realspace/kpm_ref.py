"""Rebuild a reference DOS/PDOS through the SAME Jackson-KPM kernel the
stochastic run used, and load distributed real-space maps.

Comparing a KPM curve against a Gaussian of matching FWHM is not a fair test:
a Jackson kernel is only approximately Gaussian, and at a van Hove singularity
the shape difference swamps the stochastic error one is trying to measure (it
inflated diamond's DOS deviation from 1.9% to 121% of peak). Everything here
mirrors rsatb/calc/kpm.hpp and observables.hpp exactly -- moment count, edge
inset, and the 2*rho amplitude convention -- so the residual is the stochastic
estimator alone.
"""
import glob
import os

import numpy as np

JACKSON_FWHM = np.sqrt(8 * np.log(2)) * np.pi  # 7.396


def jackson(m):
    """Exactly rsatb/calc/kpm.hpp:jackson_kernel."""
    g = np.empty(m)
    denom = m + 1.0
    th = np.pi / denom
    cot = np.cos(th) / np.sin(th)
    i = np.arange(1, m)
    g[0] = 1.0
    g[1:] = ((denom - i) * np.cos(th * i) + cot * np.sin(th * i)) / denom
    return g


def n_moments_for(energy, de_task):
    """The run's moment count, from its own energy grid."""
    nt = energy.size // 2
    scale = energy[nt] - energy[0]
    return int(np.ceil(JACKSON_FWHM * scale / de_task))


def kpm_curves(eps, rows, energy, n_moments, n_basis):
    """Reference DOS/PDOS on the run's convention (2*rho, moments with mu_0=1).

    `rows` is (n_row, n_state) of per-state weights; a row of ones gives the
    total. Returns (n_row, n_energy).
    """
    nt = energy.size // 2
    shift = energy[nt]
    scale = shift - energy[0]
    y = np.clip((eps - shift) / scale, -1.0, 1.0)

    mu = np.empty((rows.shape[0], n_moments))
    t_prev, t_curr = np.ones_like(y), y.copy()
    mu[:, 0] = rows @ t_prev / n_basis
    if n_moments > 1:
        mu[:, 1] = rows @ t_curr / n_basis
    for m in range(2, n_moments):
        t_prev, t_curr = t_curr, 2.0 * y * t_curr - t_prev
        mu[:, m] = rows @ t_curr / n_basis

    inset = 0.5 / nt
    x = np.clip((energy - shift) / scale, -1.0 + inset, 1.0 - inset)
    g = jackson(n_moments)
    acc = np.outer(mu[:, 0] * g[0], np.ones_like(x))
    t_prev, t_curr = np.ones_like(x), x.copy()
    if n_moments > 1:
        acc += 2.0 * np.outer(mu[:, 1] * g[1], t_curr)
    for m in range(2, n_moments):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
        acc += 2.0 * np.outer(mu[:, m] * g[m], t_curr)
    return 2.0 * acc / (np.pi * np.sqrt(np.maximum(1e-14, 1.0 - x * x)))


def dev(a, b, scale):
    """(max, rms) deviation as a percentage of `scale`."""
    d = a - b
    return 100 * np.abs(d).max() / scale, 100 * np.sqrt((d ** 2).mean()) / scale


def load_ranks(resdir, pattern, key, npt):
    """Reassemble a per-rank distributed map into global linear grid order."""
    files = sorted(glob.glob(os.path.join(resdir, pattern)))
    if not files:
        return None
    parts = [np.load(p) for p in files]
    flat = np.zeros(parts[0][key].shape[:-1] + (npt,))
    for p in parts:
        flat[..., p["point_index"]] = p[key]
    return flat
