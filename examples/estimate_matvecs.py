#!/usr/bin/env python3
"""Estimate the matrix-vector product count of a JSON input, before running it.

The calculation itself is a JSON file and `rsatb run`. This is the one piece
that has no place in an input file: a cost model. Matrix-vector products are
what an RSATB run spends its time on, and their number follows from the
Chebyshev expansions the requested tasks use -- so it can be worked out from
the input alone, on a laptop, before anything is queued.

    python examples/estimate_matvecs.py examples/06_matblg_transport.json
    python examples/estimate_matvecs.py input.json --bounds -11 11

The spectral window is needed because two of the expansions are truncated
against it. An input with `run.bounds.mode = "fixed"` already states it;
otherwise pass `--bounds EMIN EMAX`, or point `--summary` at the summary.json
of an earlier run and the window it recorded is used.

Nothing here imports the compiled extension.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.special as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsatb.inputs import load_and_resolve

HBAR_EV_FS = 0.6582119569
KB_EV_PER_K = 8.617333262e-5
PRECISION = 1.0e-14


# --- the expansions, and where each one stops -----------------------------


def chebyshev_coeff_count(time_step: float, precision: float = PRECISION) -> int:
    """Retained coefficients of a Chebyshev time propagation.

    Matches rsatb::calc::build_bessel_vector: a propagation with N retained
    coefficients performs N - 1 recurrence applications, and the series stops
    once both J_n(t) and J_{n+1}(t) are below `precision` (rsatb/calc/bessel.hpp).
    """
    count = 0
    for n in range(65536):
        if abs(sc.jv(n, time_step)) > precision:
            count += 1
            continue
        if abs(sc.jv(n + 1, time_step)) > precision:
            count += 1
            continue
        return count
    raise RuntimeError("Bessel series not converged for the matvec estimate")


def jackson_kernel(n: int) -> np.ndarray:
    denom = float(n + 1)
    theta = np.pi / denom
    out = np.empty(n, dtype=np.float64)
    out[0] = 1.0
    if n > 1:
        m = np.arange(1, n, dtype=np.float64)
        cot_theta = np.cos(theta) / np.sin(theta)
        out[1:] = ((denom - m) * np.cos(theta * m) + cot_theta * np.sin(theta * m)) / denom
    return out


def occupation_projector(x: np.ndarray, beta: float, mu: float) -> np.ndarray:
    if not np.isfinite(beta) or beta <= 0.0:
        return (x <= mu).astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    above = x >= mu
    tmp = np.exp(beta * (mu - x[above]))
    out[above] = tmp / (1.0 + tmp)
    tmp = np.exp(beta * (x[~above] - mu))
    out[~above] = 1.0 / (1.0 + tmp)
    return out


def fermi_kpm_coeff_count(n_kpm, center, half_width, temp, mu,
                          precision: float = PRECISION) -> int:
    """Retained coefficients of the Fermi operator, as rsatb::calc::fermi_dirac_kpm."""
    beta = (1.0 / (temp * KB_EV_PER_K)) if temp > 0.0 else 0.0
    beta_re = beta * half_width
    mu_re = (mu - center) / half_width

    dtheta = np.pi / float(n_kpm)
    x = np.cos(dtheta * np.arange(2 * n_kpm, dtype=np.float64))
    values = occupation_projector(x, beta_re, mu_re).astype(np.complex128)
    coeffs = np.fft.fft(values)[:n_kpm]
    coeffs = coeffs.real * jackson_kernel(n_kpm) / float(n_kpm)
    coeffs[0] *= 0.5

    for i in range(n_kpm - 1):
        if abs(coeffs[i]) <= precision and abs(coeffs[i + 1]) <= precision:
            return max(1, i + 1)
    return n_kpm


# --- one cost model per task type ------------------------------------------


def dos_cost(task, window):
    time_coeffs = chebyshev_coeff_count(np.pi)
    total = task["nt"] * max(time_coeffs - 1, 0)
    return total, "{} time steps x {} Chebyshev terms".format(task["nt"], time_coeffs - 1)


def msd_cost(task, window):
    """Gamma-affine MSD: two H0 products and one H1 first-moment product per step."""
    _, half_width = window
    time_coeffs = chebyshev_coeff_count(half_width * task["dt_fs"] / HBAR_EV_FS)
    density = (task["n_steps"] + 2) * max(task["n_kpm"] - 1, 0)
    affine = task["n_steps"] * max(time_coeffs - 1, 0) * 3
    return density + affine, "KPM density {} + affine time {}".format(density, affine)


def quantum_metric_cost(task, window):
    """Runs through the AC conductivity: four Fermi/time propagations plus velocity."""
    center, half_width = window
    time_coeffs = chebyshev_coeff_count(np.pi)
    fermi_coeffs = fermi_kpm_coeff_count(
        task["n_kpm"], center, half_width, task["temp"], task["mu"])
    h0 = 4 * max(fermi_coeffs - 1, 0) + 4 * task["nt"] * max(time_coeffs - 1, 0)
    velocity = 3 * (task["nt"] + 2)
    return h0 + velocity, "H0 {} + velocity {}".format(h0, velocity)


ac_cost = quantum_metric_cost  # the same expansion, without the final integral


def free_cost(task, window):
    return 0, "reported from setup; no expansion of its own"


COSTS = {
    "dos": dos_cost,
    "msd": msd_cost,
    "quantum_metric": quantum_metric_cost,
    "ac": ac_cost,
    "energy_bounds": free_cost,
}


# --- driver ----------------------------------------------------------------


def spectral_window(resolved, override, summary_path):
    """Returns (centre, half-width) in eV, and where it came from."""
    if override:
        emin, emax = override
        source = "--bounds"
    elif summary_path:
        with open(summary_path) as handle:
            emin, emax = json.load(handle)["setup"]["energy_bounds"]
        source = os.path.basename(summary_path)
    else:
        bounds = resolved.get("run", {}).get("bounds", {})
        if bounds.get("mode") != "fixed":
            raise SystemExit(
                "the spectral window is estimated at setup for this input, so it is not\n"
                "known ahead of time. Pass --bounds EMIN EMAX, or --summary from an\n"
                "earlier run, or set run.bounds.mode to 'fixed' in the input.")
        emin, emax = bounds["energy"]
        source = "run.bounds.energy"
    if not emin < emax:
        raise SystemExit("the spectral window needs emin < emax")
    return (0.5 * (emin + emax), 0.5 * (emax - emin)), source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the matrix-vector product count of an RSATB JSON input.")
    parser.add_argument("input", help="the JSON input file")
    parser.add_argument("--bounds", type=float, nargs=2, metavar=("EMIN", "EMAX"),
                        help="spectral window in eV, when the input does not fix one")
    parser.add_argument("--summary", help="summary.json of an earlier run to take the "
                                          "window from")
    args = parser.parse_args()

    resolved, issues = load_and_resolve(args.input, check_paths=False)
    errors = [issue for issue in issues if issue.is_error]
    if errors:
        for issue in errors:
            print(str(issue), file=sys.stderr)
        return 1

    window, source = spectral_window(resolved, args.bounds, args.summary)
    print("{}".format(args.input))
    print("spectral window   [{:+.3f}, {:+.3f}] eV   (from {})".format(
        window[0] - window[1], window[0] + window[1], source))
    print("")

    total = 0
    unknown = []
    for task in resolved.get("tasks", []):
        label = task.get("label") or task["type"]
        if task["type"] not in COSTS:
            unknown.append(label)
            continue
        count, detail = COSTS[task["type"]](task, window)
        total += count
        print("  {:<18} {:>12,d} matvecs   {}".format(label, count, detail))

    print("")
    print("  {:<18} {:>12,d}".format("total", total))
    if unknown:
        print("")
        print("  not counted: {}".format(", ".join(unknown)))
        print("  Only the expansions whose coefficient truncation was checked against the\n"
              "  C++ are modelled here. A cost model that has not been verified against\n"
              "  the code it predicts is worse than no number at all.")
    print("")
    print("Per random sample and per k-point: multiply by run.nsamples ({}) and by the\n"
          "k-points this rank owns. One matvec touches the whole distributed H."
          .format(resolved.get("run", {}).get("nsamples", 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
