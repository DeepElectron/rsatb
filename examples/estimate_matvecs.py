#!/usr/bin/env python3
"""Estimate the matrix-vector product count of a JSON input, before running it.

The calculation itself is a JSON file and `rsatb run`. This is the one piece
that has no place in an input file: a cost model. Matrix-vector products are
what an RSATB run spends its time on, and their number follows from the
Chebyshev expansions the requested tasks use -- so it can be worked out from
the input alone, on a laptop, before anything is queued.

    python examples/estimate_matvecs.py examples/06_matblg_transport.json
    python examples/estimate_matvecs.py input.json --bounds -11 11

The spectral window is needed because several of the expansions are sized
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


def operator_is_hermitian(resolved) -> bool:
    """Mirrors the ChebyOpForm selection (rsatb/calc/cheby.hpp, model/setup.hpp).

    The orthogonal (no overlap) and sandwich forms are Hermitian; only the
    solve form S^-1 H is not. Whether an overlap exists is a model-family
    fact (inputs/spec.py MODEL): sktb is orthogonal unless 'overlap' is
    given, tb_file always is, and lcao/dftb never are.
    """
    model = resolved["model"]
    if model["type"] == "sktb":
        has_overlap = "overlap" in model
    elif model["type"] == "tb_file":
        has_overlap = False
    else:  # lcao, dftb
        has_overlap = True
    if not has_overlap:
        return True
    return resolved["run"]["preconditioner"]["operator_form"] == "sandwich"


def moment_recurrence_matvecs(n_moments: int, hermitian: bool) -> int:
    """Matvecs of one kpm_dos_moments / kpm_pdos_moments recurrence.

    A Hermitian operator gives two moments per matvec through the Chebyshev
    product identity (rsatb/calc/dos.hpp), so N moments cost floor(N/2)
    applications; the solve form falls back to the plain recurrence at N - 1.
    """
    if n_moments <= 1:
        return 0
    return n_moments // 2 if hermitian else n_moments - 1


# --- one cost model per task type ------------------------------------------

# The Jackson kernel at N moments is a Gaussian of FWHM
# sqrt(8 ln 2) * pi * half_width / N (Weisse et al., RMP 78, 275).
JACKSON_FWHM = np.sqrt(8.0 * np.log(2.0)) * np.pi

# rsatb::calc::kAcOversample: the AC correlator carries transitions up to the
# full spectral span, so its Nyquist step is half the DOS step. A constant in
# the code, not a knob.
AC_OVERSAMPLE = 2


def dos_moment_count(de_ev, window, method):
    """Step and moment counts as rsatb::calc::ComputeDos derives them from dE."""
    _, half_width = window
    span = 2.0 * half_width
    nt = max(1, int(np.ceil(span / de_ev)))
    if method == "kpm":
        n_moments = max(1, int(np.ceil(JACKSON_FWHM * half_width / de_ev)))
    else:  # corr needs the moments of the longest reconstructed time e^{-i Op nt*pi}
        n_moments = chebyshev_coeff_count(nt * np.pi)
    return nt, n_moments


def dos_cost(task, window, hermitian):
    _, n_moments = dos_moment_count(task["dE"], window, task["method"])
    total = moment_recurrence_matvecs(n_moments, hermitian)
    return total, "{} moments ({}), {} recurrence".format(
        n_moments, task["method"], "doubled" if hermitian else "plain")


def pdos_cost(task, window, hermitian):
    """One dos-kpm moment recurrence; the group projection is dot products."""
    _, n_moments = dos_moment_count(task["dE"], window, "kpm")
    total = moment_recurrence_matvecs(n_moments, hermitian)
    return total, "{} moments (kpm), {} recurrence".format(
        n_moments, "doubled" if hermitian else "plain")


def msd_cost(task, window, hermitian):
    """Gamma-affine MSD: two H0 products and one H1 first-moment product per step."""
    _, half_width = window
    time_coeffs = chebyshev_coeff_count(half_width * task["dt_fs"] / HBAR_EV_FS)
    # Doubled KPM recurrence (msd.hpp metric_density_moments): two moments
    # per apply, so n_kpm moments cost floor(n_kpm / 2) applies.
    density = (task["n_steps"] + 2) * (task["n_kpm"] // 2)
    affine = task["n_steps"] * max(time_coeffs - 1, 0) * 3
    return density + affine, "KPM density {} + affine time {}".format(density, affine)


def ac_cost(task, window, hermitian):
    """Mirrors corr_ac / corr_ac_concurrent (rsatb/calc/ac.hpp).

    Four fixed-degree Fermi expansions of exactly n_kpm - 1 applications
    each (fermi_dirac_kpm truncates nothing when the degree is pinned) seed
    the correlator; the velocity enters once per sample plus twice at setup;
    the evolution is four propagations -- stepwise at dt = pi/2, or one
    shared Chebyshev recurrence per block on the concurrent route.
    """
    nt = task["nt"]
    ns = nt * AC_OVERSAMPLE
    fermi = 4 * max(task["n_kpm"] - 1, 0)
    velocity = 3 * (ns + 2)
    if task.get("method", "concurrent") == "sequential":
        prop = 4 * ns * max(chebyshev_coeff_count(np.pi / AC_OVERSAMPLE) - 1, 0)
    else:
        block = task.get("block", 0)
        if block <= 0:
            block = min(nt, 128)
        block = min(block, nt)
        full, rem = divmod(nt, block)
        prop = full * 4 * max(chebyshev_coeff_count(block * np.pi) - 1, 0)
        if rem:
            prop += 4 * max(chebyshev_coeff_count(rem * np.pi) - 1, 0)
    total = fermi + prop + velocity
    return total, "Fermi {} + propagation {} + velocity {}".format(
        fermi, prop, velocity)


# The quantum metric runs ComputeAc on the concurrent route with its default
# block, then integrates -- no expansion of its own. The task carries neither
# 'method' nor 'block', so ac_cost's defaults are exactly its configuration.
quantum_metric_cost = ac_cost


def free_cost(task, window, hermitian):
    return 0, "reported from setup; no expansion of its own"


COSTS = {
    "dos": dos_cost,
    "pdos": pdos_cost,
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
    hermitian = operator_is_hermitian(resolved)
    print("{}".format(args.input))
    print("spectral window   [{:+.3f}, {:+.3f}] eV   (from {})".format(
        window[0] - window[1], window[0] + window[1], source))
    print("operator          {}".format(
        "Hermitian (moment recurrences doubled)" if hermitian
        else "solve form S^-1 H (plain recurrences)"))
    print("")

    total = 0
    unknown = []
    for task in resolved.get("tasks", []):
        label = task.get("label") or task["type"]
        if task["type"] not in COSTS:
            unknown.append(label)
            continue
        count, detail = COSTS[task["type"]](task, window, hermitian)
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
