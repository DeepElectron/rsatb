#!/usr/bin/env python3
"""
Build the clean pz-only MATBLG tight-binding model used in

    P. Alcazar Guerrero et al.,
    Phys. Rev. Lett. 134, 126301 (2025)

from an externally prepared relaxed structure and run a few RSATB observables.

This example matches the paper's atomistic Hamiltonian ingredients:

- one pz orbital per carbon atom
- Slater-Koster pp_pi / pp_sigma hopping channels
- distance-dependent hoppings with the paper's parameters
- periodic supercell sampled at Gamma

What it does not reproduce yet:

- structural relaxation itself
- public Anderson-disorder injection through the Python RSATB API

For the exact disordered workflow from the paper, you currently need either:

1. a small public wrapper around rsatb::calc::apply_anderson_disorder, or
2. an externally generated disordered Hamiltonian loaded through HDF5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rsatb import PreconditionerOptions, RSATB, SKTBModel, SetupOptions, sk_spline


PAPER_PROBE_ENERGIES_EV = np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float64)


def _paper_pz_sktb_config(length_unit: str = "angstrom", n_samples: int = 2048) -> dict:
    """
    Return a compact SKTB.native config for the clean MATBLG model in the paper.

    The paper parameters are:
      Vpp_pi^0 = -2.7 eV
      Vpp_sigma^0 = 0.3675 eV
      q_pi / a0 = q_sigma / d0 = 22.18 nm^-1
      a0 = 0.1439 nm
      d0 = 0.33 nm
      rc = 0.614 nm
      lambda_c = 0.0265 nm

    RSATB needs a finite graph cutoff, so we truncate the smooth paper cutoff at
    rc + 8 * lambda_c, where the residual tail is already very small.
    """
    if n_samples < 16:
        raise ValueError("n_samples must be at least 16")

    unit = length_unit.strip().lower()
    if unit == "angstrom":
        a0 = 1.439
        d0 = 3.30
        rc = 6.14
        lambda_c = 0.265
        decay_inv_length = 2.218
    elif unit == "nm":
        a0 = 0.1439
        d0 = 0.33
        rc = 0.614
        lambda_c = 0.0265
        decay_inv_length = 22.18
    else:
        raise ValueError("length_unit must be 'angstrom' or 'nm'")

    vpp_pi_0 = -2.7
    vpp_sigma_0 = 0.3675

    q_pi = decay_inv_length * a0
    q_sigma = decay_inv_length * d0

    r_cut = rc + 8.0 * lambda_c
    r = np.linspace(1.0e-6, r_cut, n_samples, dtype=np.float64)
    cutoff = 1.0 / (1.0 + np.exp((r - rc) / lambda_c))

    vpp_pi = vpp_pi_0 * np.exp(q_pi * (1.0 - r / a0)) * cutoff
    vpp_sigma = vpp_sigma_0 * np.exp(q_sigma * (1.0 - r / d0)) * cutoff

    return {
        "basis": {
            "C": ["pz"],
        },
        "onsite": {
            "C": {"pz": 0.0},
        },
        "pairs": {
            ("C", "C"): {
                "H": {
                    "pp": {
                        "pi": sk_spline(r, vpp_pi, r_cut),
                        "sigma": sk_spline(r, vpp_sigma, r_cut),
                    },
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean pz-only MATBLG SKTB setup aligned with "
            "Guerrero et al. PRL 134, 126301 (2025)."
        )
    )
    parser.add_argument(
        "structure",
        type=Path,
        help="Relaxed MATBLG structure file in XYZ or POSCAR format.",
    )
    parser.add_argument(
        "--length-unit",
        choices=("angstrom", "nm"),
        default="angstrom",
        help="Length unit used in the structure file and SK parameters.",
    )
    parser.add_argument(
        "--npar",
        type=int,
        default=1,
        help="RSATB process partition count.",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=1024,
        help="Number of DOS / AC / QM Chebyshev time steps.",
    )
    parser.add_argument(
        "--n-kpm",
        type=int,
        default=2048,
        help="Number of KPM moments for MSD / AC / QM.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of MSD time steps.",
    )
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=1.0,
        help="MSD time step in femtoseconds.",
    )
    parser.add_argument(
        "--omega-min-ev",
        type=float,
        default=0.0,
        help="Lower frequency cutoff used in the quantum-metric integral.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("matblg_pz_sktb_results.npz"),
        help="Output NPZ file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RSATB output.",
    )
    parser.add_argument(
        "--paper-clean-defaults",
        action="store_true",
        help=(
            "Use paper-aligned clean-transport defaults where the main text is explicit: "
            "n_kpm=3500 and dt_fs=10. If n_steps is left at its generic default, it is "
            "raised to 200 so the ballistic MSD trace extends to 2 ps."
        ),
    )
    args = parser.parse_args()

    if args.paper_clean_defaults:
        if args.n_kpm == 2048:
            args.n_kpm = 3500
        if args.dt_fs == 1.0:
            args.dt_fs = 10.0
        if args.n_steps == 50:
            args.n_steps = 200

    sktb_config = _paper_pz_sktb_config(args.length_unit)

    model = SKTBModel.native(
        basis=sktb_config["basis"],
        onsite=sktb_config["onsite"],
        pairs=sktb_config["pairs"],
    )
    atomic_spec = model.prepare_atomic_spec_from_file(str(args.structure))
    rsatb = RSATB.from_file(
        npar=args.npar,
        filename=str(args.structure),
        r_max_per_type=atomic_spec.r_max_per_type,
        type_norb=atomic_spec.type_norb,
        verbose=args.verbose,
    )

    # The current MSD entrypoint is Gamma-only, which also matches the
    # supercell real-space use case in the paper.

    print("Setting up RSATB SKTB...")

    rsatb.setup_sktb(
        model,
        SetupOptions(
            kpoints=[[0.0, 0.0, 0.0]],
            preconditioner=PreconditionerOptions(mode="None"),
            nsamples=1,
            compute_bounds=True,
        ),
    )

    print("Running RSATB observables...")

    print("  DOS...")
    dos, energy_dos = rsatb.compute_dos(args.nt)

    print("  MSD and diffusion...")
    msd, energy_msd, time_fs = rsatb.compute_msd_spectrum(
        args.n_steps,
        args.dt_fs,
        args.n_kpm,
        axis="x",
    )
    diffusion = 0.5 * np.gradient(msd, time_fs, axis=1, edge_order=1)

    print("  Quantum metric and conductivity...")
    gxx, sigma_xx, omega_ev = rsatb.compute_quantum_metric(
        args.nt,
        args.n_kpm,
        temp=0.0,
        mu=0.0,
        axis="abc",
        omega_min_ev=args.omega_min_ev,
    )

    np.savez(
        args.output,
        dos=dos,
        energy_dos=energy_dos,
        msd=msd,
        diffusion=diffusion,
        energy_msd=energy_msd,
        time_fs=time_fs,
        gxx=gxx,
        sigma_xx=sigma_xx,
        omega_ev=omega_ev,
        paper_probe_energies_ev=PAPER_PROBE_ENERGIES_EV,
    )

    print(f"Wrote {args.output}")
    print("Stored arrays: dos, energy_dos, msd, diffusion, energy_msd, time_fs, gxx, sigma_xx, omega_ev, paper_probe_energies_ev")


if __name__ == "__main__":
    main()
