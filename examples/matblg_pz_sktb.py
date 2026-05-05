#!/usr/bin/env python3
"""
Build the clean pz-only MATBLG tight-binding model used in

    P. Alcazar Guerrero et al.,
    Phys. Rev. Lett. 134, 126301 (2025)

from an externally prepared relaxed moire unit cell, tile it into a transport
supercell, and run a few RSATB observables.

This example matches the paper's atomistic Hamiltonian ingredients:

- one pz orbital per carbon atom
- Slater-Koster pp_pi / pp_sigma hopping channels
- distance-dependent hoppings with the paper's parameters
- 16 x 16 x 1 moire-cell supercell sampled at Gamma by default

What it does not reproduce yet:

- structural relaxation itself
- public Anderson-disorder injection through the Python RSATB API

For the exact disordered workflow from the paper, you currently need either:

1. a small public wrapper around rsatb::calc::apply_anderson_disorder, or
2. an externally generated disordered Hamiltonian loaded through HDF5.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.special as sc
from mpi4py import MPI

from rsatb import PreconditionerOptions, RSATB, SKTBModel, SetupOptions, sk_spline


PAPER_PROBE_ENERGIES_EV = np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float64)
PAPER_SUPERCELL = (16, 16, 1)


def _is_identity_supercell(supercell: tuple[int, int, int]) -> bool:
    return all(value == 1 for value in supercell)


def _safe_path_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _runtime_supercell_path(structure: Path, supercell: tuple[int, int, int]) -> Path:
    job_parts = [
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_STEP_ID"),
        os.environ.get("OMPI_COMM_WORLD_JOBID"),
        os.environ.get("PMIX_NAMESPACE"),
        os.environ.get("PMI_JOBID"),
    ]
    job_token = _safe_path_token("_".join(part for part in job_parts if part) or "local")
    supercell_token = "x".join(str(value) for value in supercell)
    stem = _safe_path_token(structure.stem)
    return Path(f"rsatb_matblg_supercell_{stem}_{supercell_token}_{job_token}.vasp")


def _prepare_structure_file(
    structure: Path,
    supercell: tuple[int, int, int],
    comm,
) -> tuple[Path, Optional[Path]]:
    """
    Return the structure file passed to RSATB.

    For the identity supercell, the original file is used directly. Otherwise
    rank 0 tiles the moire unit cell with ASE and writes a temporary POSCAR.
    RSATB.from_file then handles MPI distribution from that file.
    """
    if _is_identity_supercell(supercell):
        return structure, None

    rank = comm.Get_rank()
    runtime_structure = _runtime_supercell_path(structure, supercell)

    if rank == 0:
        from ase.io import read, write

        atoms = read(str(structure), parallel=False)
        if atoms.cell.rank < 3:
            raise ValueError(
                "Creating a supercell requires a structure with a complete 3D cell."
            )

        atoms = atoms.repeat(supercell)
        write(
            str(runtime_structure),
            atoms,
            format="vasp",
            direct=True,
            vasp5=True,
            parallel=False,
        )

    comm.Barrier()
    return runtime_structure, runtime_structure


def _cleanup_structure_file(path: Optional[Path], comm) -> None:
    if path is None:
        return
    comm.Barrier()
    if comm.Get_rank() == 0:
        path.unlink(missing_ok=True)
    comm.Barrier()


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


def _chebyshev_coeff_count(time_step: float, precision: float = 1.0e-14) -> int:
    """
    Match rsatb::calc::build_bessel_vector for Chebyshev propagation counts.

    A propagation with N retained coefficients performs N - 1 recurrence
    applications. The convergence check follows rsatb/calc/bessel.hpp: both
    J_n(t) and J_{n+1}(t) must be below precision before the series stops.
    """
    n_max = 65536
    count = 0
    for n in range(n_max):
        value = sc.jv(n, time_step)
        if abs(value) > precision:
            count += 1
            continue

        next_value = sc.jv(n + 1, time_step)
        if abs(next_value) > precision:
            count += 1
            continue

        return count

    raise RuntimeError("Bessel series not converged for MVP estimate")


def _infer_half_width_from_dos_grid(energy_dos: np.ndarray, nt: int) -> float:
    """Infer Chebyshev half-width from RSATB's DOS grid layout."""
    if nt <= 0 or energy_dos.size <= nt:
        raise ValueError("Cannot infer spectral half-width from DOS grid")
    return float(energy_dos[nt] - energy_dos[0])


def _jackson_kernel(n: int) -> np.ndarray:
    denom = float(n + 1)
    theta = np.pi / denom
    out = np.empty(n, dtype=np.float64)
    out[0] = 1.0
    if n > 1:
        m = np.arange(1, n, dtype=np.float64)
        cot_theta = np.cos(theta) / np.sin(theta)
        out[1:] = (
            (denom - m) * np.cos(theta * m) + cot_theta * np.sin(theta * m)
        ) / denom
    return out


def _occupation_projector(x: np.ndarray, beta: float, mu: float) -> np.ndarray:
    if not np.isfinite(beta) or beta <= 0.0:
        return (x <= mu).astype(np.float64)

    out = np.empty_like(x, dtype=np.float64)
    above = x >= mu
    tmp = np.exp(beta * (mu - x[above]))
    out[above] = tmp / (1.0 + tmp)
    tmp = np.exp(beta * (x[~above] - mu))
    out[~above] = 1.0 / (1.0 + tmp)
    return out


def _fermi_kpm_coeff_count(
    n_kpm: int,
    center: float,
    half_width: float,
    temp: float,
    mu: float,
    precision: float = 1.0e-14,
) -> int:
    """Match the coefficient count used by rsatb::calc::fermi_dirac_kpm."""
    kb_ev_per_k = 8.617333262e-5
    beta = (1.0 / (temp * kb_ev_per_k)) if temp > 0.0 else 0.0
    beta_re = beta * half_width
    mu_re = (mu - center) / half_width

    dtheta = np.pi / float(n_kpm)
    x = np.cos(dtheta * np.arange(2 * n_kpm, dtype=np.float64))
    values = _occupation_projector(x, beta_re, mu_re).astype(np.complex128)
    coeffs = np.fft.fft(values)[:n_kpm]
    coeffs = coeffs.real * _jackson_kernel(n_kpm) / float(n_kpm)
    coeffs[0] *= 0.5

    keep = n_kpm
    for i in range(n_kpm - 1):
        if abs(coeffs[i]) <= precision and abs(coeffs[i + 1]) <= precision:
            keep = i + 1
            break
    return max(1, keep)


def _dos_mvp_count(nt: int, time_coeff_count: int) -> int:
    return nt * max(time_coeff_count - 1, 0)


def _gamma_affine_msd_mvp_counts(
    n_steps: int,
    n_kpm: int,
    time_coeff_count: int,
) -> tuple[int, int, int]:
    """
    Return (density_kpm_h0, affine_time_all_sparse, total_all_sparse).

    In gamma-affine MSD, each affine Chebyshev recurrence application performs
    two H0 products and one H1 first-moment product.
    """
    density_kpm_h0 = (n_steps + 2) * max(n_kpm - 1, 0)
    affine_time_all_sparse = n_steps * max(time_coeff_count - 1, 0) * 3
    return (
        density_kpm_h0,
        affine_time_all_sparse,
        density_kpm_h0 + affine_time_all_sparse,
    )


def _quantum_metric_mvp_counts(
    nt: int,
    fermi_coeff_count: int,
    time_coeff_count: int,
) -> tuple[int, int, int]:
    """
    Return (h0_mvp, velocity_mvp, total_all_sparse) for compute_quantum_metric.

    The quantum-metric entry point calls AC conductivity. The Fermi KPM count is
    passed in after matching the same coefficient truncation used by RSATB. AC
    applies four Fermi/time propagations, and it applies three velocity
    operators before the loop, three at t=0 on the left, and three inside each
    time step.
    """
    h0_mvp = (
        4 * max(fermi_coeff_count - 1, 0)
        + 4 * nt * max(time_coeff_count - 1, 0)
    )
    velocity_mvp = 3 * (nt + 2)
    return h0_mvp, velocity_mvp, h0_mvp + velocity_mvp


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
        help="Relaxed MATBLG moire unit-cell structure file in XYZ or POSCAR format.",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=PAPER_SUPERCELL,
        help=(
            "Integer tiling applied to the input moire unit cell before RSATB setup. "
            "Defaults to the paper transport value: 16 16 1."
        ),
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
        default=3500,
        help="Number of KPM moments for MSD / AC / QM.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of MSD time steps.",
    )
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=10.0,
        help="MSD time step in femtoseconds.",
    )
    parser.add_argument(
        "--omega-min-ev",
        type=float,
        default=0.0,
        help="Lower frequency cutoff used in the quantum-metric integral.",
    )
    parser.add_argument(
        "--energy-bounds",
        type=float,
        nargs=2,
        metavar=("EMIN_EV", "EMAX_EV"),
        default=None,
        help=(
            "Explicit spectral bounds passed to SetupOptions. For large MATBLG "
            "cells this is usually faster and more robust than automatic bound estimation."
        ),
    )
    parser.add_argument(
        "--bounds-zero-tol",
        type=float,
        default=1.0e-12,
        help="Zero-Hamiltonian tolerance used by automatic spectral bound estimation.",
    )
    parser.add_argument(
        "--bounds-max-matvecs",
        type=int,
        default=10000,
        help="Maximum PRIMME matvecs used by automatic spectral bound estimation.",
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
    args = parser.parse_args()

    energy_bounds = None
    if args.energy_bounds is not None:
        energy_bounds = tuple(args.energy_bounds)
        if not energy_bounds[0] < energy_bounds[1]:
            parser.error("--energy-bounds requires EMIN_EV < EMAX_EV")
    if args.bounds_zero_tol < 0.0:
        parser.error("--bounds-zero-tol must be non-negative")
    if args.bounds_max_matvecs < 0:
        parser.error("--bounds-max-matvecs must be non-negative")
    if args.nt <= 0:
        parser.error("--nt must be positive")
    if args.n_kpm <= 0:
        parser.error("--n-kpm must be positive")
    if args.n_steps < 0:
        parser.error("--n-steps must be non-negative")
    if args.dt_fs < 0.0:
        parser.error("--dt-fs must be non-negative")
    supercell = tuple(args.supercell)
    if any(value < 1 for value in supercell):
        parser.error("--supercell values must be positive integers")

    sktb_config = _paper_pz_sktb_config(args.length_unit)

    model = SKTBModel.native(
        basis=sktb_config["basis"],
        onsite=sktb_config["onsite"],
        pairs=sktb_config["pairs"],
    )
    atomic_spec = model.prepare_atomic_spec_from_file(str(args.structure))

    comm = MPI.COMM_WORLD
    runtime_structure, cleanup_path = _prepare_structure_file(
        args.structure,
        supercell,
        comm,
    )

    try:
        rsatb = RSATB.from_file(
            npar=args.npar,
            filename=str(runtime_structure),
            r_max_per_type=atomic_spec.r_max_per_type,
            type_norb=atomic_spec.type_norb,
            verbose=args.verbose,
        )
    except Exception:
        _cleanup_structure_file(cleanup_path, comm)
        raise
    _cleanup_structure_file(cleanup_path, comm)

    # The current MSD entrypoint is Gamma-only, which also matches the
    # supercell real-space use case in the paper.

    print("Setting up RSATB SKTB...")

    rsatb.setup_sktb(
        model,
        SetupOptions(
            kpoints=[[0.0, 0.0, 0.0]],
            preconditioner=PreconditionerOptions(mode="None"),
            nsamples=1,
            compute_bounds=energy_bounds is None,
            energy_bounds=energy_bounds,
            bounds_zero_tol=args.bounds_zero_tol,
            bounds_max_matvecs=args.bounds_max_matvecs,
        ),
    )

    print("Running RSATB observables...")

    time_coeff_count = _chebyshev_coeff_count(np.pi)
    dos_mvp = _dos_mvp_count(args.nt, time_coeff_count)

    print(f"  DOS... estimated MVPs: {dos_mvp}")
    dos, energy_dos = rsatb.compute_dos(args.nt)

    center = float(energy_dos[args.nt])
    half_width = _infer_half_width_from_dos_grid(energy_dos, args.nt)
    msd_time_coeff_count = _chebyshev_coeff_count(
        half_width * args.dt_fs / 0.6582119569
    )
    msd_kpm_mvp, msd_time_mvp, msd_mvp = _gamma_affine_msd_mvp_counts(
        args.n_steps,
        args.n_kpm,
        msd_time_coeff_count,
    )

    print(
        "  MSD and diffusion... estimated gamma-affine MVPs: "
        f"{msd_mvp} (KPM H0: {msd_kpm_mvp}, affine time H0/H1: {msd_time_mvp})"
    )
    msd, diffusion, energy_msd, time_fs = rsatb.compute_diffusion_spectrum(
        args.n_steps,
        args.dt_fs,
        args.n_kpm,
        axis="x",
    )

    qm_h0_mvp, qm_velocity_mvp, qm_mvp = _quantum_metric_mvp_counts(
        args.nt,
        _fermi_kpm_coeff_count(args.n_kpm, center, half_width, temp=0.0, mu=0.0),
        time_coeff_count,
    )

    print(
        "  Quantum metric and conductivity... estimated MVPs: "
        f"{qm_mvp} (KPM/time H0: {qm_h0_mvp}, velocity: {qm_velocity_mvp})"
    )
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
        supercell=np.array(supercell, dtype=np.int32),
        paper_probe_energies_ev=PAPER_PROBE_ENERGIES_EV,
    )

    print(f"Wrote {args.output}")
    print("Stored arrays: dos, energy_dos, msd, diffusion, energy_msd, time_fs, gxx, sigma_xx, omega_ev, supercell, paper_probe_energies_ev")


if __name__ == "__main__":
    main()
