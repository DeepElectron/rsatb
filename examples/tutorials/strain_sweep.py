"""Graphene DOS under biaxial strain: a parameter sweep through the Python API.

A JSON input describes one calculation. A sweep -- here, five strains, each
with its own geometry -- is a loop, so it is written against the objects the
input is built from:

    python examples/tutorials/strain_sweep.py
    mpirun -n 4 python examples/tutorials/strain_sweep.py

The hopping decays exponentially with bond length,

    t(r) = t0 * exp(-beta * (r / a0 - 1)),   t0 = -2.7 eV, a0 = 1.42 A, beta = 3.37,

which is exactly the 'exp_poly' radial form P(r) * exp(-lambda * r) with a
one-term polynomial P = t0 * exp(beta) and lambda = beta / a0. Stretching the
lattice therefore narrows the band: the van Hove peaks at +-|t| move inwards.

Writes strain_sweep.npz next to this script (rank 0 only).
"""

import math
import pathlib

from mpi4py import MPI  # initialise MPI before any native object is built

import numpy as np
import rsatb
from rsatb import RSATB, SetupOptions, SkParameters, SktbModel, Structure, sk_exp_poly

T0, A0, BETA = -2.7, 1.42, 3.37
STRAINS = [0.00, 0.025, 0.05, 0.075, 0.10]
TILING = [128, 128, 1]                  # 32,768 atoms


def graphene(strain):
    """Primitive graphene cell, stretched isotropically in-plane."""
    s = 1.0 + strain
    a = math.sqrt(3.0) * A0 * s
    cell = [a, 0.0, 0.0,
            a / 2, a * math.sqrt(3.0) / 2, 0.0,
            0.0, 0.0, 20.0]                         # row-major, Angstrom
    pos = [0.0, 0.0, 10.0,
           0.0, A0 * s, 10.0]                       # flat, three values per atom
    return Structure.from_arrays(pos=pos, z=[6, 6], cell=cell,
                                 pbc=[True, True, False], supercell=TILING)


def model():
    """One pz per carbon; nearest-neighbour hopping that decays with distance."""
    hopping = sk_exp_poly(coeffs=[T0 * math.exp(BETA)], r_cut=2.0,
                          lambda_value=BETA / A0, r_on=1.8)
    parameters = SkParameters(onsite={"C": {"pz": 0.0}},
                              pairs={("C", "C"): {"pp_pi": hopping}})
    return SktbModel(basis={"C": ["pz"]}, hamiltonian=parameters)


def main():
    rsatb.set_log_level("silent")
    rank = MPI.COMM_WORLD.Get_rank()
    options = SetupOptions(kpoints=[[0.0, 0.0, 0.0]], nsamples=4)

    curves = []
    for strain in STRAINS:
        calc = RSATB(graphene(strain), model(), options)
        dos, energy = calc.compute_dos(dE=0.1)      # flat: (n_kpoint * n_energy,)
        dos = dos.reshape(-1, len(energy))[0]        # the single (Gamma) k-point
        emin, emax = calc.energy_bounds()
        # Per state: the DOS is in states/eV of the cell and integrates to n_states.
        curves.append((energy, dos / calc.n_states()))
        if rank == 0:
            t = T0 * math.exp(-BETA * strain)
            print("strain {:5.1%}   |t| = {:.3f} eV   window [{:.2f}, {:.2f}] eV".format(
                strain, abs(t), emin, emax))

    if rank == 0:
        out = pathlib.Path(__file__).with_name("strain_sweep.npz")
        np.savez(out, strains=np.array(STRAINS),
                 **{"energy_{}".format(i): e for i, (e, _) in enumerate(curves)},
                 **{"dos_{}".format(i): d for i, (_, d) in enumerate(curves)})
        print("wrote", out)


if __name__ == "__main__":
    main()
