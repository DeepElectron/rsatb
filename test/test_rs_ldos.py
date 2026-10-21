"""Real-space LDOS smoke test.

The Hamiltonian must describe the SAME atoms as the structure. This file used
to load the 2-atom primitive-cell Hamiltonian (block keys over atom indices
0..1) against a pre-tiled 128-atom POSCAR without declaring the supercell, so
the other 126 atoms got zero rows, so the operator carried a ~1100-state null space, the spectrum
collapsed to a delta at E=0 with no band structure, and the spectral bounds
came out -483..751 eV instead of -8.6..271 eV.

It passed anyway, because `ldos >= 0` and `max > 0` hold for a garbage
operator too. The assertions below are chosen to fail on that mismatch: the
spectral window is checked, and the LDOS is required to be strongly suppressed
in diamond's gap, which a null-space-dominated spectrum has nowhere.
"""
from pathlib import Path

from mpi4py import MPI  # noqa: F401  (initializes MPI before the extension)
from rsatb import (HamiltonianFile, LcaoModel, OrbitalFiles,
                   RSATB, SetupOptions, Structure)

# Diamond in this basis: valence band -8.6 to 13.4 eV, then a 4.35 eV gap.
GAP_LO, GAP_HI = 13.43, 17.78
DIAMOND_DIR = Path(__file__).resolve().parent / "data" / "diamond"

structure = Structure.from_file(str(DIAMOND_DIR / "conf.vasp"), supercell=[4, 4, 4])
model = LcaoModel(
    hamiltonian=HamiltonianFile(str(DIAMOND_DIR / "hamiltonians.h5"), format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir=str(DIAMOND_DIR)),
)
runner = RSATB(structure, model, SetupOptions(kpoints=[[0.0, 0.0, 0.0]], nsamples=8),
               npar=1)
emin, emax = runner.energy_bounds()
print("bounds", emin, emax, flush=True)
# A Hamiltonian covering only part of the cell leaves a null space that widens
# the window enormously; the matched one lands here.
assert -12.0 < emin < -5.0, f"unexpected emin {emin}: does H cover every atom?"
assert 200.0 < emax < 350.0, f"unexpected emax {emax}: does H cover every atom?"

# Two valence energies, one in the gap.
energies = [2.0, 6.0, 15.6]
out = runner.compute_rs_ldos(energies, eta=1.0, spacing=0.6, n_samples=16, block=8)
print("shape", out["shape"], "n_point", out["n_point"], "degree", out["filter_degree"],
      "residual", out["sampler_residual"], flush=True)

ldos = out["ldos"]
print("min", ldos.min(), "max", ldos.max())
assert ldos.min() >= 0.0 and ldos.max() > 0.0
assert "grid" in out and out["grid"].shape == (len(energies),) + tuple(out["shape"])

# The gap must be far emptier than the bands. Integrated over the cell rather
# than pointwise, because the stochastic error is ~sqrt(2/n_samples) at a point
# and much smaller once integrated.
weight = out["grid"].reshape(len(energies), -1).sum(axis=1)
band = min(weight[0], weight[1])
assert GAP_LO < energies[2] < GAP_HI
ratio = weight[2] / band
print("gap / band =", ratio)
assert ratio < 0.1, f"no gap in the LDOS (ratio {ratio}) -- check H matches the cell"

print("SMOKE OK")
