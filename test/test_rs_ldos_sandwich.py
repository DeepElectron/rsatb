"""Real-space LDOS through the Z H Z operator form.

Same fixture and same physical assertions as test_rs_ldos.py, which runs the
default S^-1 H form -- see that file for why these particular checks are the
ones that fail on a mismatched Hamiltonian.

What this file adds is the form itself. `operator_form="sandwich"` releases the
overlap after the preconditioner is built, and the LDOS used to read that
release as "orthogonal model, no basis functions to collocate" and refuse to
run. It never needed S: the sandwich form iterates Z^H H Z, its seed already
has covariance I, and the metric reappears as one Z apply at read-out. So the
map must come out with the same physics the solve form produces -- the same
spectral window, the same gap.
"""
from mpi4py import MPI  # noqa: F401  (initializes MPI before the extension)
from rsatb import (HamiltonianFile, LcaoModel, OrbitalFiles,
                   PreconditionerOptions, RSATB,
                   SetupOptions, Structure)

GAP_LO, GAP_HI = 13.43, 17.78

structure = Structure.from_file("./data/diamond/conf.vasp", supercell=[4, 4, 4])
model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
)
runner = RSATB(
    structure,
    model,
    SetupOptions(
        kpoints=[[0.0, 0.0, 0.0]],
        nsamples=8,
        preconditioner=PreconditionerOptions(operator_form="sandwich"),
    ),
    npar=1,
)

# The bounds are measured on Z H Z here rather than on the (H, S) pencil. They
# are the same numbers -- that is the whole premise of the form -- so the same
# window the solve-form test asserts has to hold.
emin, emax = runner.energy_bounds()
print("bounds", emin, emax, flush=True)
assert -12.0 < emin < -5.0, f"unexpected emin {emin}: is Z H Z the right operator?"
assert 200.0 < emax < 350.0, f"unexpected emax {emax}: is Z H Z the right operator?"

energies = [2.0, 6.0, 15.6]
out = runner.compute_rs_ldos(energies, eta=1.0, spacing=0.6, n_samples=16, block=8)
print("shape", out["shape"], "n_point", out["n_point"], "degree", out["filter_degree"],
      "residual", out["sampler_residual"], flush=True)

# No metric sampler is built in this form, so there is no certificate to
# report and the field keeps its "not measured" sentinel. This is the cheap
# observable proof that the sandwich path, not the solve path, actually ran.
assert out["sampler_residual"] == -1.0, "a metric sampler was built in the sandwich form"

ldos = out["ldos"]
print("min", ldos.min(), "max", ldos.max())
assert ldos.min() >= 0.0 and ldos.max() > 0.0
assert "grid" in out and out["grid"].shape == (len(energies),) + tuple(out["shape"])

weight = out["grid"].reshape(len(energies), -1).sum(axis=1)
band = min(weight[0], weight[1])
assert GAP_LO < energies[2] < GAP_HI
ratio = weight[2] / band
print("gap / band =", ratio)
assert ratio < 0.1, f"no gap in the LDOS (ratio {ratio})"

print("SMOKE OK")
