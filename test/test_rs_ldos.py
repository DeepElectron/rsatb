from mpi4py import MPI  # noqa: F401  (initializes MPI before the extension)
from rsatb import (HamiltonianFile, LcaoModel, OrbitalFiles,
                   PseudopotentialFiles, RSATB, SetupOptions, Structure)

structure = Structure.from_file("./data/diamond/diamond_sc444.vasp")
model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/C2.0/hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
    pseudopotentials=PseudopotentialFiles(["C_ONCV_PBE-1.0.upf"], dir="./data/diamond/"),
)
runner = RSATB(structure, model, SetupOptions(kpoints=[[0.0, 0.0, 0.0]], nsamples=8),
               npar=1)
emin, emax = runner.energy_bounds()
print("bounds", emin, emax, flush=True)
energies = [emin + 0.3 * (emax - emin), emin + 0.5 * (emax - emin)]
out = runner.compute_rs_ldos(energies, eta=0.05 * (emax - emin), spacing=0.8,
                             n_samples=8, block=8)
print("shape", out["shape"], "n_point", out["n_point"], "degree", out["filter_degree"],
      "residual", out["sampler_residual"], flush=True)
ldos = out["ldos"]
print("min", ldos.min(), "max", ldos.max())
assert ldos.min() >= 0.0 and ldos.max() > 0.0
assert "grid" in out and out["grid"].shape == (2,) + tuple(out["shape"])
print("SMOKE OK")
