from mpi4py import MPI  # noqa: F401  (initializes MPI before RSATB)
from rsatb import (
    HamiltonianFile,
    LcaoModel,
    OrbitalFiles,
    PreconditionerOptions,
    PseudopotentialFiles,
    RSATB,
    SetupOptions,
    Structure,
)

kpoints = [
    [0. , 0. , 0. ]
]

model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/C2.0/hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
    pseudopotentials=PseudopotentialFiles(["C_ONCV_PBE-1.0.upf"], dir="./data/diamond/"),
)

model_runner = RSATB(
    Structure.from_file("./data/diamond/diamond_sc111.vasp"),
    model,
    SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        preconditioner=PreconditionerOptions(
            target="overlap",
            tol=1e-7,
            sptol=1e-5,
            maxiter=10,
        ),
    ),
    npar=1,
    verbose=True,
)

model_runner.test_solve()
