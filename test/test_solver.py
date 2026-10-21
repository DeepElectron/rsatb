from mpi4py import MPI  # noqa: F401  (initializes MPI before RSATB)
from pathlib import Path
from rsatb import (
    HamiltonianFile,
    LcaoModel,
    OrbitalFiles,
    PreconditionerOptions,
    RSATB,
    SetupOptions,
    Structure,
)

kpoints = [
    [0. , 0. , 0. ]
    ]
DIAMOND_DIR = Path(__file__).resolve().parent / "data" / "diamond"

model = LcaoModel(
    hamiltonian=HamiltonianFile(str(DIAMOND_DIR / "hamiltonians.h5"), format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir=str(DIAMOND_DIR)),
)

model_runner = RSATB(
    Structure.from_file(str(DIAMOND_DIR / "conf.vasp")),
    model,
    SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        preconditioner=PreconditionerOptions(
            tol=1e-7,
            sptol=1e-5,
            maxiter=10,
        ),
    ),
    npar=1,
)

model_runner.test_solve()
