import sys
import os

# Do NOT add local paths to sys.path to ensure we test the installed version
# sys.path.append(os.path.abspath("rsatb"))

try:
    from rsatb import (DftbModel, HamiltonianFile, LcaoModel, OrbitalFiles, PreconditionerOptions,
                       PseudopotentialFiles, RSATB, SetupOptions, SkParameters, SktbModel,
                       Structure, sk_spline)
    print("Successfully imported RSATB from installed package")
except ImportError as e:
    print(f"Failed to import RSATB: {e}")
    sys.exit(1)

try:
    from mpi4py import MPI  # noqa: F401
    print("Successfully imported mpi4py for RSATB initialization")
except ImportError as e:
    print(f"Failed to import mpi4py: {e}")
    sys.exit(1)

print("RSATB class methods:", [m for m in dir(RSATB) if not m.startswith('_')])
print("RSATB docstring:", RSATB.__doc__)

# Basic instantiation check (requires no args or dummy args if possible)
# Since we don't have a small test file handy, we just check if the class exists and has methods.
required_methods = ['from_graph', 'compute_dos', 'compute_dc', 'compute_dc_spectrum']
missing = [name for name in required_methods if not hasattr(RSATB, name)]

if not missing:
    print("Verification successful: RSATB class exposes the FFTCheb DC APIs.")
else:
    print(f"Verification failed: RSATB class missing methods: {missing}")
    sys.exit(1)

try:
    model = SktbModel(
        basis={"C": ["pz"]},
        hamiltonian=SkParameters(
            onsite={"C": {"pz": 0.0}},
            pairs={("C", "C"): {"pp_pi": sk_spline([0.0, 1.5], [0.2, 0.2], 1.5)}},
        ),
    )
    # The basis is derived from the model, not declared alongside it.
    assert list(model.basis.type_symbols) == ["C"], model.basis.type_symbols
    assert list(model.basis.type_norb) == [1], model.basis.type_norb
    assert model.capabilities.orthogonal(), "SKTB without overlap must be orthogonal"

    structure = Structure()
    structure.n_atom = 2
    structure.pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    structure.z = [6, 6]
    structure.cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    structure.pbc = [False, False, False]

    rs = RSATB(
        structure,
        model,
        SetupOptions(
            kpoints=[[0.0, 0.0, 0.0]],
            preconditioner=PreconditionerOptions(target="none"),
        ),
        npar=1,
        verbose=False,
    )
    print("Verification successful: RSATB typed model setup works.")
except Exception as e:
    print(f"Verification failed during SKTB binding smoke test: {e}")
    sys.exit(1)
