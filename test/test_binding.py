import sys
import os

# Do NOT add local paths to sys.path to ensure we test the installed version
# sys.path.append(os.path.abspath("rsatb"))

try:
    from rsatb import DFTBModel, Hdf5Input, LCAOModel, PreconditionerOptions, RSATB, SKTBModel, SetupOptions, sk_spline
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
required_methods = ['setup', 'setup_model', 'setup_lcao', 'setup_sktb', 'setup_dftb', 'setup_config', 'compute_dc', 'compute_dc_spectrum']
missing = [name for name in required_methods if not hasattr(RSATB, name)]

if not missing:
    print("Verification successful: RSATB class exposes the FFTCheb DC APIs.")
else:
    print(f"Verification failed: RSATB class missing methods: {missing}")
    sys.exit(1)

try:
    sktb_config = {
        "basis": {"C": ["pz"]},
        "onsite": {"C": {"pz": 0.0}},
        "pairs": {
            ("C", "C"): {
                "H": {
                    "pp": {
                        "pi": sk_spline([0.0, 1.5], [0.2, 0.2], 1.5),
                    },
                },
            },
        },
    }
    model = SKTBModel.native(
        basis=sktb_config["basis"],
        onsite=sktb_config["onsite"],
        pairs=sktb_config["pairs"],
    )
    atomic_spec = model.prepare_atomic_spec_from_atomic_numbers([6, 6])
    rs = RSATB.from_points(
        1,
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [6, 6],
        [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
        [False, False, False],
        atomic_spec.r_max_per_type,
        atomic_spec.type_norb,
        verbose=False,
    )
    rs.setup_model(
        model,
        SetupOptions(
            kpoints=[[0.0, 0.0, 0.0]],
            preconditioner=PreconditionerOptions(mode="None"),
        ),
    )
    print("Verification successful: RSATB typed model setup works.")
except Exception as e:
    print(f"Verification failed during SKTB binding smoke test: {e}")
    sys.exit(1)
