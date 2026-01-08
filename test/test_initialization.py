from mpi4py import MPI
import sys
import os
import numpy as np

# Ensure we test the installed version
try:
    import rsatb
    from rsatb import RSATB
    print(f"Successfully imported RSATB from {rsatb.__file__}")
except ImportError as e:
    print(f"Failed to import RSATB: {e}")
    sys.exit(1)

print("\nTesting RSATB.from_points...")
try:
    npar = 1
    pos = [0.0, 0.0, 0.0, 0.25, 0.25, 0.25]
    z = [14, 14]
    cell = [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]
    pbc = [True, True, True]
    r_max_per_type = [5.0]
    type_norb = [13]
    
    obj = RSATB.from_points(npar, pos, z, cell, pbc, r_max_per_type, type_norb)
    print("Successfully initialized RSATB using from_points")
    print(f"Kernel info: {obj.get_kernel_info()}")
except Exception as e:
    print(f"Failed to initialize RSATB using from_points: {e}")

print("\nTesting RSATB.from_file...")
# We'll try to use the silicon STRU file if it exists
stru_path = "test/data/silicon/STRU"
if os.path.exists(stru_path):
    try:
        obj = RSATB.from_file(npar, stru_path, r_max_per_type, type_norb)
        print(f"Successfully initialized RSATB using from_file with {stru_path}")
    except Exception as e:
        print(f"Failed to initialize RSATB using from_file: {e}")
else:
    print(f"Skipping from_file test: {stru_path} not found")

print("\nTesting RSATB.from_raw_data...")
try:
    # Dummy data for raw data test
    n_atom = 2
    N_atom = 2
    atom_offset = 0
    n_edge = 1
    N_edge = 1
    atom_index = [0, 1]
    atom_type = [0, 0]
    edge_index = [0, 1]
    type_norb = [13]
    edge_shift_vec = [0, 0, 0]
    cell = [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]
    pos = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    
    obj = RSATB.from_raw_data(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                              atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
                              cell, pos)
    print("Successfully initialized RSATB using from_raw_data")
except Exception as e:
    print(f"Failed to initialize RSATB using from_raw_data: {e}")

print("\nVerification of classmethods complete.")
