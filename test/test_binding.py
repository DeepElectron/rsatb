import sys
import os

# Do NOT add local paths to sys.path to ensure we test the installed version
# sys.path.append(os.path.abspath("rsatb"))

try:
    from rsatb import RSATB
    print("Successfully imported RSATB from installed package")
except ImportError as e:
    print(f"Failed to import RSATB: {e}")
    sys.exit(1)

print("RSATB class methods:", [m for m in dir(RSATB) if not m.startswith('_')])
print("RSATB docstring:", RSATB.__doc__)

# Basic instantiation check (requires no args or dummy args if possible)
# Since we don't have a small test file handy, we just check if the class exists and has methods.
required_methods = ['setup', 'compute_dc', 'compute_dc_spectrum']
missing = [name for name in required_methods if not hasattr(RSATB, name)]

if not missing:
    print("Verification successful: RSATB class exposes the FFTCheb DC APIs.")
else:
    print(f"Verification failed: RSATB class missing methods: {missing}")
