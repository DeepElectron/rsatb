import os
import sys
import pathlib

# Add the package directory and build directory to sys.path
# to support absolute imports of the C extension (_rsatb)
_pkg_dir = pathlib.Path(__file__).parent
_build_dir = _pkg_dir / "build"

if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))

sys.setdlopenflags(os.RTLD_LAZY)

from .driver import RSATB