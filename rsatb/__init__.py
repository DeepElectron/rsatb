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

if hasattr(os, "RTLD_LAZY"):
    sys.setdlopenflags(os.RTLD_LAZY)

# The model and option types live in C++ and are bound directly, so importing
# any of them loads the extension. That import is deferred to first use: the
# JSON interface -- `rsatb.inputs` and everything `rsatb check`, `rsatb doc`,
# `rsatb schema` and `rsatb template` do -- is pure Python, and has to keep
# working where the solver cannot be built or run.
_DRIVER_EXPORTS = (
    "BasisSpec",
    "Capabilities",
    "DftbModel",
    "HamiltonianFile",
    "LcaoModel",
    "Model",
    "OrbitalFiles",
    "PreconditionerOptions",
    "PseudopotentialFiles",
    "RSATB",
    "SetupOptions",
    "SkParameters",
    "SkRadialFunction",
    "SktbModel",
    "SpinSpec",
    "Structure",
    "TightBindingFileModel",
    "atomic_number",
    "sk_exp_poly",
    "sk_spline",
)

__all__ = list(_DRIVER_EXPORTS) + ["inputs"]


def __getattr__(name):
    if name in _DRIVER_EXPORTS:
        import importlib
        value = getattr(importlib.import_module(__name__ + ".driver"), name)
    elif name == "inputs":
        # import_module rather than `from . import inputs`: the latter re-enters
        # this hook through _handle_fromlist during `from rsatb import *`, which
        # recurses until the stack runs out.
        import importlib
        value = importlib.import_module(__name__ + ".inputs")
    else:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    # Bind it, so this hook runs at most once per name.
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
