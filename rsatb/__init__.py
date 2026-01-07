# package/core/__init__.py
from importlib import import_module
import sys, pathlib

_build_dir = pathlib.Path(__file__).with_name("build")
if _build_dir.is_dir():
    sys.path.insert(0, str(_build_dir))

import os
sys.setdlopenflags(os.RTLD_LAZY)
_binding = import_module(".binding", package=__name__)   # name of xxx.so without suffix
globals().update(_binding.__dict__)    # re‑export its symbols (optional)