from __future__ import annotations

__submodules__ = ["ModuleBase", "ModuleNAO"]
__all__ = list(__submodules__)

def __getattr__(attr):
    if attr == "ModuleBase":
        import pyabacus.ModuleBase as ModuleBase
        return ModuleBase
    elif attr == "ModuleNAO":
        import pyabacus.ModuleNAO as ModuleNAO
        return ModuleNAO
    else:
        raise AttributeError(f"module {__name__} has no attribute {attr}")