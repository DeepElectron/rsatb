"""The JSON input interface.

An RSATB calculation is described by one JSON file, and this package is what
reads it: `spec` declares every field once, `check` validates a document
against that declaration, `build` turns a validated document into the objects
the library is constructed from, and `tasks` runs the observables it asks for.

`spec`, `nodes`, `check`, `report` and `templates` import nothing compiled, so
an input can be written and validated where the solver cannot run. Only `build`
and `tasks` reach into the extension.

    from rsatb.inputs import load_and_resolve, build_calculation, run_tasks

    resolved, issues = load_and_resolve("run.json")
    calculation, model, options = build_calculation(resolved)
    results = run_tasks(calculation, resolved["tasks"], options)
"""

from .check import (capabilities, check_consistency, check_files_exist, load,
                    load_and_resolve, resolve, summarize)
from .nodes import ERROR, Issue, WARNING
from .spec import INPUT_VERSION, ROOT, topics

__all__ = [
    "ERROR",
    "INPUT_VERSION",
    "Issue",
    "ROOT",
    "WARNING",
    "capabilities",
    "check_consistency",
    "check_files_exist",
    "load",
    "load_and_resolve",
    "resolve",
    "summarize",
    "topics",
    # Reaching into the extension; imported on demand below.
    "build_calculation",
    "run_tasks",
    "write_results",
]


def __getattr__(name):
    """Defers the extension-backed entry points until they are asked for."""
    if name == "build_calculation":
        from .build import build_calculation
        return build_calculation
    if name in ("run_tasks", "write_results"):
        from . import tasks
        return getattr(tasks, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
