"""Running the 'tasks' section, and writing what it produced.

One task is one observable. They run in order against the same setup, so
asking for several costs one setup rather than several, and every result is
written by rank 0 with a summary that names what went where.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List

from .build import build_energies


class Result:
    """What one task produced: named arrays, named scalars, and its label."""

    def __init__(self, label: str, kind: str):
        self.label = label
        self.kind = kind
        self.arrays: Dict[str, Any] = {}
        self.scalars: Dict[str, Any] = {}
        self.seconds = 0.0
        self.file = ""


def _n_kpoint(options) -> int:
    return max(len(options.kpoints), 1)


# --- one function per task type -------------------------------------------


def _dos(calc, task, options, result):
    import numpy as np

    dos, energy = calc.compute_dos(task["dE"], method=task.get("method", "kpm"))
    result.arrays["energy"] = energy
    result.arrays["dos"] = np.asarray(dos).reshape(_n_kpoint(options), -1)


def _energy_bounds(calc, task, options, result):
    emin, emax = calc.energy_bounds()
    result.scalars["emin"] = float(emin)
    result.scalars["emax"] = float(emax)


def _ac(calc, task, options, result):
    import numpy as np

    sigma, omega, properties = calc.compute_ac(
        task["nt"], task["n_kpm"], task["temp"], task["mu"], task["axis"],
        method=task.get("method", "concurrent"), block=task.get("block", 0))
    n_k = _n_kpoint(options)
    result.arrays["omega_cm1"] = np.asarray(omega)
    result.arrays["sigma"] = np.asarray(sigma).reshape(n_k, task["nt"] + 1, 6)
    for name, values in properties.items():
        result.arrays[name] = np.asarray(values).reshape(n_k, task["nt"] + 1, 6)


def _dc(calc, task, options, result):
    import numpy as np

    sigma = calc.compute_dc(
        task["n_moments"], task["mu"], task["temp"], task["formulation"], task["axis"],
        task["partition_sections"], task["max_workspace_mb"], task["kernel"],
        method=task.get("method", "fftcheb"))
    result.arrays["sigma"] = np.asarray(sigma)


def _dc_spectrum(calc, task, options, result):
    import numpy as np

    sigma, energy = calc.compute_dc_spectrum(
        task["n_moments"], task["formulation"], task["axis"],
        task["partition_sections"], task["max_workspace_mb"], task["kernel"],
        method=task.get("method", "fftcheb"))
    result.arrays["energy"] = np.asarray(energy)
    result.arrays["sigma"] = np.asarray(sigma)


def _occupation(calc, task, options, result):
    values = calc.compute_occupation(task["mu"], task["temp"], task["indicator_threshold"],
                                     task["n_max"])
    result.scalars.update({key: _plain(value) for key, value in values.items()})


def _fermi_level(calc, task, options, result):
    guess = task.get("mu_guess", float("nan"))
    values = calc.find_fermi_level(task["n_electrons"], task["temp"], task["indicator_threshold"],
                                   task["n_max"], guess)
    result.scalars.update({key: _plain(value) for key, value in values.items()})


def _msd(calc, task, options, result):
    msd, diffusion, energy, times = calc.compute_diffusion_spectrum(
        task["n_steps"], task["dt_fs"], task["n_kpm"], axis=task["axis"])
    result.arrays["msd"] = msd
    result.arrays["diffusion"] = diffusion
    result.arrays["energy"] = energy
    result.arrays["time_fs"] = times


def _quantum_metric(calc, task, options, result):
    gxx, sigma_xx, omega = calc.compute_quantum_metric(
        task["nt"], task["n_kpm"], task["temp"], task["mu"], task["axis"],
        task["omega_min_ev"])
    result.arrays["gxx"] = gxx
    result.arrays["sigma_xx"] = sigma_xx
    result.arrays["omega_ev"] = omega


RUNNERS: Dict[str, Callable] = {
    "energy_bounds": _energy_bounds,
    "dos": _dos,
    "ac": _ac,
    "dc": _dc,
    "dc_spectrum": _dc_spectrum,
    "occupation": _occupation,
    "fermi_level": _fermi_level,
    "msd": _msd,
    "quantum_metric": _quantum_metric,
}


def _plain(value):
    """Converts a numpy scalar to something json.dump accepts."""
    if hasattr(value, "item"):
        return value.item()
    return value


def format_duration(seconds: float) -> str:
    """41.3s / 2m 08s / 1h 04m -- for humans, not for parsing."""
    if seconds < 60.0:
        return "{:.1f}s".format(seconds)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{}h {:02d}m".format(hours, minutes)
    return "{}m {:02d}s".format(minutes, secs)


def run_tasks(calc, tasks: List[dict], options, log=None, on_task_done=None) -> List[Result]:
    """Runs every task against one setup, in the order given.

    on_task_done(result), when given, is called after each task completes.
    The CLI uses it to write that task's arrays immediately: a cluster time
    limit during task N must not cost tasks 1..N-1 -- hours of finished
    physics were once lost exactly that way.
    """
    results: List[Result] = []
    for index, task in enumerate(tasks):
        kind = task["type"]
        label = task.get("label") or kind
        result = Result(label, kind)
        if log:
            log("  [{}/{}] {}".format(index + 1, len(tasks), label))
        started = time.time()
        RUNNERS[kind](calc, task, options, result)
        result.seconds = time.time() - started
        if log:
            log("        \u2713 {}".format(format_duration(result.seconds)))
        results.append(result)
        if on_task_done is not None:
            on_task_done(result)

    # The whole-run table: the setup phases printed earlier plus what the tasks
    # spent, so the cost of a run can be read off one place. Per-task wall time
    # is already in result.seconds; what this adds is the breakdown inside them,
    # and the reminder of what setup cost to get there.
    _report_profile()
    return results


def _report_profile():
    """Prints the C++ phase profile, if the extension is loaded."""
    try:
        import rsatb  # noqa: F401  -- puts the extension on sys.path
        import _rsatb
    except ImportError:
        return
    _rsatb.profile_report()


# --- writing ---------------------------------------------------------------


def resolve_output_dir(resolved: dict, base_dir: str) -> str:
    directory = resolved.get("output", {}).get("dir", ".")
    if not os.path.isabs(directory):
        directory = os.path.join(base_dir, directory)
    return directory


def write_single_result(result: Result, resolved: dict, base_dir: str, log=None) -> None:
    """Writes one task's arrays now. Called as each task finishes, so a
    scheduler killing the job mid-run costs only the task in flight.

    Only rank 0 should call this. Idempotent with write_results below, which
    skips results already on disk.
    """
    import numpy as np

    output = resolved.get("output", {})
    fmt = output.get("format", "npz")
    if not result.arrays or fmt == "none":
        return
    directory = resolve_output_dir(resolved, base_dir)
    os.makedirs(directory, exist_ok=True)
    prefix = output.get("prefix", "")

    if fmt == "npz":
        path = os.path.join(directory, "{}{}.npz".format(prefix, result.label))
        np.savez_compressed(path, **{name: np.asarray(value)
                                     for name, value in result.arrays.items()})
    else:
        path = os.path.join(directory, "{}{}.json".format(prefix, result.label))
        with open(path, "w") as handle:
            json.dump({name: np.asarray(value).tolist()
                       for name, value in result.arrays.items()}, handle)
    result.file = os.path.relpath(path, directory)
    if log:
        log("  wrote {}".format(_readable(path, base_dir)))


def write_results(results: List[Result], resolved: dict, setup: dict,
                  base_dir: str, log=None) -> str:
    """Writes every result not already written, and the summary.

    Only rank 0 should call this. Results with .file set were written
    incrementally by write_single_result and are not rewritten.
    """
    import numpy as np

    output = resolved.get("output", {})
    directory = resolve_output_dir(resolved, base_dir)
    os.makedirs(directory, exist_ok=True)
    prefix = output.get("prefix", "")

    for result in results:
        if not result.file:
            write_single_result(result, resolved, base_dir, log=log)

    summary_path = ""
    if output.get("summary", True):
        summary = {
            "input": resolved,
            "setup": setup,
            "results": [
                {
                    "label": result.label,
                    "type": result.kind,
                    "seconds": round(result.seconds, 3),
                    "scalars": result.scalars,
                    "arrays": {
                        name: {"shape": list(np.asarray(value).shape),
                               "dtype": str(np.asarray(value).dtype)}
                        for name, value in result.arrays.items()
                    },
                    "file": result.file,
                }
                for result in results
            ],
        }
        summary_path = os.path.join(directory, "{}summary.json".format(prefix))
        from .report import pretty_json
        with open(summary_path, "w") as handle:
            handle.write(pretty_json(summary) + "\n")
        if log:
            log("  wrote {}".format(_readable(summary_path, base_dir)))
    return summary_path


def _readable(path: str, base_dir: str) -> str:
    """Shortens a path against the input's directory, unless that is worse."""
    relative = os.path.relpath(path, base_dir)
    return path if relative.startswith("..") else relative
