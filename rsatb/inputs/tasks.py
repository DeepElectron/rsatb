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
        # True when every rank holds a different piece of the arrays and must
        # write its own file. The alternative -- gathering to rank 0 -- is
        # what the compact grid layout exists to avoid, so a distributed
        # result is never collected just to be written.
        self.distributed = False


def _n_kpoint(options) -> int:
    return max(len(options.kpoints), 1)


# --- one function per task type -------------------------------------------


def _dos(calc, task, options, result):
    import numpy as np

    dos, energy = calc.compute_dos(task["dE"], method=task.get("method", "kpm"))
    result.arrays["energy"] = energy
    result.arrays["dos"] = np.asarray(dos).reshape(_n_kpoint(options), -1)


def _ldos(calc, task, options, result):
    import numpy as np

    energies = build_energies(task["energies"])
    out = calc.compute_rs_ldos(
        energies, task["eta"], spacing=task["spacing"], box_edge=task["box_edge"],
        n_samples=task["n_samples"], block=task["block"], seed=task["seed"],
        fragment_core=task["fragment_core"], fragment_buffer=task["fragment_buffer"],
        subspace_cv=task["subspace_cv"], subspace_filters=task["subspace_filters"],
        local_capture=task["local_capture"], local_pack=task["local_pack"],
        local_pack_replicas=task["local_pack_replicas"])

    # Compact and per-rank: 'point_index' says where this rank's values sit in
    # the flat grid, so the pieces reassemble by scattering, never by summing.
    # The driver also offers a dense 'grid' on serial runs; it is deliberately
    # not written, because a format that changes shape with the rank count is
    # worse than one that does not.
    result.distributed = True
    result.arrays["energy"] = np.asarray(energies)
    result.arrays["shape"] = np.asarray(out["shape"], dtype=np.int64)
    result.arrays["point_index"] = out["point_index"]
    result.arrays["ldos"] = out["ldos"]
    result.scalars["n_point"] = int(out["n_point"])
    result.scalars["filter_degree"] = int(out["filter_degree"])
    result.scalars["n_states"] = int(out["n_states"])
    result.scalars["sampler_residual"] = float(out["sampler_residual"])
    if int(out["n_fragments"]):
        result.scalars["n_fragments"] = int(out["n_fragments"])
        result.scalars["correction_rms"] = float(np.max(out["correction_rms"]))
        result.arrays["control_beta"] = out["control_beta"]
    if len(out["subspace_capture"]):
        # Keep the per-energy ranks beside the per-energy capture fractions.
        # A single minimum is useful in a short summary, but it is not enough
        # for a reproducible rank/capture study when one task contains several
        # spectral windows.
        result.scalars["subspace_rank"] = int(np.min(out["subspace_rank"]))
        result.scalars["subspace_rank_min"] = int(np.min(out["subspace_rank"]))
        result.scalars["subspace_rank_max"] = int(np.max(out["subspace_rank"]))
        result.arrays["subspace_capture"] = out["subspace_capture"]
        result.arrays["subspace_rank"] = out["subspace_rank"]
    if int(out["local_regions"]):
        # The local-capture pass actually ran for some energy: report which
        # energies it served, its measured packing error bar, and the audited
        # same-color distance of the packing geometry.
        result.arrays["local_route"] = out["local_route"]
        result.arrays["local_packing_se"] = out["local_packing_se"]
        result.scalars["local_regions"] = int(out["local_regions"])
        result.scalars["local_colors"] = int(out["local_colors"])
        result.scalars["min_same_color_distance"] = float(
            out["min_same_color_distance"])


def _pdos(calc, task, options, result):
    import numpy as np

    out = calc.compute_pdos(task["dE"], projection=task["projection"])
    result.arrays["energy"] = out["energy"]
    result.arrays["pdos"] = out["pdos"]
    result.arrays["total"] = out["total"]
    result.scalars["n_group"] = int(out["n_group"])
    result.scalars["population"] = out["population"]
    # Which atom type and orbital slot each group is; empty for projection
    # "atom", whose group id is already the atom's input index.
    if len(out["group_type"]):
        result.arrays["group_type"] = out["group_type"]
        result.arrays["group_slot"] = out["group_slot"]


def _density(calc, task, options, result):
    import numpy as np

    out = calc.compute_rs_density(
        task["mu"], task["temp"], spin_degeneracy=task["spin_degeneracy"],
        spacing=task["spacing"], box_edge=task["box_edge"],
        n_samples=task["n_samples"], block=task["block"], seed=task["seed"],
        fragment_core=task["fragment_core"], fragment_buffer=task["fragment_buffer"],
        energy_windows=task["energy_windows"],
        fragment_coefficient=task["fragment_coefficient"])

    # Same compact per-rank layout as the ldos task, for the same reason.
    result.distributed = True
    result.arrays["shape"] = np.asarray(out["shape"], dtype=np.int64)
    result.arrays["point_index"] = out["point_index"]
    result.arrays["density"] = out["density"]
    result.scalars["n_point"] = int(out["n_point"])
    result.scalars["filter_degree"] = int(out["filter_degree"])
    result.scalars["n_states"] = int(out["n_states"])
    result.scalars["sampler_residual"] = float(out["sampler_residual"])
    result.scalars["electrons_on_grid"] = float(out["electrons_on_grid"])
    if int(out["n_fragments"]):
        result.scalars["n_fragments"] = int(out["n_fragments"])
        result.scalars["correction_rms"] = float(out["correction_rms"])
        result.scalars["control_beta"] = float(out["control_beta"])
    if int(out["n_windows"]) > 1:
        result.scalars["n_windows"] = int(out["n_windows"])


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
        task["n_steps"], task["dt_fs"], task["n_kpm"], axis=task["axis"],
        method=task["method"], n_kpm_compute=task["n_kpm_compute"])
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
    "pdos": _pdos,
    "ldos": _ldos,
    "density": _density,
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


def write_single_result(result: Result, resolved: dict, base_dir: str, rank: int = 0,
                        n_rank: int = 1, log=None) -> None:
    """Writes one task's arrays now. Called as each task finishes, so a
    scheduler killing the job mid-run costs only the task in flight.

    EVERY rank calls this. An ordinary result is written by rank 0 alone, as
    before; a distributed one (result.distributed) is written by every rank
    into its own file, with no gather and no collective -- which is the point,
    since the arrays it holds are exactly the piece of the grid that rank
    owns. Idempotent with write_results below, which skips results already on
    disk.
    """
    import numpy as np

    output = resolved.get("output", {})
    fmt = output.get("format", "npz")
    if not result.arrays or fmt == "none":
        return
    if rank != 0 and not result.distributed:
        return
    directory = resolve_output_dir(resolved, base_dir)
    # Every rank may create it; exist_ok makes the race benign.
    os.makedirs(directory, exist_ok=True)
    prefix = output.get("prefix", "")

    # Rank-tagged names, zero-padded so a directory listing sorts in rank
    # order. The width follows the launch, so 1000 ranks do not silently
    # collide with a 5-digit assumption.
    if result.distributed:
        width = max(5, len(str(max(n_rank - 1, 0))))
        stem = "{}{}.rank{:0{}d}".format(prefix, result.label, rank, width)
    else:
        stem = "{}{}".format(prefix, result.label)

    if fmt == "npz":
        path = os.path.join(directory, stem + ".npz")
        np.savez_compressed(path, **{name: np.asarray(value)
                                     for name, value in result.arrays.items()})
    else:
        path = os.path.join(directory, stem + ".json")
        with open(path, "w") as handle:
            json.dump({name: np.asarray(value).tolist()
                       for name, value in result.arrays.items()}, handle)
    result.file = os.path.relpath(path, directory)
    if result.distributed:
        # One line for the set, not one per rank: `log` is rank-0 only, so
        # the other ranks' writes would be invisible rather than merely quiet,
        # and a reader needs the pattern, not rank 0's own file name.
        result.file = "{}{}.rank*{}".format(prefix, result.label,
                                            ".npz" if fmt == "npz" else ".json")
        if log:
            log("  wrote {} ({} rank{})".format(
                result.file, n_rank, "" if n_rank == 1 else "s"))
    elif log:
        log("  wrote {}".format(_readable(path, base_dir)))


def write_results(results: List[Result], resolved: dict, setup: dict,
                  base_dir: str, rank: int = 0, n_rank: int = 1, log=None) -> str:
    """Writes every result not already written, and the summary.

    EVERY rank calls this, because a distributed result is written by all of
    them; the summary is rank 0's alone. Results with .file set were written
    incrementally by write_single_result and are not rewritten.
    """
    import numpy as np

    output = resolved.get("output", {})
    directory = resolve_output_dir(resolved, base_dir)
    if rank == 0:
        os.makedirs(directory, exist_ok=True)
    prefix = output.get("prefix", "")

    for result in results:
        if not result.file:
            write_single_result(result, resolved, base_dir, rank=rank, n_rank=n_rank,
                                log=log)

    summary_path = ""
    if rank != 0:
        return summary_path
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
                    # For a distributed result these shapes are RANK 0's
                    # share, not the whole array, and saying so is the
                    # difference between a summary and a wrong summary. The
                    # global size is in scalars (n_point for the LDOS grid).
                    "distributed": result.distributed,
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
