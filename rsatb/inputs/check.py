"""Validation of a whole input: schema, paths, and cross-section consistency.

The schema in `spec.py` answers "is each field well formed". This module
answers the questions that need two sections at once -- does the model supply
the operator this task needs, does the file named actually exist, does the
k-point set contain the Gamma point an observable is defined at.

Nothing here imports the compiled extension, so every check runs on a machine
that cannot run the solver. That is the point: an input can be checked before
it is queued.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import spec
from .nodes import Issue, Node, Obj, WARNING, join_path

# What each model family can build, decided from the input alone.
#
# This restates the operator-source table each C++ model declares in its own
# file (model/operator_source.hpp, and one `operator_sources()` per family). It
# has to be restated rather than read, because `rsatb check` must run where the
# extension cannot be built -- that is the whole point of checking an input
# before queueing it.
#
# The restatement is kept honest by a test rather than by discipline:
# test_json_input.py::test_capabilities_match_the_compiled_models builds one
# model of every family and asserts these answers equal the compiled
# `capabilities()`. If a family's table changes in C++ and not here, that test
# fails.
_CAPABILITY_DOC = {
    "sktb": "an SKTB model is orthogonal unless 'overlap' is given, and an "
            "orthogonal Slater-Koster model supplies velocity and position for free",
    "lcao": "LCAO is always non-orthogonal; the velocity is the commutator "
            "built from H, S and the dipole integrals, so it needs no "
            "pseudopotential files",
    "dftb": "DFTB carries both H and S, and neither velocity nor position is "
            "available for a non-orthogonal Slater-Koster model yet",
    "tb_file": "a file-backed tight-binding model is orthogonal, so velocity and "
               "position follow from H(R)",
}

# Which tasks need which operator, and which need the Gamma point. msd runs
# on the commutator velocity alone (both its kernels do), so it sits with
# the other velocity consumers.
_NEEDS_VELOCITY = ("ac", "dc", "dc_spectrum", "quantum_metric", "msd")
_NEEDS_GAMMA_ONLY = ("ldos",)
_NEEDS_GAMMA = ("msd",)
_NEEDS_ORBITAL_FILES = ("ldos",)


def capabilities(model: dict) -> Dict[str, bool]:
    """Returns the operators a model can supply, decided from the input alone."""
    family = model.get("type")
    if family == "sktb":
        orthogonal = "overlap" not in model
        return {"overlap": not orthogonal, "velocity": orthogonal, "position": orthogonal}
    if family == "lcao":
        # The velocity is the section-4 commutator, assembled from the images
        # and the orbital dipole integrals -- available unconditionally.
        return {"overlap": True, "velocity": True, "position": True}
    if family == "dftb":
        return {"overlap": True, "velocity": False, "position": False}
    if family == "tb_file":
        return {"overlap": False, "velocity": True, "position": True}
    return {"overlap": False, "velocity": False, "position": False}


# --- path handling ---------------------------------------------------------


def map_paths(node: Node, value: Any, transform) -> Any:
    """Rewrites every value the schema declares to be a path.

    Walking the schema rather than a hand-kept list of keys means a new path
    field is picked up by declaring its type, with nothing else to update.
    """
    from .nodes import Array, Choice, Mapping, Scalar, Union

    if isinstance(node, Scalar):
        if node.type_name == "path" and isinstance(value, str) and value:
            return transform(value)
        return value
    if isinstance(node, Array) and isinstance(value, list):
        return [map_paths(node.item, entry, transform) for entry in value]
    if isinstance(node, Mapping) and isinstance(value, dict):
        return {key: map_paths(node.value, entry, transform) for key, entry in value.items()}
    if isinstance(node, Obj) and isinstance(value, dict):
        return {key: map_paths(node.fields[key].node, entry, transform)
                if key in node.fields else entry
                for key, entry in value.items()}
    if isinstance(node, Union) and isinstance(value, dict):
        variant = node.variants.get(value.get(node.tag))
        if variant is None:
            return value
        mapped = map_paths(variant, value, transform)
        mapped[node.tag] = value[node.tag]
        return mapped
    if isinstance(node, Choice):
        from .nodes import _kind
        option = node.options.get(_kind(value))
        if option is not None:
            return map_paths(option, value, transform)
    return value


def _existing_paths(node: Node, value: Any, path: str, out: List[Tuple[str, str]]) -> None:
    """Collects (document path, filesystem path) for every declared path field."""
    from .nodes import Array, Choice, Mapping, Scalar, Union, _kind

    if isinstance(node, Scalar):
        if node.type_name == "path" and isinstance(value, str) and value:
            out.append((path, value))
    elif isinstance(node, Array) and isinstance(value, list):
        for index, entry in enumerate(value):
            _existing_paths(node.item, entry, join_path(path, index), out)
    elif isinstance(node, Mapping) and isinstance(value, dict):
        for key, entry in value.items():
            _existing_paths(node.value, entry, join_path(path, key), out)
    elif isinstance(node, Obj) and isinstance(value, dict):
        for key, entry in value.items():
            if key in node.fields:
                _existing_paths(node.fields[key].node, entry, join_path(path, key), out)
    elif isinstance(node, Union) and isinstance(value, dict):
        variant = node.variants.get(value.get(node.tag))
        if variant is not None:
            _existing_paths(variant, value, path, out)
    elif isinstance(node, Choice):
        option = node.options.get(_kind(value))
        if option is not None:
            _existing_paths(option, value, path, out)


# Paths that name something to be written, not read.
_OUTPUT_PATHS = ("output.dir", "model.log_file")


def check_files_exist(document: dict) -> List[Issue]:
    """Reports inputs that name a file or directory that is not there."""
    found: List[Tuple[str, str]] = []
    _existing_paths(spec.ROOT, document, "", found)

    issues: List[Issue] = []
    for where, path in found:
        if where in _OUTPUT_PATHS:
            continue
        if where.endswith("orbitals.dir") or where.endswith("pseudopotentials.dir"):
            continue  # checked together with the names below
        if os.path.exists(path):
            continue
        issues.append(Issue(where, "no such file or directory: {}".format(path),
                            hint="paths are resolved against the input file's directory"))

    model = document.get("model", {})
    for group in ("orbitals", "pseudopotentials"):
        block = model.get(group)
        if not isinstance(block, dict):
            continue
        directory = block.get("dir", "")
        for index, name in enumerate(block.get("names", [])):
            candidate = os.path.join(directory, name) if directory else name
            if not os.path.exists(candidate):
                issues.append(Issue(
                    "model.{}.names[{}]".format(group, index),
                    "no such file: {}".format(candidate),
                    hint="names are resolved against model.{}.dir".format(group)))
    return issues


# --- cross-section checks --------------------------------------------------


def _kpoint_list(kpoints) -> List[List[float]]:
    """Expands the k-point section the way the run will.

    A mesh or a line is checked against what it actually generates rather than
    against the shorthand, so "npar exceeds the k-point count" and "this
    observable needs Gamma" mean the same thing however the set was written.
    """
    from .build import build_kpoints
    return build_kpoints(kpoints)


def _has_gamma(kpoints) -> bool:
    return any(all(abs(component) < 1e-12 for component in k)
               for k in _kpoint_list(kpoints))


def check_consistency(document: dict) -> List[Issue]:
    """Reports inputs whose sections are each valid but disagree with each other."""
    issues: List[Issue] = []
    model = document.get("model", {})
    run = document.get("run", {})
    tasks = document.get("tasks", [])
    caps = capabilities(model)
    family = model.get("type", "")
    why = _CAPABILITY_DOC.get(family, "")

    kpoints = run.get("kpoints", [[0.0, 0.0, 0.0]])
    n_kpoint = len(_kpoint_list(kpoints))

    # File-backed repetition needs a declared primitive (i,j,R) convention.
    # DeePTB is currently the only registered format and its convention is
    # mapped during operator construction. Future formats remain opt-in here.
    supercell = document.get("structure", {}).get("supercell", [1, 1, 1])
    if any(n > 1 for n in supercell) and family in ("lcao", "tb_file"):
        file_format = document.get("model", {}).get("hamiltonian", {}).get(
            "format", "deeptb")
        if file_format != "deeptb":
            issues.append(Issue(
                "structure.supercell",
                "the '{}' Hamiltonian format has not defined how primitive (i,j,R) "
                "blocks map into a supercell".format(file_format),
                hint="use the DeePTB HDF5 format, or supply a Hamiltonian already "
                     "indexed for the expanded structure"))

    labels = {}
    for index, task in enumerate(tasks):
        kind = task.get("type", "")
        where = "tasks[{}]".format(index)
        label = task.get("label") or kind
        if label in labels:
            issues.append(Issue(
                join_path(where, "label"),
                "result name {!r} is already used by tasks[{}]".format(label, labels[label]),
                hint="give one of them an explicit 'label'"))
        labels[label] = index

        if kind in _NEEDS_VELOCITY and not caps["velocity"]:
            issues.append(Issue(
                where, "'{}' needs the velocity operator, which the '{}' model with "
                       "these inputs does not supply".format(kind, family),
                hint=why))
        if kind in _NEEDS_ORBITAL_FILES and family != "lcao":
            issues.append(Issue(
                where, "'{}' needs the radial orbital files, which only an LCAO model "
                       "carries".format(kind),
                hint="a Slater-Koster or file-backed model describes its basis by label "
                     "only, so there is nothing to evaluate in real space"))
        if kind in _NEEDS_GAMMA_ONLY:
            if n_kpoint != 1:
                issues.append(Issue(
                    where, "'{}' currently requires a single Gamma point, but "
                           "run.kpoints has {}".format(kind, n_kpoint)))
            elif not _has_gamma(kpoints):
                issues.append(Issue(where, "'{}' requires the Gamma point".format(kind)))
        elif kind in _NEEDS_GAMMA and not _has_gamma(kpoints):
            issues.append(Issue(
                where, "'{}' is evaluated at Gamma, which is not in "
                       "run.kpoints".format(kind)))

    npar = run.get("npar", 1)
    if npar > n_kpoint:
        issues.append(Issue(
            "run.npar",
            "npar is {} but there are only {} k-points, so {} sub-communicators have "
            "nothing to do".format(npar, n_kpoint, npar - n_kpoint),
            severity=WARNING))

    issues.extend(_check_radial_tails(model))
    return issues


def _check_radial_tails(model: dict) -> List[Issue]:
    """Warns when a tabulated channel is held constant out to its cutoff.

    A spline clamps to its last tabulated value between the end of the table
    and r_cut. That is silent, and it is almost never what was meant.
    """
    issues: List[Issue] = []
    for matrix in ("hamiltonian", "overlap"):
        block = model.get(matrix)
        if not isinstance(block, dict):
            continue
        for pair, channels in block.get("pairs", {}).items():
            for channel, radial in channels.items():
                if not isinstance(radial, dict) or radial.get("kind") != "spline":
                    continue
                table = radial.get("r")
                values = radial.get("values")
                if not table or not values:
                    continue
                r_cut = radial.get("r_cut", 0.0)
                if r_cut > table[-1] + 1e-9 and abs(values[-1]) > 1e-6:
                    issues.append(Issue(
                        "model.{}.pairs.{}.{}".format(matrix, pair, channel),
                        "the table ends at r={} but r_cut is {}; the value is held at "
                        "{} over the gap".format(table[-1], r_cut, values[-1]),
                        severity=WARNING,
                        hint="extend the table to r_cut, or lower r_cut to the table end"))
    return issues


# --- public entry points ---------------------------------------------------


def load(path: str) -> dict:
    """Reads a JSON input file, with parse errors reported by line and column."""
    with open(path, "r") as handle:
        text = handle.read()
    try:
        document = json.loads(text)
    except ValueError as error:
        raise ValueError("{}: {}".format(path, error))
    if not isinstance(document, dict):
        raise ValueError("{}: the top level of an input file must be an object".format(path))
    return document


def resolve(document: dict, base_dir: Optional[str] = None,
            check_paths: bool = True) -> Tuple[dict, List[Issue]]:
    """Validates a document and returns it with defaults and paths resolved.

    Args:
        document: The parsed JSON input.
        base_dir: Directory relative paths are resolved against; usually the
            directory holding the input file. None leaves paths untouched.
        check_paths: Whether to report inputs naming a file that is not there.

    Returns:
        The resolved document, and every issue found. The document is usable
        only when no issue has severity 'error'.
    """
    resolved, issues = spec.ROOT.check(document, "")
    if any(issue.is_error for issue in issues):
        return resolved, issues

    if base_dir:
        resolved = map_paths(
            spec.ROOT, resolved,
            lambda value: value if os.path.isabs(value)
            else os.path.normpath(os.path.join(base_dir, value)))

    if check_paths:
        issues.extend(check_files_exist(resolved))
    issues.extend(check_consistency(resolved))
    return resolved, issues


def load_and_resolve(path: str, check_paths: bool = True) -> Tuple[dict, List[Issue]]:
    """Reads and validates an input file, resolving paths against its directory."""
    document = load(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    return resolve(document, base_dir, check_paths=check_paths)


def summarize(resolved: dict) -> List[str]:
    """Returns the one-screen description of what an input will do."""
    model = resolved.get("model", {})
    run = resolved.get("run", {})
    caps = capabilities(model)
    lines = []

    structure = resolved.get("structure", {})
    if "file" in structure:
        where = structure["file"]
    else:
        where = "{} atoms, inline".format(len(structure.get("positions", [])))
    tiling = structure.get("supercell", [1, 1, 1])
    if any(n > 1 for n in tiling):
        where += ", tiled {}".format("x".join(str(n) for n in tiling))
    lines.append("structure     {}".format(where))

    lines.append("model         {}{}".format(
        model.get("type", "?"),
        "  " + ", ".join(sorted(model.get("basis", {}))) if model.get("basis") else ""))
    lines.append("capabilities  {}{}{}".format(
        "non-orthogonal" if caps["overlap"] else "orthogonal",
        ", velocity" if caps["velocity"] else "",
        ", position" if caps["position"] else ""))

    kpoints = run.get("kpoints", [])
    if isinstance(kpoints, list):
        written = "written out"
    elif "mesh" in kpoints:
        written = "{} mesh".format("x".join(str(n) for n in kpoints["mesh"]))
    else:
        written = "line through {} vertices".format(len(kpoints["line"]))
    lines.append("k-points      {} ({})".format(len(_kpoint_list(kpoints)), written))

    bounds = run.get("bounds", {})
    lines.append("bounds        {}".format(
        "fixed {}".format(bounds.get("energy")) if bounds.get("mode") == "fixed"
        else "estimated at setup"))
    lines.append("parallel      npar={}, nsamples={}".format(
        run.get("npar", 1), run.get("nsamples", 1)))

    tasks = resolved.get("tasks", [])
    if tasks:
        lines.append("tasks         {}".format(", ".join(
            task.get("label") or task.get("type", "?") for task in tasks)))
    else:
        lines.append("tasks         none -- sets up and reports the spectral window")
    return lines
