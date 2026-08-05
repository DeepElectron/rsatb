"""Turns a resolved JSON input into the objects the library is constructed from.

Every function here maps one JSON section onto one constructor argument. There
is no normalization layer and no shape guessing: the schema has already decided
what each value means, so this module only translates.
"""

from __future__ import annotations

import os
from typing import Any, List, Sequence


def _atomic_number(symbol: str) -> int:
    """Returns the atomic number of a chemical symbol.

    The table is the C++ one in vbcsr::atomic, so the reader, the model layer
    and this share a single list of elements. `capitalize` first because a JSON
    input may well spell it "si", where the C++ side matches exactly.
    """
    from .. import atomic_number
    return atomic_number(symbol.strip().capitalize())


# --- structure -------------------------------------------------------------


def tile(positions: Sequence[Sequence[float]], numbers: Sequence[int],
         cell: Sequence[Sequence[float]], supercell: Sequence[int]):
    """Repeats a cell, leaving the original coordinates of every image intact.

    Images are laid down in the order a1, then a2, then a3, with the original
    atom order inside each image -- the same convention ASE's `repeat` uses.
    """
    nx, ny, nz = (int(n) for n in supercell)
    tiled_positions: List[List[float]] = []
    tiled_numbers: List[int] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                offset = [i * cell[0][axis] + j * cell[1][axis] + k * cell[2][axis]
                          for axis in range(3)]
                for position, number in zip(positions, numbers):
                    tiled_positions.append([position[axis] + offset[axis] for axis in range(3)])
                    tiled_numbers.append(number)
    tiled_cell = [[cell[0][axis] * nx for axis in range(3)],
                  [cell[1][axis] * ny for axis in range(3)],
                  [cell[2][axis] * nz for axis in range(3)]]
    return tiled_positions, tiled_numbers, tiled_cell


def build_structure(config: dict):
    """Builds a Structure from the 'structure' section."""
    from .. import Structure

    supercell = config.get("supercell", [1, 1, 1])
    tiled = any(n > 1 for n in supercell)

    if "file" in config:
        structure = Structure.from_file(config["file"], config.get("format", ""))
        if not tiled:
            return structure
        positions = [list(structure.pos[3 * i:3 * i + 3]) for i in range(structure.n_atom)]
        numbers = list(structure.z)
        cell = [list(structure.cell[3 * i:3 * i + 3]) for i in range(3)]
        pbc = list(structure.pbc)
    else:
        positions = [list(p) for p in config["positions"]]
        numbers = (list(config["numbers"]) if "numbers" in config
                   else [_atomic_number(s) for s in config["symbols"]])
        cell = [list(row) for row in config["cell"]]
        pbc = list(config.get("pbc", [True, True, True]))

    if tiled:
        positions, numbers, cell = tile(positions, numbers, cell, supercell)

    return Structure.from_arrays(
        pos=[component for position in positions for component in position],
        z=numbers,
        cell=[component for row in cell for component in row],
        pbc=pbc,
    )


# --- k-points --------------------------------------------------------------


def build_kpoints(config) -> List[List[float]]:
    """Expands the 'run.kpoints' section into an explicit fractional list."""
    if isinstance(config, list):
        return [list(k) for k in config]

    if "mesh" in config:
        nx, ny, nz = config["mesh"]
        shift = config.get("shift", [0.0, 0.0, 0.0])
        return [[i / nx + shift[0], j / ny + shift[1], k / nz + shift[2]]
                for i in range(nx) for j in range(ny) for k in range(nz)]

    vertices = config["line"]
    per_segment = config.get("points_per_segment", 11)
    points: List[List[float]] = []
    for start, end in zip(vertices, vertices[1:]):
        for index in range(per_segment):
            fraction = index / (per_segment - 1)
            point = [start[axis] + fraction * (end[axis] - start[axis]) for axis in range(3)]
            if points and index == 0:
                continue  # the shared vertex is already there
            points.append(point)
    return points


def build_energies(config) -> List[float]:
    """Expands an energy grid section into an explicit list."""
    if isinstance(config, list):
        return [float(e) for e in config]
    low, high, count = config["min"], config["max"], config["n"]
    step = (high - low) / (count - 1)
    return [low + step * index for index in range(count)]


# --- models ----------------------------------------------------------------


def _read_two_columns(path: str):
    """Reads a two-column 'distance value' text file, '#' starting a comment."""
    distances, values = [], []
    with open(path, "r") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(
                    "{}: expected two columns per line, got {!r}".format(path, line))
            distances.append(float(fields[0]))
            values.append(float(fields[1]))
    if len(distances) < 2:
        raise ValueError("{}: a spline needs at least two tabulated points".format(path))
    return distances, values


def build_radial(config: dict):
    """Builds one SkRadialFunction from a radial section."""
    from .. import sk_exp_poly, sk_spline

    kind = config["kind"]
    if kind == "constant":
        r_cut = config["r_cut"]
        value = config["value"]
        return sk_spline([0.0, r_cut], [value, value], r_cut)
    if kind == "spline":
        if "file" in config:
            distances, values = _read_two_columns(config["file"])
        else:
            distances, values = config["r"], config["values"]
        return sk_spline(list(distances), list(values), config["r_cut"])
    if kind == "exp_poly":
        return sk_exp_poly(list(config["coeffs"]), config["r_cut"],
                           config.get("lambda", 0.0), config.get("r_on", 0.0))
    raise ValueError("Unsupported radial function kind: {!r}".format(kind))


def settings(mapping: dict) -> dict:
    """Drops the underscore-prefixed keys the schema carries through as notes."""
    return {key: value for key, value in mapping.items() if not key.startswith("_")}


def build_sk_parameters(config: dict):
    """Builds one SkParameters -- the onsite terms and pair channels of H or S."""
    from .. import SkParameters

    pairs = {}
    for key, channels in settings(config.get("pairs", {})).items():
        row, column = key.split("-", 1)
        pairs[(row, column)] = {name: build_radial(radial)
                                for name, radial in settings(channels).items()}
    onsite = {species: settings(orbitals)
              for species, orbitals in settings(config.get("onsite", {})).items()}
    return SkParameters(onsite=onsite, pairs=pairs)


def build_model(config: dict):
    """Builds the model named by the 'model' section."""
    from .. import (DftbModel, HamiltonianFile, LcaoModel, OrbitalFiles,
                    PseudopotentialFiles, SktbModel, TightBindingFileModel)

    family = config["type"]
    spin = config.get("spin", "none")

    if family == "sktb":
        return SktbModel(
            basis=settings(config["basis"]),
            hamiltonian=build_sk_parameters(config["hamiltonian"]),
            overlap=(build_sk_parameters(config["overlap"])
                     if "overlap" in config else None),
            spin=spin,
        )

    if family == "lcao":
        orbitals = config["orbitals"]
        pseudopotentials = config.get("pseudopotentials", {})
        return LcaoModel(
            hamiltonian=HamiltonianFile(config["hamiltonian"]["path"],
                                        format=config["hamiltonian"]["format"]),
            orbitals=OrbitalFiles(list(orbitals["names"]), dir=orbitals.get("dir", "")),
            pseudopotentials=PseudopotentialFiles(
                list(pseudopotentials.get("names", [])),
                dir=pseudopotentials.get("dir", "")),
            spin=spin,
            log_file=config.get("log_file", "op2c.log"),
        )

    if family == "dftb":
        return DftbModel.from_skf_directory(
            skf_dir=config["skf_dir"],
            basis=settings(config["basis"]),
            heteronuclear=config.get("heteronuclear", "reject_ambiguous"),
            spin=spin,
        )

    if family == "tb_file":
        return TightBindingFileModel(
            hamiltonian=HamiltonianFile(config["hamiltonian"]["path"],
                                        format=config["hamiltonian"]["format"]),
            type_symbols=list(config["type_symbols"]),
            type_norb=list(config["type_norb"]),
            cutoff_radius=list(config["cutoff_radius"]),
            spin=spin,
        )

    raise ValueError("Unsupported model type: {!r}".format(family))


# --- run options -----------------------------------------------------------


def build_options(config: dict, model) -> Any:
    """Builds SetupOptions from the 'run' section.

    `preconditioner.target` defaults to 'auto', which is resolved here from the
    model's own capabilities: there is nothing to precondition in an orthogonal
    basis, and a non-orthogonal one always wants it.
    """
    from .. import PreconditionerOptions, SetupOptions

    precondition = dict(config.get("preconditioner", {}))
    target = precondition.pop("target", "auto")
    if target == "auto":
        target = "overlap" if model.capabilities.overlap else "none"

    bounds = config.get("bounds", {})
    fixed = bounds.get("mode") == "fixed"

    return SetupOptions(
        kpoints=build_kpoints(config.get("kpoints", [[0.0, 0.0, 0.0]])),
        nsamples=config.get("nsamples", 1),
        compute_bounds=not fixed,
        energy_bounds=list(bounds["energy"]) if fixed else None,
        bounds_zero_tol=bounds.get("zero_tol", 1e-12),
        bounds_max_matvecs=bounds.get("max_matvecs", 10000),
        preconditioner=PreconditionerOptions(target=target, **precondition),
    )


def build_calculation(resolved: dict):
    """Builds the ready-to-compute object from a resolved input.

    Returns:
        A tuple of (RSATB, model, options). The model is returned because the
        velocity and position operators are built lazily through it.
    """
    from .. import RSATB

    model = build_model(resolved["model"])
    options = build_options(resolved.get("run", {}), model)
    structure = build_structure(resolved["structure"])
    run = resolved.get("run", {})
    # log_level is the real control; verbose is the pre-log_level spelling of
    # "debug" and still wins when it is set, so old inputs keep their behaviour.
    # (The CLI's --log-level clears run.verbose before this point, which is how
    # a flag on the command line beats a "verbose": true left in the file.)
    level = "debug" if run.get("verbose", False) else run.get("log_level", "summary")
    import rsatb  # ensures rsatb/build is on sys.path for the extension
    import _rsatb
    _rsatb.set_log_level(level)
    # The constructor's bool is derived from the level, not read from the file
    # again: it feeds the parts that predate the level (PRIMME's printLevel,
    # vbcsr's phase timers), and reading run.verbose here would make
    # log_level "debug" quieter than verbose true -- two spellings of the same
    # request with different behaviour.
    calculation = RSATB(structure, model, options,
                        npar=run.get("npar", 1),
                        verbose=(level == "debug"))
    return calculation, model, options
