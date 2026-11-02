"""The RSATB JSON input schema.

This module is the single source of truth for what an input file may contain.
`rsatb check` validates against it, `rsatb show` fills its defaults, `rsatb doc`
renders it for a reader, and `rsatb schema` exports it as JSON Schema. Adding a
field here makes it work in all four.

The schema mirrors the object model the library already has -- a structure, a
model, run options, and the observables to compute -- so a JSON key maps onto a
constructor argument rather than onto a private normalization layer.
"""

from __future__ import annotations

import re
from typing import List

from .nodes import (Array, Choice, Field as F, Issue, Mapping, Node, Obj, Scalar,
                    Union, WARNING, join_path)

INPUT_VERSION = 1

# --- small reusable scalars ------------------------------------------------


def _num(doc, **kwargs):
    return Scalar(doc, "number", **kwargs)


def _int(doc, **kwargs):
    return Scalar(doc, "integer", **kwargs)


def _str(doc, **kwargs):
    return Scalar(doc, "string", **kwargs)


def _path(doc, **kwargs):
    return Scalar(doc, "path", **kwargs)


def _bool(doc, **kwargs):
    return Scalar(doc, "boolean", **kwargs)


VEC3 = Array(_num("Cartesian or fractional component."), "Three components.", length=3)

SPIN = _str(
    "How spin is represented in the basis. 'none': one state per orbital. "
    "'collinear': two independent spin blocks. 'noncollinear' and 'spinorbit': "
    "every orbital carries both spin components, which doubles the basis.",
    choices=("none", "collinear", "noncollinear", "spinorbit"))

AXIS = _str(
    "The cell axes that define the volume the conductivity is divided by. "
    "'abc': the cell volume. A pair ('ab', 'bc', 'ca'): the area of that "
    "lattice plane, for 2D systems with vacuum in the cell. A single axis: "
    "the length of that lattice vector, for 1D systems. All six Cartesian "
    "components of the tensor are computed in each case.",
    choices=("abc", "ab", "bc", "ca", "a", "b", "c"))

HAMILTONIAN_FILE = Obj(
    "A file holding a real-space Hamiltonian H(R), and its format.",
    {
        "path": F(_path("Path to the Hamiltonian file, relative to the input file's "
                        "directory unless absolute."), required=True),
        "format": F(_str("File format. 'deeptb': DeePTB HDF5, with one group '0' "
                         "holding datasets named 'i_j_Rx_Ry_Rz'. 'abacus': the sparse "
                         "H(R) that ABACUS writes with out_mat_hs2 (hrs1_nao.csr, or "
                         "data-HR-sparse_SPIN0.csr in older versions), as text or "
                         "binary. 'nextham': NextHam's HR_atom_pairs.h5. The 'abacus' "
                         "and 'nextham' formats need an 'lcao' model, a structure that "
                         "lists the atoms in the order of the ABACUS calculation, and "
                         "spin 'spinorbit' for a spin-orbit (nspin = 4) calculation.",
                         choices=("deeptb", "abacus", "nextham")), default="deeptb"),
    })


# --- structure -------------------------------------------------------------


def _structure_checks(value: dict, path: str) -> List[Issue]:
    issues: List[Issue] = []
    if "positions" in value:
        n_atom = len(value["positions"])
        if "cell" not in value:
            issues.append(Issue(path, "'cell' is required when positions are given inline"))
        if "symbols" not in value and "numbers" not in value:
            issues.append(Issue(path, "give either 'symbols' or 'numbers' alongside 'positions'"))
        for key in ("symbols", "numbers"):
            if key in value and len(value[key]) != n_atom:
                issues.append(Issue(
                    join_path(path, key),
                    "has {} entries but there are {} positions".format(
                        len(value[key]), n_atom)))
        if "symbols" in value and "numbers" in value:
            issues.append(Issue(path, "give either 'symbols' or 'numbers', not both"))
    else:
        for key in ("cell", "symbols", "numbers"):
            if key in value:
                issues.append(Issue(
                    join_path(path, key),
                    "only applies to an inline structure; it is read from 'file' here"))
    if any(n < 1 for n in value.get("supercell", [1, 1, 1])):
        issues.append(Issue(join_path(path, "supercell"), "tiling factors must be >= 1"))
    return issues


STRUCTURE = Obj(
    "The atomic structure, read from a file or written in the input.\n\n"
    "Atoms are used in the given order and at the given positions; they are "
    "not sorted, renumbered or wrapped into the cell, because Hamiltonian "
    "files refer to atoms by their order and position. Atoms may lie outside "
    "the cell.",
    {
        "file": F(_path("Structure file: POSCAR (.vasp), .xyz or .cif. The format "
                        "is taken from the extension unless 'format' is given.")),
        "format": F(_str("Format of 'file'. Empty: taken from the extension."),
                    default=""),
        "positions": F(Array(VEC3, "Cartesian positions in Angstrom, one triple per atom.")),
        "symbols": F(Array(_str("Chemical symbol."), "Chemical symbol per atom, e.g. "
                                                     "[\"C\", \"C\"].")),
        "numbers": F(Array(_int("Atomic number."), "Atomic number per atom; an "
                                                   "alternative to 'symbols'.")),
        "cell": F(Array(VEC3, "Lattice vectors in Angstrom, as three rows.", length=3)),
        "pbc": F(Array(_bool("Periodic along this lattice vector."),
                       "Periodicity along each lattice vector.", length=3),
                 default=[True, True, True]),
        "supercell": F(Array(_int("Tiling factor.", minimum=1),
                             "Number of repetitions of the cell along each lattice "
                             "vector. [1, 1, 1] uses the cell as given.", length=3),
                       default=[1, 1, 1]),
    },
    require_exactly_one=[["file", "positions"]],
    checks=[_structure_checks],
    example={"file": "graphene.vasp", "supercell": [4, 4, 1]})


# --- Slater-Koster parameters ---------------------------------------------

ORBITAL_LABELS = ("s", "p", "d", "px", "py", "pz",
                  "dxy", "dyz", "dzx", "dxz", "dz2", "dx2-y2", "dx2y2")

CHANNEL_PATTERN = re.compile(r"^[spd][spd]_(sigma|pi|delta)$")

PAIR_KEY_PATTERN = re.compile(r"^([A-Za-z][a-z]?)-([A-Za-z][a-z]?)$")


def _check_pair_key(key: str, path: str) -> List[Issue]:
    if not PAIR_KEY_PATTERN.match(key):
        return [Issue(path, "{!r} is not a species pair".format(key),
                      hint="write the two species joined by a hyphen, e.g. \"C-C\" or \"C-H\"")]
    return []


def _check_channel_key(key: str, path: str) -> List[Issue]:
    if not CHANNEL_PATTERN.match(key.strip().lower()):
        return [Issue(path, "{!r} is not a Slater-Koster channel".format(key),
                      hint="shell pair plus bond type, e.g. \"ss_sigma\", \"sp_sigma\", "
                           "\"pp_sigma\", \"pp_pi\", \"pd_pi\", \"dd_delta\"")]
    return []


def _check_orbital_key(key: str, path: str) -> List[Issue]:
    if key.strip().lower().replace("dx2y2", "dx2-y2") not in ORBITAL_LABELS:
        return [Issue(path, "{!r} is not an orbital label".format(key),
                      hint="one of: " + ", ".join(ORBITAL_LABELS))]
    return []


def _radial_spline_checks(value: dict, path: str) -> List[Issue]:
    issues: List[Issue] = []
    has_inline = "r" in value and "values" in value
    if not has_inline and "file" not in value:
        issues.append(Issue(path, "a spline needs tabulated data: give 'r' and 'values', "
                                  "or 'file'"))
    if has_inline:
        if len(value["r"]) != len(value["values"]):
            issues.append(Issue(path, "'r' has {} entries and 'values' has {}".format(
                len(value["r"]), len(value["values"]))))
        if len(value["r"]) < 2:
            issues.append(Issue(path, "a spline needs at least two tabulated points"))
        if any(b <= a for a, b in zip(value["r"], value["r"][1:])):
            issues.append(Issue(join_path(path, "r"), "distances must be strictly increasing"))
        if "file" in value:
            issues.append(Issue(path, "give either 'file' or 'r' and 'values', not both"))
    return issues


def _radial_exp_poly_checks(value: dict, path: str) -> List[Issue]:
    if value.get("r_on", 0.0) > value.get("r_cut", 0.0):
        return [Issue(join_path(path, "r_on"),
                      "the taper must start inside the cutoff: 0 <= r_on <= r_cut")]
    return []


R_CUT = _num("Cutoff distance. The channel is zero beyond it. Neighbours are "
             "found up to the largest cutoff of each species pair.",
             unit="Angstrom", exclusive_minimum=0.0)

RADIAL = Union(
    "Distance dependence of one Slater-Koster channel.",
    "kind",
    {
        "constant": Obj(
            "A constant value up to the cutoff, for example a single "
            "nearest-neighbour hopping.",
            {
                "value": F(_num("Channel value.", unit="eV for H, dimensionless for S"),
                           required=True),
                "r_cut": F(R_CUT, required=True),
            },
            example={"kind": "constant", "value": -2.7, "r_cut": 1.6}),
        "spline": Obj(
            "A natural cubic spline through tabulated values.",
            {
                "r": F(Array(_num("Distance.", unit="Angstrom"),
                             "Strictly increasing distances.")),
                "values": F(Array(_num("Channel value at the matching distance."),
                                  "Channel values, one per distance.")),
                "file": F(_path("Two-column text file of distance and value; '#' starts "
                                "a comment. An alternative to inline 'r'/'values'.")),
                "r_cut": F(R_CUT, required=True),
            },
            checks=[_radial_spline_checks],
            example={"kind": "spline", "r": [1.0, 1.42, 2.0], "values": [-3.4, -2.7, -1.1],
                     "r_cut": 2.0}),
        "exp_poly": Obj(
            "An exponential times a polynomial, with a cosine taper between "
            "r_on and r_cut: value(r) = P(r) * exp(-lambda * r) * taper(r).",
            {
                "coeffs": F(Array(_num("Polynomial coefficient."),
                                  "Polynomial coefficients in ascending order of r.",
                                  min_length=1), required=True),
                "r_cut": F(R_CUT, required=True),
                "lambda": F(_num("Exponential decay rate.", unit="1/Angstrom", minimum=0.0),
                            default=0.0),
                "r_on": F(_num("Distance at which the cosine taper starts. 0: no taper; "
                               "the value drops to zero at r_cut.", unit="Angstrom",
                               minimum=0.0),
                          default=0.0),
            },
            checks=[_radial_exp_poly_checks],
            example={"kind": "exp_poly", "coeffs": [-2.7], "lambda": 1.7,
                     "r_on": 3.0, "r_cut": 4.0}),
    },
    tag_doc="Which radial form the remaining keys describe.")

SK_PARAMETERS = Obj(
    "Onsite energies and two-centre channels of one matrix, H or S.\n\n"
    "The Hamiltonian and the overlap are given as separate objects. An sktb "
    "model without 'overlap' is orthogonal.",
    {
        "onsite": F(Mapping(
            Mapping(_num("Onsite value.", unit="eV for H, dimensionless for S"),
                    "Value per orbital of this species.", key_doc="orbital",
                    key_check=_check_orbital_key),
            "Onsite terms, per species and orbital.", key_doc="species"),
            default={}),
        "pairs": F(Mapping(
            Mapping(RADIAL, "Radial function per Slater-Koster channel.",
                    key_doc="channel", key_check=_check_channel_key),
            "Two-centre channels, keyed by species pair written as \"A-B\".",
            key_doc="species pair", key_check=_check_pair_key),
            default={}),
    },
    example={"onsite": {"C": {"pz": 0.0}},
             "pairs": {"C-C": {"pp_pi": {"kind": "constant", "value": -2.7, "r_cut": 1.6}}}})

BASIS = Mapping(
    Array(_str("Orbital or shell label."),
          "Orbitals of this species. A shell label stands for its orbitals: "
          "\"p\" is py, pz, px.",
          min_length=1),
    "The orbitals of each species. \"C\": [\"pz\"] gives one orbital per carbon "
    "atom.",
    key_doc="species")


# --- models ----------------------------------------------------------------


def _sktb_basis_crosscheck(value: dict, path: str) -> List[Issue]:
    """Every species named in the parameters must appear in the basis."""
    issues: List[Issue] = []
    species = set(value.get("basis", {}))
    for matrix in ("hamiltonian", "overlap"):
        block = value.get(matrix)
        if not isinstance(block, dict):
            continue
        for name in block.get("onsite", {}):
            if name not in species:
                issues.append(Issue(
                    join_path(join_path(join_path(path, matrix), "onsite"), name),
                    "species {!r} is not in 'basis'".format(name),
                    hint="declared species: " + ", ".join(sorted(species))))
        for key in block.get("pairs", {}):
            match = PAIR_KEY_PATTERN.match(key)
            if not match:
                continue
            for name in match.groups():
                if name not in species:
                    issues.append(Issue(
                        join_path(join_path(join_path(path, matrix), "pairs"), key),
                        "species {!r} is not in 'basis'".format(name),
                        hint="declared species: " + ", ".join(sorted(species))))
    return issues


MODEL = Union(
    "The Hamiltonian. 'type' selects one of four model types.\n\n"
    "Orbital counts and cutoff radii are read from the parameters or from the "
    "orbital files. Only 'tb_file' needs them in the input, because its file "
    "formats do not record them.",
    "type",
    {
        "sktb": Obj(
            "A Slater-Koster model with parameters written in the input.\n\n"
            "The model is orthogonal unless 'overlap' is given. An orthogonal "
            "sktb model provides the velocity and position operators, so all "
            "tasks except the real-space maps (ldos, density) are available.",
            {
                "basis": F(BASIS, required=True),
                "hamiltonian": F(SK_PARAMETERS, required=True),
                "overlap": F(SK_PARAMETERS,
                             note="Absent means an orthogonal basis."),
                "spin": F(SPIN, default="none"),
            },
            checks=[_sktb_basis_crosscheck]),
        "lcao": Obj(
            "A Hamiltonian computed by another program (DeePTB, ABACUS or "
            "NextHam), together with the numerical atomic orbitals it was "
            "computed in.\n\n"
            "The overlap matrix and the velocity operator are computed from the "
            "orbital files. Orbital counts and cutoff radii are read from the "
            ".orb files.",
            {
                "hamiltonian": F(HAMILTONIAN_FILE, required=True),
                "orbitals": F(Obj(
                    "Numerical atomic orbital files, one per species, in "
                    "order of increasing atomic number.",
                    {
                        "names": F(Array(_str("File name."), "One .orb file per species.",
                                         min_length=1), required=True),
                        "dir": F(_path("Directory containing the files."),
                                 default=""),
                    }), required=True),
                "pseudopotentials": F(Obj(
                    "Pseudopotential files, in the same order as the orbital "
                    "files. Not needed for normal use; they enable an "
                    "alternative velocity operator used for testing.",
                    {
                        "names": F(Array(_str("File name."), "One .upf file per species."),
                                   default=[]),
                        "dir": F(_path("Directory containing the files."),
                                 default=""),
                    }), default={}),
                "spin": F(SPIN, default="none"),
                "log_file": F(_path("Log file written while the orbital integrals are "
                                    "computed."),
                              default="op2c.log"),
            }),
        "dftb": Obj(
            "DFTB parameters read from a directory of .skf files.\n\n"
            "DFTB parameters include an overlap, so the model is "
            "non-orthogonal. The basis must be given, because a .skf file "
            "tabulates all channels up to d whether or not they are used.",
            {
                "skf_dir": F(_path("Directory holding the .skf files, one per ordered "
                                   "species pair (C-C.skf, C-H.skf, H-C.skf, ...)."),
                             required=True),
                "basis": F(BASIS, required=True),
                "heteronuclear": F(_str(
                    "How to read a channel between two different shells when "
                    "the bases do not determine which shell belongs to which "
                    "species. This happens only when both species have both "
                    "shells. 'reject_ambiguous' stops with an error.",
                    choices=("reject_ambiguous", "assume_lower_shell_on_row_species")),
                    default="reject_ambiguous"),
                "spin": F(SPIN, default="none"),
            }),
        "tb_file": Obj(
            "An orthogonal H(R) read from a file, such as a DeePTB "
            "Slater-Koster export.\n\n"
            "The basis must be given, because these files do not record "
            "orbital counts.",
            {
                "hamiltonian": F(HAMILTONIAN_FILE, required=True),
                "type_symbols": F(Array(_str("Chemical symbol."),
                                        "Species, in the order the file uses for atom types.",
                                        min_length=1), required=True),
                "type_norb": F(Array(_int("Orbital count.", minimum=1),
                                     "Number of orbitals of each species, including spin.",
                                     min_length=1),
                               required=True),
                "cutoff_radius": F(Array(_num("Cutoff.", unit="Angstrom", exclusive_minimum=0.0),
                                         "Cutoff radius of each species. Two atoms are "
                                         "neighbours if their distance is at most "
                                         "r_i + r_j.",
                                         min_length=1), required=True),
                "spin": F(SPIN, default="none"),
            },
            checks=[lambda v, p: _equal_lengths(
                v, p, ("type_symbols", "type_norb", "cutoff_radius"))]),
    },
    tag_doc="Which model family the remaining keys describe.")


def _equal_lengths(value: dict, path: str, keys) -> List[Issue]:
    lengths = {key: len(value[key]) for key in keys if key in value}
    if len(set(lengths.values())) > 1:
        return [Issue(path, "one entry per species is required, but got " + ", ".join(
            "{}={}".format(key, n) for key, n in lengths.items()))]
    return []


# --- run options -----------------------------------------------------------


def _bounds_checks(value: dict, path: str) -> List[Issue]:
    issues: List[Issue] = []
    if value.get("mode") == "fixed":
        if "energy" not in value:
            issues.append(Issue(path, "mode 'fixed' needs 'energy': [emin, emax]"))
        elif value["energy"][0] >= value["energy"][1]:
            issues.append(Issue(join_path(path, "energy"), "emin must be below emax"))
    elif "energy" in value:
        issues.append(Issue(join_path(path, "energy"),
                            "only applies with mode 'fixed'; mode is 'auto' here",
                            severity=WARNING,
                            hint="set \"mode\": \"fixed\" to use these bounds"))
    return issues


BOUNDS = Obj(
    "The energy window that the Chebyshev expansions are scaled to.\n\n"
    "The window must contain the whole spectrum. Results at or beyond its "
    "endpoints are zero.",
    {
        "mode": F(_str("'auto': estimate the window from the extreme eigenvalues "
                       "during setup. 'fixed': use 'energy' and skip the estimate; "
                       "the window must then contain the whole spectrum.",
                       choices=("auto", "fixed")), default="auto"),
        "energy": F(Array(_num("Energy.", unit="eV"), "[emin, emax], used with mode 'fixed'.",
                          length=2)),
        "zero_tol": F(_num("If the norm of the Hamiltonian is below this value, the "
                           "estimate treats it as zero.", minimum=0.0),
                      default=1e-12),
        "max_matvecs": F(_int("Maximum number of matrix-vector products in the estimate.",
                              minimum=0),
                         default=10000),
    },
    checks=[_bounds_checks])

PRECONDITIONER = Obj(
    "How the overlap matrix S of a non-orthogonal model is handled.",
    {
        "operator_form": F(_str(
            "The operator that the Chebyshev expansions apply. 'solve' "
            "(default): S^-1 H, with S^-1 applied by a preconditioned "
            "conjugate-gradient solve at each step. 'sandwich': Z H Z with "
            "Z = S^-1/2, which must be stored; S is released after setup, so "
            "tasks that need S are not available.",
            choices=("sandwich", "solve")), default="solve"),
        "tol": F(_num("Convergence tolerance of the S^-1/2 iteration.",
                      exclusive_minimum=0.0), default=1e-7),
        "sptol": F(_num("Entries of S^-1/2 smaller than this are dropped.",
                        minimum=0.0), default=1e-5),
        "maxiter": F(_int(
            "Number of Newton-Schulz iterations that refine S^-1/2. 0 "
            "(default): the sparse approximation of S^-1/2 is used without "
            "refinement, as the preconditioner of the 'solve' form; its memory "
            "grows linearly with the number of atoms. The 'sandwich' form "
            "applies S^-1/2 directly and needs refinement, typically 20 "
            "iterations; where the refined S^-1/2 fits in memory, it is faster "
            "than 'solve'.",
            minimum=0), default=0),
    })

KPOINTS = Choice(
    "The k-points, in fractional coordinates: a list, a mesh or a path.\n\n"
    "Results have one row per k-point, in this order.",
    {
        "array": Array(VEC3, "Explicit fractional k-points."),
        "object": Obj(
            "A generated set of k-points.",
            {
                "mesh": F(Array(_int("Divisions along this reciprocal vector.", minimum=1),
                                "Gamma-centred mesh divisions.", length=3)),
                "shift": F(Array(_num("Fractional offset."),
                                 "Offset added to each mesh point.", length=3),
                           default=[0.0, 0.0, 0.0]),
                "line": F(Array(VEC3, "Vertices of a path through the Brillouin zone.",
                                min_length=2)),
                "points_per_segment": F(_int("Points on each segment of 'line', including "
                                             "its ends; a vertex shared by two segments "
                                             "appears once.",
                                             minimum=2), default=11),
            },
            require_exactly_one=[["mesh", "line"]],
            example={"mesh": [4, 4, 1]}),
    })

RUN = Obj(
    "Settings shared by all tasks: k-points, k-point parallelization, random "
    "vectors, the energy window and the handling of the overlap.",
    {
        "npar": F(_int(
            "Number of k-point groups the MPI processes are divided into. "
            "Groups work on different k-points; the processes in a group "
            "divide the atoms. Each process must own at least one atom, so for "
            "a small cell with many processes, use more groups.",
            minimum=1), default=1),
        "nsamples": F(_int("Number of random vectors. The statistical error falls as "
                           "1/sqrt(nsamples * N_orbital).", minimum=1),
                      default=1),
        "kpoints": F(KPOINTS, default=[[0.0, 0.0, 0.0]]),
        "bounds": F(BOUNDS, default={}),
        "preconditioner": F(PRECONDITIONER, default={}),
    })


# --- tasks -----------------------------------------------------------------

LABEL = F(_str("Name of this result in the output. Default: the task type, with an "
               "index added if the type appears more than once."))

N_MOMENTS = _int("Number of Chebyshev moments. The energy resolution is about "
                 "pi * (emax - emin) / (2 * n_moments).", minimum=1)

KERNEL = _str("Kernel applied to the Chebyshev moments. 'jackson': the result is "
              "positive. 'costaper': a smaller bias, better when the result is "
              "only used inside an integral over energy.",
              choices=("jackson", "costaper"))

FORMULATION = _str("Kubo formulation. 'greenwood': the Fermi-surface term. "
                   "'bastin': also the Fermi-sea term.",
                   choices=("greenwood", "bastin"))

ENERGY_GRID = Choice(
    "Energies to evaluate at, in eV.",
    {
        "array": Array(_num("Energy.", unit="eV"), "Explicit energies."),
        "object": Obj(
            "A uniform energy grid.",
            {
                "min": F(_num("First energy.", unit="eV"), required=True),
                "max": F(_num("Last energy.", unit="eV"), required=True),
                "n": F(_int("Number of points, endpoints included.", minimum=2),
                       required=True),
            }),
    })

TASK = Union(
    "One calculation. Tasks are computed in order and share one setup.",
    "type",
    {
        "energy_bounds": Obj(
            "Report the energy window. Used to check that the model and "
            "structure set up correctly.",
            {"label": LABEL}),
        "dos": Obj(
            "Density of states.",
            {
                "label": LABEL,
                "dE": F(_num("Energy resolution. The number of Chebyshev "
                             "moments follows from the energy window; the "
                             "cost is proportional to (emax - emin) / dE.",
                             unit="eV",
                             exclusive_minimum=0.0), default=0.1),
                "method": F(_str("How the result is formed from the Chebyshev "
                                 "moments. 'kpm': the Jackson kernel; the "
                                 "result is positive. 'corr': a Hann window "
                                 "on the time correlation function "
                                 "reconstructed from the moments.",
                                 choices=("kpm", "corr")), default="kpm"),
            }),
        "pdos": Obj(
            "Density of states projected onto groups of orbitals. Each group "
            "costs about as much as one dos task, and the groups sum to the "
            "total.\n\n"
            "The projection follows the operator form: 'sandwich' projects "
            "onto Lowdin orbitals (non-negative); 'solve' gives Mulliken "
            "populations, which can be negative. The result records which "
            "was used.",
            {
                "label": LABEL,
                "dE": F(_num("Energy resolution, as in the dos task.",
                             unit="eV", exclusive_minimum=0.0), default=0.1),
                "projection": F(_str("'atom': one group per atom, in input "
                                     "order. 'orbital_slot': one group per "
                                     "element and orbital index within the "
                                     "atom; the output names each group by "
                                     "'group_type' (atomic number, in "
                                     "increasing order) and 'group_slot' (the "
                                     "orbital index, in the order of the "
                                     "basis).",
                                     choices=("atom", "orbital_slot")),
                                default="atom"),
            }),
        "ldos": Obj(
            "Local density of states on a real-space grid, at a list of "
            "energies.\n\n"
            "Requires an lcao model and a single k-point at Gamma. All energies "
            "are computed together, so additional energies add little cost. "
            "The result includes its estimated relative error, "
            "'predicted_rel_error', which falls as 1/sqrt(n_samples).\n\n"
            "Each MPI process writes its own file, <label>.rank00000.npz and "
            "so on, with its share of the grid points and their indices "
            "('point_index') in the full grid of shape 'shape'.",
            {
                "label": LABEL,
                "energies": F(ENERGY_GRID, required=True),
                "eta": F(_num("Gaussian broadening. The cost is proportional to "
                              "(emax - emin) / eta. For a non-orthogonal model the "
                              "energy window is set by the small eigenvalues of "
                              "the overlap and is wider than the band width.",
                              unit="eV", exclusive_minimum=0.0), default=0.1),
                "spacing": F(_num("Grid spacing. The actual spacing is at most this "
                                  "value. The number of grid points grows as "
                                  "spacing^-3.", unit="Angstrom",
                                  exclusive_minimum=0.0), default=0.4),
                "box_edge": F(_int("Side of the grid boxes the calculation works "
                                   "on, in grid points. 0: boxes about 2.4 "
                                   "Angstrom wide at any spacing.", minimum=0),
                              default=0),
                "n_samples": F(_int("Number of random vectors. The relative error "
                                    "falls as 1/sqrt(n_samples). At least 2.",
                                    minimum=2), default=64),
                "block": F(_int("Random vectors processed together in one "
                                "Chebyshev sweep; larger values use more memory.",
                                minimum=1), default=16),
                "seed": F(_int("Random seed. The random vectors are tied to the "
                               "atoms, not to the MPI processes, so the result "
                               "does not depend on the number of processes apart "
                               "from rounding.", minimum=0), default=12345),
                "sketch_families": F(_int("Number of Chebyshev filter families per "
                                          "random vector. Families of the same "
                                          "vector are correlated, so keep the "
                                          "default and increase n_samples for "
                                          "accuracy.",
                                          minimum=1),
                                     default=6),
            }),
        "density": Obj(
            "Electron density on a real-space grid, for the given Hamiltonian "
            "at chemical potential mu and temperature temp. It is not a "
            "self-consistent density.\n\n"
            "Requires an lcao model and a single k-point at Gamma. Each MPI "
            "process writes its own file, as for ldos. 'electrons_on_grid' is "
            "the density integrated over the grid; compare it with the "
            "electron count to check the grid.\n\n"
            "The cost is proportional to (emax - emin) / temp, so low "
            "temperatures are expensive. For an insulator the density does "
            "not depend on the temperature while it is small compared with "
            "the band gap.",
            {
                "label": LABEL,
                "mu": F(_num("Chemical potential; the fermi_level task finds "
                             "it from an electron count.", unit="eV"),
                        required=True),
                "temp": F(_num("Temperature of the Fermi-Dirac occupation.", unit="K",
                               exclusive_minimum=0.0), default=300.0),
                "spin_degeneracy": F(_num("Electrons per state: 2 for a "
                                          "spin-degenerate basis.",
                                          exclusive_minimum=0.0), default=2.0),
                "spacing": F(_num("Grid spacing, as in ldos.",
                                  unit="Angstrom", exclusive_minimum=0.0),
                             default=0.4),
                "box_edge": F(_int("Grid box side in points, as in ldos.",
                                   minimum=0), default=0),
                "n_samples": F(_int("Number of random vectors.",
                                    minimum=1), default=64),
                "block": F(_int("Random vectors processed together in one "
                                "Chebyshev sweep.", minimum=1),
                           default=16),
                "seed": F(_int("Random seed; see ldos.", minimum=0), default=12345),
                "fragment_core": F(_num("Size of the core region of each fragment "
                                        "(see fragment_buffer). It affects only "
                                        "the cost of the fragment calculations.",
                                        unit="Angstrom",
                                        exclusive_minimum=0.0), default=4.5),
                "fragment_buffer": F(_num("Buffer radius of the fragment "
                                          "correction; 0 disables it. A "
                                          "deterministic calculation on local "
                                          "fragments gives most of the density, "
                                          "and the random vectors correct the "
                                          "remainder. This reduces the noise at "
                                          "the same number of matrix-vector "
                                          "products. A larger buffer makes the "
                                          "fragment part more accurate.",
                                          unit="Angstrom", minimum=0.0),
                                     default=0.0),
                "fragment_workspace_mb": F(_int("Memory limit for the fragment "
                                                "calculations, in MB. 0: no limit "
                                                "(one fragment per thread).",
                                                minimum=0), default=0),
                "fragment_coefficient": F(
                    _str("Coefficient of the fragment correction. 'fixed': "
                         "1, which keeps the estimate unbiased. 'fitted': "
                         "estimated locally from the same random vectors.",
                         choices=("fixed", "fitted")),
                    default="fitted"),
                "energy_windows": F(_int("Number of energy windows the occupation "
                                         "is divided into. Each window is "
                                         "sampled separately in the same "
                                         "Chebyshev sweep, which reduces the "
                                         "noise without extra matrix-vector "
                                         "products. Use it for metals; for "
                                         "gapped systems use fragment_buffer "
                                         "instead, as the two do not combine "
                                         "well. 1 disables it.",
                                         minimum=1), default=1),
            }),
        "ac": Obj(
            "Optical (AC) conductivity in S/m, including both spins, and the "
            "dielectric function, refractive index and absorption coefficient "
            "computed from it. Requires the velocity operator.",
            {
                "label": LABEL,
                "nt": F(_int("Number of time steps.", minimum=1), default=200),
                "n_kpm": F(_int("Number of Chebyshev moments for the Fermi-Dirac "
                                "occupation.", minimum=1),
                           default=1000),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), default=300.0),
                "mu": F(_num("Chemical potential.", unit="eV"), default=0.0),
                "axis": F(AXIS, default="abc"),
                "method": F(_str("Time evolution. 'concurrent': time steps "
                                 "share Chebyshev states, which needs fewer "
                                 "matrix-vector products. 'sequential': one "
                                 "step at a time; used as an independent "
                                 "check.",
                                 choices=("concurrent", "sequential")),
                            default="concurrent"),
                "block": F(_int("For 'concurrent': time steps per block. Larger "
                                "blocks need fewer matrix-vector products and "
                                "more memory. Default: min(nt, 128).",
                                minimum=1)),
            }),
        "dc": Obj(
            "DC conductivity tensor at one chemical potential and temperature, "
            "in S/m, including both spins. Requires the velocity operator.",
            {
                "label": LABEL,
                "n_moments": F(N_MOMENTS, default=1000),
                "mu": F(_num("Chemical potential.", unit="eV"), required=True),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), required=True),
                "formulation": F(FORMULATION, default="greenwood"),
                "axis": F(AXIS, default="abc"),
                "kernel": F(KERNEL, default="jackson"),
                "method": F(_str("'fftcheb': evaluates the whole energy grid; "
                                 "required for 'bastin'. 'qe': evaluates only "
                                 "the energies near mu that contribute at this "
                                 "temperature, with less memory; 'greenwood' "
                                 "only.",
                                 choices=("fftcheb", "qe")), default="fftcheb"),
                "partition_sections": F(_int("Number of row blocks the calculation is "
                                             "split into to limit memory. 0: chosen "
                                             "from max_workspace_mb.",
                                             minimum=0), default=0),
                "max_workspace_mb": F(_int("Memory limit, in MB, used to choose "
                                           "partition_sections. 0: no limit.",
                                           minimum=0), default=0),
            }),
        "dc_spectrum": Obj(
            "DC conductivity at zero temperature as a function of energy, in "
            "S/m, including both spins. Requires the velocity operator.",
            {
                "label": LABEL,
                "n_moments": F(N_MOMENTS, default=1000),
                "formulation": F(FORMULATION, default="greenwood"),
                "axis": F(AXIS, default="abc"),
                "kernel": F(KERNEL, default="jackson"),
                "method": F(_str("'fftcheb' (default): evaluates the whole "
                                 "energy grid. 'qe': quasi-eigenstate "
                                 "accumulation, 'greenwood' only; mainly "
                                 "useful in the dc task, which needs only "
                                 "energies near mu.",
                                 choices=("fftcheb", "qe")), default="fftcheb"),
                "partition_sections": F(_int("Number of row blocks, as in dc.",
                                             minimum=0), default=0),
                "max_workspace_mb": F(_int("Memory limit in MB, as in dc.",
                                           minimum=0), default=0),
            }),
        "occupation": Obj(
            "Electron count and band energy at one chemical potential and "
            "temperature. No spin factor is applied: a spin-degenerate basis "
            "counts one spin.",
            {
                "label": LABEL,
                "mu": F(_num("Chemical potential.", unit="eV"), required=True),
                "temp": F(_num("Temperature; 0 gives a step function.", unit="K",
                               minimum=0.0), required=True),
                "indicator_threshold": F(
                    _num("Threshold of the error indicator that chooses the "
                         "expansion degree. The indicator is not an error "
                         "bound: for systems with few states near mu, check "
                         "the result with a larger n_max.",
                         exclusive_minimum=0.0),
                    default=1e-4),
                "n_max": F(_int("Maximum expansion degree.", minimum=1),
                           default=65536),
            }),
        "fermi_level": Obj(
            "The chemical potential at which the electron count equals "
            "n_electrons.\n\n"
            "The search reuses one set of Chebyshev moments, so it costs about "
            "as much as one occupation task.",
            {
                "label": LABEL,
                "n_electrons": F(_num("Target electron count. No spin factor is "
                                      "applied: for a spin-degenerate basis, give "
                                      "half the number of electrons.",
                                      exclusive_minimum=0.0), required=True),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), required=True),
                "indicator_threshold": F(
                    _num("Threshold of the error indicator; see the occupation "
                         "task.",
                         exclusive_minimum=0.0),
                    default=1e-4),
                "n_max": F(_int("Maximum expansion degree.", minimum=1),
                           default=65536),
                "mu_guess": F(_num("Energy at which the expansion degree is chosen. "
                                   "Default: the centre of the energy window.",
                                   unit="eV")),
            }),
        "msd": Obj(
            "Mean-square displacement DeltaX^2(E, t) of wave packets, and the "
            "diffusion coefficient D = 0.5 d/dt DeltaX^2. Requires the Gamma "
            "point among the k-points.",
            {
                "label": LABEL,
                "n_steps": F(_int("Number of time steps.", minimum=1), default=200),
                "dt_fs": F(_num("Time step.", unit="fs", exclusive_minimum=0.0),
                           default=10.0),
                "n_kpm": F(_int("Number of Chebyshev moments; sets the energy "
                                "resolution.", minimum=1),
                           default=1000),
                "axis": F(_str("Direction the displacement is measured along.",
                               choices=("x", "y", "z")), default="x"),
                "method": F(_str("Algorithm; both give the same result. "
                                 "'stepwise': suited to orthogonal models. "
                                 "'spectral': the cost does not depend on the "
                                 "number of time steps; suited to "
                                 "non-orthogonal models. 'auto': chooses one; "
                                 "set the method explicitly when comparing "
                                 "runs.",
                                 choices=("auto", "stepwise", "spectral")),
                            default="auto"),
                "n_kpm_compute": F(_int("Compute this many moments and extend them "
                                        "to n_kpm by linear prediction; 0 "
                                        "disables it. Not reliable for small "
                                        "cells with discrete spectra.",
                                        minimum=0),
                                   default=0),
            }),
        "quantum_metric": Obj(
            "Longitudinal quantum metric Gxx, computed from the optical "
            "conductivity. Requires the velocity operator.",
            {
                "label": LABEL,
                "nt": F(_int("Number of time steps.", minimum=1), default=200),
                "n_kpm": F(_int("Number of Chebyshev moments for the Fermi-Dirac "
                                "occupation.", minimum=1),
                           default=1000),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), default=0.0),
                "mu": F(_num("Chemical potential.", unit="eV"), default=0.0),
                "axis": F(AXIS, default="abc"),
                "omega_min_ev": F(_num("Lower frequency cutoff of the integral.",
                                       unit="eV", minimum=0.0), default=0.0),
            }),
    },
    tag_doc="Which calculation the remaining keys configure.")


# --- output ----------------------------------------------------------------

OUTPUT = Obj(
    "Where and how results are written. MPI process 0 writes all files, "
    "except the real-space maps (ldos, density), which each process writes "
    "to its own <label>.rankNNNNN.npz.",
    {
        "dir": F(_path("Output directory, created if it does not exist. A "
                       "relative path is relative to the input file."), default="."),
        "prefix": F(_str("Text added to the start of each file name."), default=""),
        "format": F(_str("'npz': one compressed archive per task. 'json': arrays "
                         "written as nested lists. 'none': only the summary.",
                         choices=("npz", "json", "none")), default="npz"),
        "summary": F(_bool("Write <prefix>summary.json, which records the resolved "
                           "input, the basis, the energy window, the scalar "
                           "results, and the shape and file of each array."),
                     default=True),
    })


# --- the whole document ----------------------------------------------------

ROOT = Obj(
    "An RSATB input: the structure, the model, the run settings, the tasks "
    "and the output.\n\n"
    "Unknown keys are errors. Keys that start with an underscore, such as "
    "\"_comment\", are ignored at any level and can be used for comments.",
    {
        "version": F(_int("Input-format version.", choices=(INPUT_VERSION,)),
                     default=INPUT_VERSION),
        "structure": F(STRUCTURE, required=True),
        "model": F(MODEL, required=True),
        "run": F(RUN, default={}),
        "tasks": F(Array(TASK, "Tasks to compute, in order. With an empty list, "
                               "RSATB sets up the calculation and reports the "
                               "basis and energy window without computing "
                               "anything."),
                   default=[]),
        "output": F(OUTPUT, default={}),
    })


# --- topic index for `rsatb doc` -------------------------------------------

def topics() -> "dict[str, Node]":
    """Returns every documentable node, keyed by the name `rsatb doc` takes."""
    index = {
        "input": ROOT,
        "structure": STRUCTURE,
        "model": MODEL,
        "run": RUN,
        "run.kpoints": KPOINTS,
        "run.bounds": BOUNDS,
        "run.preconditioner": PRECONDITIONER,
        "tasks": TASK,
        "output": OUTPUT,
        "radial": RADIAL,
    }
    for name, variant in MODEL.variants.items():
        index["model." + name] = variant
    for name, variant in TASK.variants.items():
        index["tasks." + name] = variant
    for name, variant in RADIAL.variants.items():
        index["radial." + name] = variant
    index["model.sktb.parameters"] = SK_PARAMETERS
    return index
