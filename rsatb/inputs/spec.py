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
    "How spin enters the basis. 'none' is one channel per orbital; 'collinear' "
    "keeps one channel but two independent spin blocks; 'noncollinear' and "
    "'spinorbit' double the basis size.",
    choices=("none", "collinear", "noncollinear", "spinorbit"))

AXIS = _str(
    "Cell axes spanning the normalization volume in the transport prefactor. "
    "'abc' is the full cell volume; a cyclic pair ('ab', 'bc', 'ca') the area "
    "of that lattice plane, for 2D systems padded with vacuum; a single "
    "letter that lattice vector's length, for 1D. The conductivity tensor "
    "itself always keeps all six Cartesian components.",
    choices=("abc", "ab", "bc", "ca", "a", "b", "c"))

HAMILTONIAN_FILE = Obj(
    "A precomputed real-space Hamiltonian H(R) and the file layout it uses.",
    {
        "path": F(_path("Path to the Hamiltonian file, relative to the input file's "
                        "directory unless absolute."), required=True),
        "format": F(_str("On-disk layout. 'deeptb' is HDF5 with a single group '0' "
                         "holding datasets named 'i_j_Rx_Ry_Rz'. 'abacus' is ABACUS's "
                         "sparse H(R) from out_mat_hs2 (hrs1_nao.csr; data-HR-sparse_SPIN0.csr "
                         "in older versions), text or binary, "
                         "in Rydberg. 'nextham' is NextHam's post-processed "
                         "HR_atom_pairs.h5. Both of the latter are in ABACUS's orbital "
                         "order, which is converted using the orbital files, so they "
                         "need the 'lcao' model, a structure listing the atoms in "
                         "ABACUS's order, and a spin-degenerate (nspin=1) run.",
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
            issues.append(Issue(path, "'symbols' and 'numbers' are two spellings of the "
                                      "same thing; give one"))
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
    "Where the atoms are. Read from a file, or written inline.\n\n"
    "Coordinates are used exactly as given: nothing is wrapped, sorted or "
    "renumbered on the way in, because atom order and coordinates are what a "
    "Hamiltonian file's own indexing and R-vector convention refer to. Atoms "
    "may sit outside the cell.",
    {
        "file": F(_path("Structure file. POSCAR/.vasp, .xyz and .cif are read by "
                        "extension unless 'format' says otherwise.")),
        "format": F(_str("Explicit reader for 'file'; empty means infer from the "
                         "extension."), default=""),
        "positions": F(Array(VEC3, "Cartesian positions in Angstrom, one triple per atom.")),
        "symbols": F(Array(_str("Chemical symbol."), "Chemical symbol per atom, e.g. "
                                                     "[\"C\", \"C\"].")),
        "numbers": F(Array(_int("Atomic number."), "Atomic number per atom; an "
                                                   "alternative spelling of 'symbols'.")),
        "cell": F(Array(VEC3, "Lattice vectors in Angstrom, as three rows.", length=3)),
        "pbc": F(Array(_bool("Periodic along this lattice vector."),
                       "Periodicity per lattice vector.", length=3),
                 default=[True, True, True]),
        "supercell": F(Array(_int("Tiling factor.", minimum=1),
                             "Integer tiling applied before setup. [1, 1, 1] leaves the "
                             "cell alone.", length=3),
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
            issues.append(Issue(path, "'file' and inline 'r'/'values' are two ways to say "
                                      "the same thing; give one"))
    return issues


def _radial_exp_poly_checks(value: dict, path: str) -> List[Issue]:
    if value.get("r_on", 0.0) > value.get("r_cut", 0.0):
        return [Issue(join_path(path, "r_on"),
                      "the taper must start inside the cutoff: 0 <= r_on <= r_cut")]
    return []


R_CUT = _num("Hard cutoff. The channel is zero beyond it, and the atomic graph "
             "is sized from the largest cutoff of each species pair.",
             unit="Angstrom", exclusive_minimum=0.0)

RADIAL = Union(
    "Distance dependence of one Slater-Koster channel.",
    "kind",
    {
        "constant": Obj(
            "A constant value out to the cutoff. Convenient for a single "
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
                "r_on": F(_num("Distance the cosine taper starts at. 0 means the value "
                               "is cut abruptly at r_cut.", unit="Angstrom", minimum=0.0),
                          default=0.0),
            },
            checks=[_radial_exp_poly_checks],
            example={"kind": "exp_poly", "coeffs": [-2.7], "lambda": 1.7,
                     "r_on": 3.0, "r_cut": 4.0}),
    },
    tag_doc="Which radial form the remaining keys describe.")

SK_PARAMETERS = Obj(
    "Onsite energies and pair channels for ONE matrix -- H or S.\n\n"
    "H and S are separate objects, so there is nothing to disambiguate: an "
    "SKTB model with no 'overlap' is orthogonal, and one with an 'overlap' is "
    "not.",
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
          "Orbitals of this species. A shell alias expands: \"p\" is py, pz, px.",
          min_length=1),
    "The basis, per species. \"C\": [\"pz\"] is one orbital per carbon.",
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
    "What the Hamiltonian is. Four families, each with one story.\n\n"
    "The model reports the basis it works in, so orbital counts and graph "
    "cutoffs are never input: they are read from the parameters or the basis "
    "files. 'tb_file' is the one exception, because the supported file formats "
    "carry no basis header.",
    "type",
    {
        "sktb": Obj(
            "Slater-Koster model built from radial parameters you write.\n\n"
            "Orthogonal unless 'overlap' is given. An orthogonal SKTB model "
            "supplies the velocity and position operators for free, so it can "
            "run every observable.",
            {
                "basis": F(BASIS, required=True),
                "hamiltonian": F(SK_PARAMETERS, required=True),
                "overlap": F(SK_PARAMETERS,
                             note="Absent means an orthogonal basis."),
                "spin": F(SPIN, default="none"),
            },
            checks=[_sktb_basis_crosscheck]),
        "lcao": Obj(
            "A precomputed Hamiltonian plus the numerical atomic orbitals it "
            "was written in.\n\n"
            "An LCAO basis is non-orthogonal, so the orbital files are "
            "required, not optional: they are what the overlap is computed "
            "from. Adding pseudopotentials additionally enables the velocity "
            "operator. Orbital counts and cutoff radii are read from the .orb "
            "files.",
            {
                "hamiltonian": F(HAMILTONIAN_FILE, required=True),
                "orbitals": F(Obj(
                    "Numerical atomic orbital files, one per species in "
                    "ascending-Z order.",
                    {
                        "names": F(Array(_str("File name."), "One .orb file per species.",
                                         min_length=1), required=True),
                        "dir": F(_path("Directory the names are resolved against."),
                                 default=""),
                    }), required=True),
                "pseudopotentials": F(Obj(
                    "Pseudopotential files, in the same order as the orbital "
                    "files. Optional: supplying them enables the velocity "
                    "operator, and so the conductivity observables.",
                    {
                        "names": F(Array(_str("File name."), "One .upf file per species."),
                                   default=[]),
                        "dir": F(_path("Directory the names are resolved against."),
                                 default=""),
                    }), default={}),
                "spin": F(SPIN, default="none"),
                "log_file": F(_path("Where the op2c integral engine writes its log."),
                              default="op2c.log"),
            }),
        "dftb": Obj(
            "DFTB Slater-Koster tables read from a directory of .skf files.\n\n"
            "DFTB data always carries both H and S, so the model is "
            "non-orthogonal. The basis is declared rather than inferred: a "
            ".skf tabulates all ten channels up to d whether or not the "
            "parameterization uses them.",
            {
                "skf_dir": F(_path("Directory holding the .skf files, one per ordered "
                                   "species pair (C-C.skf, C-H.skf, H-C.skf, ...)."),
                             required=True),
                "basis": F(BASIS, required=True),
                "heteronuclear": F(_str(
                    "What to do with a mixed-shell channel whose reading the "
                    "declared bases do not force. 'reject_ambiguous' refuses; it "
                    "only bites when both species carry both shells of the "
                    "channel.",
                    choices=("reject_ambiguous", "assume_lower_shell_on_row_species")),
                    default="reject_ambiguous"),
                "spin": F(SPIN, default="none"),
            }),
        "tb_file": Obj(
            "An orthogonal H(R) read straight from a file -- what a DeePTB "
            "Slater-Koster export produces.\n\n"
            "The one model that must be told its basis: the supported formats "
            "carry no basis header, so orbital counts cannot be derived.",
            {
                "hamiltonian": F(HAMILTONIAN_FILE, required=True),
                "type_symbols": F(Array(_str("Chemical symbol."),
                                        "Species, in the type order the file indexes atoms by.",
                                        min_length=1), required=True),
                "type_norb": F(Array(_int("Orbital count.", minimum=1),
                                     "Orbitals per species, spin-expanded.", min_length=1),
                               required=True),
                "cutoff_radius": F(Array(_num("Cutoff.", unit="Angstrom", exclusive_minimum=0.0),
                                         "Per species: an edge exists iff d <= r_i + r_j.",
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
    "The spectral window the Chebyshev expansions are scaled to.\n\n"
    "Every observable is defined on the open interval: energies at or beyond "
    "the endpoints reconstruct to zero.",
    {
        "mode": F(_str("'auto' estimates the window with a PRIMME solve at setup. "
                       "'fixed' uses the window you give, which is usually faster "
                       "and more robust on large cells.",
                       choices=("auto", "fixed")), default="auto"),
        "energy": F(Array(_num("Energy.", unit="eV"), "[emin, emax], with mode 'fixed'.",
                          length=2)),
        "zero_tol": F(_num("Below this the Hamiltonian is treated as zero, so a "
                           "trivially empty operator does not send the estimate "
                           "into an endless solve.", minimum=0.0), default=1e-12),
        "max_matvecs": F(_int("Cap on matrix-vector products in the estimate.", minimum=0),
                         default=10000),
    },
    checks=[_bounds_checks])

PRECONDITIONER = Obj(
    "How S is preconditioned for the iterative solves.",
    {
        "operator_form": F(_str(
            "Which operator the Chebyshev expansions apply. 'sandwich' is "
            "Z H Z with Z = S^-1/2: exactly Hermitian, two SpMVs per matvec, "
            "and S is released after the build (saves its memory; observables "
            "that need S become illegal). 'solve' keeps S and applies S^-1 H "
            "by preconditioned CG each matvec -- the fallback when Z would "
            "densify beyond memory. Formerly the bool "
            "release_overlap_after_build (true = sandwich).",
            choices=("sandwich", "solve")), default="solve"),
        "tol": F(_num("Convergence tolerance of the inverse-square-root iteration.",
                      exclusive_minimum=0.0), default=1e-7),
        "sptol": F(_num("Sparsity threshold: entries below it are dropped from the "
                        "preconditioner.", minimum=0.0), default=1e-5),
        "maxiter": F(_int(
            "Newton-Schulz iteration cap. 0 keeps the graph seed itself as "
            "S^-1/2 -- sparse but only subgraph-accurate, which is the "
            "seed-only PCG route when the overlap is kept "
            "(operator_form 'solve').", minimum=0), default=20),
    })

KPOINTS = Choice(
    "Which k-points to sample, in fractional coordinates.\n\n"
    "Write them out, or describe a mesh or a line. Observables return one row "
    "per k-point, in the order given here.",
    {
        "array": Array(VEC3, "Explicit fractional k-points."),
        "object": Obj(
            "A generated set of k-points.",
            {
                "mesh": F(Array(_int("Divisions along this reciprocal vector.", minimum=1),
                                "Gamma-centred mesh divisions.", length=3)),
                "shift": F(Array(_num("Fractional offset."),
                                 "Offset added to every mesh point.", length=3),
                           default=[0.0, 0.0, 0.0]),
                "line": F(Array(VEC3, "Vertices of a path through the zone.", min_length=2)),
                "points_per_segment": F(_int("Points on each segment of 'line', endpoints "
                                             "included; shared vertices are not repeated.",
                                             minimum=2), default=11),
            },
            require_exactly_one=[["mesh", "line"]],
            example={"mesh": [4, 4, 1]}),
    })

RUN = Obj(
    "What the calculation does, independent of which model supplies the "
    "operators.",
    {
        "npar": F(_int(
            "How many k-point-parallel sub-communicators the ranks are split "
            "into. Ranks within one sub-communicator share a k-point and "
            "divide the atoms; every rank must own at least one atom, so npar "
            "is the knob that puts spare ranks on k-points instead.",
            minimum=1), default=1),
        "nsamples": F(_int("Random vectors in the stochastic trace. The statistical "
                           "error falls as 1/sqrt(nsamples * N_orbital).", minimum=1),
                      default=1),
        "kpoints": F(KPOINTS, default=[[0.0, 0.0, 0.0]]),
        "bounds": F(BOUNDS, default={}),
        "preconditioner": F(PRECONDITIONER, default={}),
    })


# --- tasks -----------------------------------------------------------------

LABEL = F(_str("Name this result is stored under. Defaults to the task type, with "
               "an index appended when a type appears more than once."))

N_MOMENTS = _int("Chebyshev moments. The energy resolution is roughly "
                 "pi * (emax - emin) / (2 * n_moments).", minimum=1)

KERNEL = _str("Chebyshev summation rule. 'jackson' is positive and pointwise "
              "clean; 'costaper' has a higher-order bias decay and is the "
              "better choice when the spectrum is only used inside an energy "
              "integral.", choices=("jackson", "costaper"))

FORMULATION = _str("Kubo formulation. 'greenwood' is the Fermi-surface term; "
                   "'bastin' adds the Fermi-sea contribution.",
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
    "One observable to compute. Tasks run in order against the same setup, so "
    "asking for several costs one setup, not several.",
    "type",
    {
        "energy_bounds": Obj(
            "Report the spectral window and stop. Cheap way to check that a "
            "model and structure set up at all.",
            {"label": LABEL}),
        "dos": Obj(
            "Density of states by stochastic Chebyshev expansion.",
            {
                "label": LABEL,
                "dE": F(_num("Target energy resolution. The Chebyshev step "
                             "and moment counts are derived from the "
                             "spectral span at run time; cost is linear in "
                             "span/dE. Replaces the former nt (= span/dE) "
                             "and n_moments.", unit="eV",
                             exclusive_minimum=0.0), default=0.1),
                "method": F(_str("Broadening kernel; both run on the same "
                                 "moment recurrence (~2*span/dE matvecs). "
                                 "'kpm' is Jackson, strictly positive; "
                                 "'corr' reconstructs the correlation series "
                                 "from the moments and applies the "
                                 "historical Hann window + FFT, matching "
                                 "legacy outputs.",
                                 choices=("kpm", "corr")), default="kpm"),
            }),
        "pdos": Obj(
            "Projected density of states, resolved onto orbital groups by a "
            "per-group reduction of the same stochastic Chebyshev run the dos "
            "task performs -- every group costs one DOS run, and the group "
            "rows sum to the returned total bin by bin.\n\n"
            "The population convention follows the operator form and is "
            "reported in the result rather than chosen: the sandwich form "
            "projects onto Lowdin orbitals (non-negative); the solve form "
            "onto Mulliken populations, which may legitimately be negative.",
            {
                "label": LABEL,
                "dE": F(_num("Target energy resolution, as the dos task.",
                             unit="eV", exclusive_minimum=0.0), default=0.1),
                "projection": F(_str("'atom' groups by atom (original input "
                                     "order); 'orbital_slot' groups by (atom "
                                     "type, orbital position within the atom "
                                     "block), so two elements never share a "
                                     "group even where their basis sizes "
                                     "coincide. It writes 'group_type' and "
                                     "'group_slot' naming each group, types "
                                     "being the distinct atomic numbers in "
                                     "ascending order; with one element the "
                                     "groups are simply the slots, in order. "
                                     "Slot maps to a shell once the caller "
                                     "applies the basis ordering.",
                                     choices=("atom", "orbital_slot")),
                                default="atom"),
            }),
        "ldos": Obj(
            "Local density of states on a real-space grid, by the "
            "survey-sampling estimator (doc/design_survey_sampling_ldos.md): a "
            "flat-top Chebyshev sketch of the energy window and its exact "
            "filtered images from one recursion per probe; the map is the "
            "captured part of the window's sketch divided by its capture "
            "fraction. Unbiased, non-negative, error sqrt(2 (1 - capture) / "
            "rank) with rank = sketch_families x n_samples -- bounded by "
            "sqrt(2 / rank) at every system size, exact once the rank reaches "
            "the states in the window. Needs the radial orbital files, so an "
            "LCAO model, at a single Gamma point.\n\n"
            "One recursion serves every energy: the marginal cost of an energy "
            "point is a few AXPYs per Chebyshev step.\n\n"
            "Written one file per rank -- <label>.rank00000.npz and so on -- "
            "because no rank holds the whole grid. Each file carries the grid "
            "shape, the energies, and its own points, so it is self-describing; "
            "reassemble by scattering 'ldos' into a flat (n_energy, n_point) "
            "array at 'point_index' and reshaping to 'shape'.",
            {
                "label": LABEL,
                "energies": F(ENERGY_GRID, required=True),
                "eta": F(_num("Gaussian broadening. The filter degree scales as "
                              "(spectral span)/eta, so this sets the cost -- and "
                              "the span of S^-1 H is set by the overlap's small "
                              "eigenvalues, not by the band width.",
                              unit="eV", exclusive_minimum=0.0), default=0.1),
                "spacing": F(_num("Grid point separation; the realised spacing is "
                                  "never coarser than this. Output volume grows as "
                                  "spacing^-3.", unit="Angstrom",
                                  exclusive_minimum=0.0), default=0.4),
                "box_edge": F(_int("Grid box side in points. 0 picks a physical "
                                   "(~2.4 Angstrom) box whatever the spacing, "
                                   "which is what keeps the per-box orbital count "
                                   "bounded on a coarse grid.", minimum=0),
                              default=0),
                "n_samples": F(_int("Random probes. The sketch rank is "
                                    "sketch_families x n_samples and the error "
                                    "scale sqrt(2 (1 - capture) / rank), so at "
                                    "a fixed family count the noise falls as "
                                    "1/sqrt(n_samples); this is the knob to buy "
                                    "accuracy with. At least 2: the capture "
                                    "fraction is cross-fitted between probe "
                                    "folds.", minimum=2), default=64),
                "block": F(_int("Probes per Chebyshev sweep, i.e. the SpMM width. "
                                "Memory holds a few blocks of that many vectors.",
                                minimum=1), default=16),
                "seed": F(_int("Random seed. A run repeats exactly at the same "
                               "seed AND the same rank count: each rank draws "
                               "its own stream, so changing the rank count "
                               "redraws the states and gives a statistically "
                               "equivalent map, not an identical one -- the two "
                               "differ by the sampling error, which falls as "
                               "1/sqrt(n_samples).", minimum=0), default=12345),
                "sketch_families": F(_int("Chebyshev modulation families per "
                                          "probe, J. The sketch rank is "
                                          "J x n_samples and the error scale "
                                          "sqrt(2 (1 - capture) / rank); each "
                                          "family lengthens the sweep by about "
                                          "a tenth of the filter degree, and "
                                          "families of one probe are correlated "
                                          "draws (design effect 1.0 at J = 6, "
                                          "2.5 at J = 12), so J ~ 6 with probes "
                                          "supplying the rest of the rank is "
                                          "the efficient setting.", minimum=1),
                                     default=6),
            }),
        "density": Obj(
            "Electron density on a real-space grid: the ldos estimator with "
            "the energy filter replaced by the square root of the Fermi-Dirac "
            "occupation at (mu, temp). One Chebyshev sweep, one accumulator, "
            "non-negative pointwise; Gamma-only and needs the radial orbital "
            "files, like ldos. A post-processing observable of the fixed "
            "Hamiltonian -- not an SCF density.\n\n"
            "Written one file per rank like ldos, with 'density' in place of "
            "'ldos' and no energy axis. 'electrons_on_grid' integrates the "
            "map; it approaches the occupation sum as the grid refines and "
            "its residual is a free quadrature diagnostic.\n\n"
            "The Chebyshev degree scales as (spectral span)/(pi*kB*temp), so "
            "low temperatures are expensive: the smearing is a numerical knob "
            "as much as a physical one.",
            {
                "label": LABEL,
                "mu": F(_num("Chemical potential; the fermi_level task finds "
                             "it from an electron count.", unit="eV"),
                        required=True),
                "temp": F(_num("Smearing temperature.", unit="K",
                               exclusive_minimum=0.0), default=300.0),
                "spin_degeneracy": F(_num("States per orbital; 2 for a "
                                          "spin-paired calculation.",
                                          exclusive_minimum=0.0), default=2.0),
                "spacing": F(_num("Grid point separation, as ldos.",
                                  unit="Angstrom", exclusive_minimum=0.0),
                             default=0.4),
                "box_edge": F(_int("Grid box side in points, as ldos.",
                                   minimum=0), default=0),
                "n_samples": F(_int("Random states in the estimator.",
                                    minimum=1), default=64),
                "block": F(_int("States per Chebyshev sweep.", minimum=1),
                           default=16),
                "seed": F(_int("Random seed; see ldos for the rank-count "
                               "caveat.", minimum=0), default=12345),
                "fragment_core": F(_num("Fragment control variate: core bin "
                                        "edge. Only sizes the dense fragment "
                                        "solves.", unit="Angstrom",
                                        exclusive_minimum=0.0), default=4.5),
                "fragment_buffer": F(_num("Fragment control variate: dressing "
                                          "radius; 0 disables. A deterministic "
                                          "fragment baseline replaces most of "
                                          "the map and the samples correct only "
                                          "the residual, cutting the pointwise "
                                          "noise at unchanged sweep cost. The "
                                          "buffer is the systematic accuracy "
                                          "knob. Either operator form; any "
                                          "rank count.",
                                          unit="Angstrom", minimum=0.0),
                                     default=0.0),
                "fragment_workspace_mb": F(_int("Workspace cap for the fragment build, "
                                                "MB: bounds the fragments in flight by "
                                                "the largest working set (Z panel |Z rows| "
                                                "x |F|, N x |F| for a dense Z, plus the "
                                                "nf x nf matrices). 0 = one per thread.",
                                                minimum=0), default=0),
                "fragment_coefficient": F(
                    _str("Coefficient multiplying the paired fragment "
                         "correction. 'fixed' uses coefficient one and is "
                         "exactly unbiased for every fixed fragment. "
                         "'fitted' estimates a box-local coefficient from "
                         "the same samples as a legacy robustness heuristic.",
                         choices=("fixed", "fitted")),
                    default="fitted"),
                "energy_windows": F(_int("Split the occupation into this many "
                                         "energy windows sampled separately "
                                         "from the same sweep and seeds "
                                         "(ew-sDFT). Removes cross-window "
                                         "sampling noise at no extra Chebyshev "
                                         "degree and can never increase the "
                                         "variance of the PLAIN estimator; "
                                         "measured 2.4x alone at diamond sc888. "
                                         "With fragments, window edges cut "
                                         "through bands and collapse per-window "
                                         "locality, so in gapped systems run "
                                         "fragments alone; windows are the "
                                         "delocalized (metallic) regime's "
                                         "lever, where fragments cannot help. "
                                         "1 disables.",
                                         minimum=1), default=1),
            }),
        "ac": Obj(
            "AC conductivity (spin-degenerate, S/m), plus the dielectric "
            "function, refractive index and absorption coefficient derived from "
            "it. Needs velocity.",
            {
                "label": LABEL,
                "nt": F(_int("Time steps.", minimum=1), default=200),
                "n_kpm": F(_int("KPM moments for the Fermi operator.", minimum=1),
                           default=1000),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), default=300.0),
                "mu": F(_num("Chemical potential.", unit="eV"), default=0.0),
                "axis": F(AXIS, default="abc"),
                "method": F(_str("Correlator evolution. 'concurrent' shares "
                                 "Chebyshev states across time steps (~9x "
                                 "fewer propagation matvecs); 'sequential' "
                                 "propagates stepwise and is kept as the "
                                 "independent cross-check. Both sample "
                                 "alias-free.",
                                 choices=("concurrent", "sequential")),
                            default="concurrent"),
                "block": F(_int("Concurrent only: time steps per block. Memory "
                                "holds ~4*(pi*block+50) states; the matvec "
                                "saving saturates around 128.", minimum=1)),
            }),
        "dc": Obj(
            "DC conductivity tensor at one chemical potential and temperature "
            "(spin-degenerate Kubo-Greenwood, S/m). Needs velocity.",
            {
                "label": LABEL,
                "n_moments": F(N_MOMENTS, default=1000),
                "mu": F(_num("Chemical potential.", unit="eV"), required=True),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), required=True),
                "formulation": F(FORMULATION, default="greenwood"),
                "axis": F(AXIS, default="abc"),
                "kernel": F(KERNEL, default="jackson"),
                "method": F(_str("Spectrum evaluation. 'qe' computes only the "
                                 "thermal window of energy nodes around mu "
                                 "via quasi-eigenstate accumulation: MBs of "
                                 "workspace instead of the full-grid row "
                                 "store, one pass, greenwood only. 'fftcheb' "
                                 "is the full-grid row-FFT route and the "
                                 "only one for 'bastin'.",
                                 choices=("fftcheb", "qe")), default="fftcheb"),
                "partition_sections": F(_int("Explicit row-partition count; 0 infers it.",
                                             minimum=0), default=0),
                "max_workspace_mb": F(_int("Workspace cap used to infer row partitioning; "
                                           "0 means unlimited.", minimum=0), default=0),
            }),
        "dc_spectrum": Obj(
            "Zero-temperature DC conductivity across the Chebyshev energy grid "
            "(spin-degenerate Kubo-Greenwood, S/m). Needs velocity.",
            {
                "label": LABEL,
                "n_moments": F(N_MOMENTS, default=1000),
                "formulation": F(FORMULATION, default="greenwood"),
                "axis": F(AXIS, default="abc"),
                "kernel": F(KERNEL, default="jackson"),
                "method": F(_str("Spectrum evaluation. 'fftcheb' (row FFTs, "
                                 "the default) is the right algorithm for "
                                 "the full grid; 'qe' (quasi-eigenstate "
                                 "accumulation, greenwood only) exists for "
                                 "cross-checks here and shines in the dc "
                                 "task, where only a thermal window of "
                                 "nodes is needed.",
                                 choices=("fftcheb", "qe")), default="fftcheb"),
                "partition_sections": F(_int("Explicit row-partition count; 0 infers it.",
                                             minimum=0), default=0),
                "max_workspace_mb": F(_int("Workspace cap used to infer row partitioning; "
                                           "0 means unlimited.", minimum=0), default=0),
            }),
        "occupation": Obj(
            "Electron count and band energy at one (mu, T), by adaptive tapered "
            "stochastic trace. No spin-degeneracy factor is applied.",
            {
                "label": LABEL,
                "mu": F(_num("Chemical potential.", unit="eV"), required=True),
                "temp": F(_num("Temperature; 0 selects the T=0 step.", unit="K",
                               minimum=0.0), required=True),
                "indicator_threshold": F(
                    _num("Level the a-posteriori error indicator must fall below, "
                         "relative to the sample-vector norm. Not an error "
                         "tolerance: no inequality between the indicator and the "
                         "true error is proven, and outside the method's regime "
                         "(many levels within the resolution window) it can fire "
                         "at a degree whose true error is far larger. Validate "
                         "against a fixed-degree run where that is in doubt.",
                         exclusive_minimum=0.0),
                    default=1e-4),
                "n_max": F(_int("Hard cap on the expansion degree.", minimum=1),
                           default=65536),
            }),
        "fermi_level": Obj(
            "The mu whose stochastic-trace electron count matches a target.\n\n"
            "The search runs in coefficient space at no extra matrix-vector "
            "products, so it costs about one occupation evaluation.",
            {
                "label": LABEL,
                "n_electrons": F(_num("Target electron count, with no spin factor.",
                                      exclusive_minimum=0.0), required=True),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), required=True),
                "indicator_threshold": F(
                    _num("Level the a-posteriori error indicator must fall below. "
                         "Not an error tolerance -- see the occupation task.",
                         exclusive_minimum=0.0),
                    default=1e-4),
                "n_max": F(_int("Hard cap on the expansion degree.", minimum=1),
                           default=65536),
                "mu_guess": F(_num("Anchor for the adaptive degree selection; omitted "
                                   "means the band centre.", unit="eV")),
            }),
        "msd": Obj(
            "Mean-square displacement DeltaX^2(E, t) and the diffusion "
            "coefficient D = 0.5 d/dt DeltaX^2. Gamma-point only.",
            {
                "label": LABEL,
                "n_steps": F(_int("Time steps.", minimum=1), default=200),
                "dt_fs": F(_num("Time step.", unit="fs", exclusive_minimum=0.0),
                           default=10.0),
                "n_kpm": F(_int("KPM moments resolving the energy axis.", minimum=1),
                           default=1000),
                "axis": F(_str("Direction the displacement is measured along.",
                               choices=("x", "y", "z")), default="x"),
                "method": F(_str("Evaluation kernel; both run the commutator "
                                 "velocity correlator. 'stepwise' re-expands "
                                 "the energy resolution at every output time "
                                 "(n_steps * n_kpm applies -- right for cheap "
                                 "orthogonal matvecs, and exactly "
                                 "positivity-preserving at full moments). "
                                 "'spectral' runs the segmented single-chain "
                                 "route: cost linear in the evolution window "
                                 "and independent of the output grid. 'auto' "
                                 "picks spectral whenever applies are "
                                 "expensive (non-orthogonal forms) and the "
                                 "apply-count estimate favours it; pin the "
                                 "method for sweeps that must not switch "
                                 "estimators.",
                                 choices=("auto", "stepwise", "spectral")),
                            default="auto"),
                "n_kpm_compute": F(_int("Compute only this many moments and "
                                        "extend to n_kpm by linear "
                                        "prediction; 0 disables.", minimum=0),
                                   default=0),
            }),
        "quantum_metric": Obj(
            "Longitudinal quantum metric Gxx, from the AC conductivity. Needs "
            "velocity.",
            {
                "label": LABEL,
                "nt": F(_int("Time steps.", minimum=1), default=200),
                "n_kpm": F(_int("KPM moments for the Fermi operator.", minimum=1),
                           default=1000),
                "temp": F(_num("Temperature.", unit="K", minimum=0.0), default=0.0),
                "mu": F(_num("Chemical potential.", unit="eV"), default=0.0),
                "axis": F(AXIS, default="abc"),
                "omega_min_ev": F(_num("Lower frequency cutoff of the integral.",
                                       unit="eV", minimum=0.0), default=0.0),
            }),
    },
    tag_doc="Which observable the remaining keys configure.")


# --- output ----------------------------------------------------------------

OUTPUT = Obj(
    "Where results go. Rank 0 writes, and the other ranks write nothing -- "
    "except for a result each rank holds a different piece of ('ldos'), which "
    "every rank writes into its own <label>.rankNNNNN file rather than "
    "gathering to one.",
    {
        "dir": F(_path("Output directory, created if missing. Relative paths are "
                       "resolved against the input file."), default="."),
        "prefix": F(_str("Prepended to every written file name."), default=""),
        "format": F(_str("'npz' writes one compressed archive per task, 'json' "
                         "writes arrays inline as nested lists, 'none' writes only "
                         "the summary.", choices=("npz", "json", "none")), default="npz"),
        "summary": F(_bool("Write <prefix>summary.json: the resolved input, the "
                           "derived basis, the spectral window, every scalar "
                           "result, and the shape and file of every array."),
                     default=True),
    })


# --- the whole document ----------------------------------------------------

ROOT = Obj(
    "An RSATB calculation: where the atoms are, what the Hamiltonian is, what "
    "the run should do, and which observables to compute.\n\n"
    "Unknown keys are errors, so a typo cannot quietly become a default. The "
    "one exception is a key starting with an underscore -- \"_comment\" and "
    "the like -- which is carried through uninterpreted, at any level, so an "
    "input can explain itself.",
    {
        "version": F(_int("Input-format version.", choices=(INPUT_VERSION,)),
                     default=INPUT_VERSION),
        "structure": F(STRUCTURE, required=True),
        "model": F(MODEL, required=True),
        "run": F(RUN, default={}),
        "tasks": F(Array(TASK, "Observables to compute, in order. An empty list "
                               "sets the calculation up and reports the basis and "
                               "spectral window without computing anything, which "
                               "is the cheapest way to check an input end to end."),
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
