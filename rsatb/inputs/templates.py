"""Ready-to-edit input files, one per model family.

`rsatb template <name>` writes one of these. They are validated by the test
suite against the same schema `rsatb check` uses, so a template is never a
stale example of an input that no longer parses.

The `_comment` key at the top of each template is the one key the schema
ignores by name: it exists so a written-out input can carry a note without
being rejected.
"""

from __future__ import annotations

from typing import Dict

A_CC = 1.42          # carbon-carbon bond length, Angstrom
A_GRAPHENE = 2.4595  # sqrt(3) * A_CC

GRAPHENE = {
    "positions": [[0.0, 0.0, 10.0], [0.0, A_CC, 10.0]],
    "symbols": ["C", "C"],
    "cell": [[A_GRAPHENE, 0.0, 0.0],
             [0.5 * A_GRAPHENE, 1.5 * A_CC, 0.0],
             [0.0, 0.0, 20.0]],
    "pbc": [True, True, False],
}

PZ_HOPPING = {
    "onsite": {"C": {"pz": 0.0}},
    "pairs": {"C-C": {"pp_pi": {"kind": "constant", "value": -2.7, "r_cut": 1.6}}},
}


TEMPLATES: Dict[str, dict] = {
    "minimal": {
        "structure": {"file": "structure.vasp"},
        "model": {
            "type": "sktb",
            "basis": {"C": ["pz"]},
            "hamiltonian": PZ_HOPPING,
        },
        "tasks": [{"type": "dos"}],
    },

    "sktb": {
        "structure": dict(GRAPHENE),
        "model": {
            "type": "sktb",
            "basis": {"C": ["pz"]},
            "hamiltonian": PZ_HOPPING,
        },
        "run": {
            "npar": 1,
            "nsamples": 64,
            "kpoints": [[0.0, 0.0, 0.0]],
        },
        "tasks": [
            {"type": "energy_bounds"},
            {"type": "dos", "dE": 0.1},
        ],
        "output": {"dir": "results", "format": "npz"},
    },

    "sktb-transport": {
        "structure": dict(GRAPHENE, supercell=[8, 8, 1]),
        "model": {
            "type": "sktb",
            "basis": {"C": ["pz"]},
            "hamiltonian": {
                "onsite": {"C": {"pz": 0.0}},
                "pairs": {
                    "C-C": {
                        "pp_pi": {"kind": "exp_poly", "coeffs": [-2.7], "lambda": 0.0,
                                  "r_on": 1.5, "r_cut": 1.7},
                    },
                },
            },
        },
        "run": {
            "nsamples": 1,
            "kpoints": [[0.0, 0.0, 0.0]],
            "bounds": {"mode": "fixed", "energy": [-9.0, 9.0]},
        },
        "tasks": [
            {"type": "dos", "dE": 0.05},
            {"type": "dc_spectrum", "n_moments": 1024, "kernel": "costaper"},
            {"type": "msd", "n_steps": 100, "dt_fs": 10.0, "n_kpm": 1024, "axis": "x"},
        ],
        "output": {"dir": "results", "prefix": "graphene_"},
    },

    "lcao": {
        "structure": {"file": "data/diamond/conf.vasp"},
        "model": {
            "type": "lcao",
            "hamiltonian": {"path": "data/diamond/hamiltonians.h5",
                            "format": "deeptb"},
            "orbitals": {"names": ["C_gga_7au_100Ry_2s2p1d.orb"], "dir": "data/diamond"},
            "spin": "none",
        },
        "run": {
            "npar": 1,
            "nsamples": 200,
            "kpoints": {"mesh": [2, 2, 2]},
            "preconditioner": {"operator_form": "sandwich"},
        },
        "tasks": [{"type": "dos", "dE": 0.2}],
        "output": {"dir": "results"},
    },

    "dftb": {
        "structure": {
            "positions": [[0.0, 0.0, 0.0], [0.8918, 0.8918, 0.8918]],
            "symbols": ["C", "C"],
            "cell": [[0.0, 1.7835, 1.7835],
                     [1.7835, 0.0, 1.7835],
                     [1.7835, 1.7835, 0.0]],
            "pbc": [True, True, True],
        },
        "model": {
            "type": "dftb",
            "skf_dir": "skf/mio-1-1",
            "basis": {"C": ["s", "p"]},
        },
        "run": {"nsamples": 32, "kpoints": {"mesh": [4, 4, 4]}},
        "tasks": [{"type": "dos", "dE": 0.1}],
        "output": {"dir": "results"},
    },

    "tb_file": {
        "structure": {
            "positions": [[0.0, 0.0, 0.0]],
            "symbols": ["C"],
            "cell": [[2.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
            "pbc": [True, False, False],
        },
        "model": {
            "type": "tb_file",
            "hamiltonian": {"path": "chain.h5", "format": "deeptb"},
            "type_symbols": ["C"],
            "type_norb": [1],
            "cutoff_radius": [1.1],
        },
        "run": {"nsamples": 1, "kpoints": {"line": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                                           "points_per_segment": 11}},
        "tasks": [{"type": "dos", "dE": 0.2}],
        "output": {"dir": "results"},
    },
}

DESCRIPTIONS = {
    "minimal": "The shortest input that validates: a structure file, one SK "
               "hopping, a DOS.",
    "sktb": "Graphene written inline with a single pz hopping. Needs no data "
            "files at all, so it runs as written.",
    "sktb-transport": "An 8x8 graphene supercell with the DC conductivity "
                      "spectrum and the mean-square displacement.",
    "lcao": "A precomputed Hamiltonian plus .orb and .upf files, on a 2x2x2 "
            "mesh. Paths match the diamond test data.",
    "dftb": "A .skf parameter set from a released DFTB directory.",
    "tb_file": "An orthogonal H(R) straight from a file, along a k-point line.",
}


def get(name: str) -> dict:
    """Returns one template by name."""
    if name not in TEMPLATES:
        raise KeyError("No such template: {!r}. Available: {}".format(
            name, ", ".join(sorted(TEMPLATES))))
    return TEMPLATES[name]


def names():
    """Returns the template names, in the order they are worth reading."""
    return ["minimal", "sktb", "sktb-transport", "lcao", "dftb", "tb_file"]
