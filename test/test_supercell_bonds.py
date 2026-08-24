"""The atomic graph must not depend on how a cell was written.

Regression test for a neighbour-list defect that silently dropped bonds. An
atom whose fractional coordinate wrapped to exactly 1.0 -- which a tiny
negative coordinate does, because `s - floor(s)` rounds up -- was binned at the
near face of the cell while its wrapped position sat at the far face, so the
fixed-radius bin search looked a whole lattice vector away from its neighbours
and never found them. No error was raised: the calculation ran and returned a
narrower spectrum.

Graphene is the sharp case. Its conventional basis puts the second carbon at
fractional (-1/3, 2/3), and tiling that cell lands other atoms on exactly the
coordinates that round. Every carbon has three neighbours at 1.42 A whatever
the supercell, so the Gamma-point eigenvalues are +-3|t| for all of them; before
the fix a 2x2 came out at 7.60 and a 4x4 at 7.81 instead of 8.10.

    mpirun -n 1 python -m pytest test/test_supercell_bonds.py

Run on one rank: the point is the graph, not the parallel layout.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rsatb import (PreconditionerOptions, RSATB, SetupOptions, SkParameters,
                       SktbModel, Structure, sk_spline)
except ImportError as error:  # the compiled extension is not built here
    pytest.skip("rsatb is not importable: {}".format(error), allow_module_level=True)

A_CC = 1.42                    # carbon-carbon bond length, Angstrom
HOPPING = -2.7                 # eV
BOUNDS_PADDING = 0.5           # eV, added outside the extreme eigenvalues
ANALYTIC_HALF_WIDTH = 3 * abs(HOPPING)   # three neighbours at Gamma: 8.1 eV


def graphene_primitive():
    a = 3.0 ** 0.5 * A_CC
    cell = [[a, 0.0, 0.0], [0.5 * a, 1.5 * A_CC, 0.0], [0.0, 0.0, 20.0]]
    # The conventional basis: the second carbon at fractional (-1/3, 2/3).
    positions = [[0.0, 0.0, 10.0], [0.0, A_CC, 10.0]]
    return positions, [6, 6], cell


def pz_model():
    return SktbModel(
        basis={"C": ["pz"]},
        hamiltonian=SkParameters(
            onsite={"C": {"pz": 0.0}},
            pairs={("C", "C"): {"pp_pi": sk_spline([0.0, 1.6], [HOPPING, HOPPING], 1.6)}},
        ),
    )


def spectral_half_width(positions, numbers, cell, model, supercell=(1, 1, 1)):
    structure = Structure.from_arrays(
        pos=[v for p in positions for v in p],
        z=numbers,
        cell=[v for row in cell for v in row],
        pbc=[True, True, False],
        supercell=supercell,
    )
    calculation = RSATB(
        structure,
        model,
        SetupOptions(kpoints=[[0.0, 0.0, 0.0]], nsamples=1,
                     preconditioner=PreconditionerOptions()),
        npar=1,
    )
    emin, emax = calculation.energy_bounds()
    return max(abs(emin), abs(emax)) - BOUNDS_PADDING


@pytest.mark.parametrize("supercell", [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]])
def test_a_graphene_supercell_keeps_every_bond(supercell):
    positions, numbers, cell = graphene_primitive()
    half_width = spectral_half_width(
        positions, numbers, cell, pz_model(), supercell=supercell)
    assert half_width == pytest.approx(ANALYTIC_HALF_WIDTH, abs=1e-6), (
        "{}: |E|max is {:.4f}, not the analytic 3|t| = {:.4f}; the supercell is "
        "missing bonds".format(supercell, half_width, ANALYTIC_HALF_WIDTH))


def test_the_result_does_not_depend_on_where_the_basis_atom_is_written():
    """Two spellings of the same crystal must give the same spectrum.

    The second carbon at (0, 1.42) is at fractional (-1/3, 2/3), outside the
    cell; at (1.22975, 0.71) it is the equivalent site inside it. Coordinates
    are used as written, so nothing may hinge on which one a caller picked.
    """
    a = 3.0 ** 0.5 * A_CC
    cell = [[a, 0.0, 0.0], [0.5 * a, 1.5 * A_CC, 0.0], [0.0, 0.0, 20.0]]
    outside = [[0.0, 0.0, 10.0], [0.0, A_CC, 10.0]]
    inside = [[0.0, 0.0, 10.0], [0.5 * a, 0.5 * A_CC, 10.0]]

    model = pz_model()
    for supercell in ([1, 1, 1], [3, 3, 1]):
        widths = []
        for basis in (outside, inside):
            widths.append(spectral_half_width(
                basis, [6, 6], cell, model, supercell=supercell))
        assert widths[0] == pytest.approx(widths[1], abs=1e-6), (
            "{}: {:.4f} vs {:.4f} for the same crystal".format(supercell, *widths))
        assert widths[0] == pytest.approx(ANALYTIC_HALF_WIDTH, abs=1e-6)
