"""DeePTB files end to end: a spin-orbit H(R), and an overlap read from a file.

The layouts are the ones DeePTB's ABACUS parser writes
(dptb/data/interfaces/abacus.py): H in h5py's single-precision compound complex
over (2 n, 2 m) blocks, spin-blocked per atom; S real over the (n, m) spatial
blocks. rsatb's basis is interleaved by spin, so a wrong reorder, a transpose
without its conjugate or a wrong spin expansion each change the matrices compared
here.
"""
import json
import pathlib

import pytest

pytest.importorskip("mpi4py.MPI")   # MPI is initialized before rsatb uses it
h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")
pytest.importorskip("rsatb")
pytest.importorskip("_rsatb")     # on the path once rsatb is imported

from rsatb import (HamiltonianFile, LcaoModel, OrbitalFiles, RSATB, SetupOptions,  # noqa: E402
                   Structure, TightBindingFileModel)
from rsatb.cli import main  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIAMOND = ROOT / "test" / "data" / "diamond"
GAAS = ROOT / "examples" / "abacus_nextham" / "data" / "gaas_soc"


def _dense(calc, which, k=0):
    rows, cols, values = calc._cpp_obj.dump_operator_coo(which, 0, k)
    size = int(max(rows.max(), cols.max())) + 1
    out = np.zeros((size, size), dtype=complex)
    np.add.at(out, (rows, cols), values)
    return out


def _spin_blocked(interleaved):
    """A (2 n, 2 m) block from rsatb's order, 2 orb + s, to DeePTB's, s n + orb."""
    n, m = interleaved.shape[0] // 2, interleaved.shape[1] // 2
    out = np.zeros_like(interleaved)
    for si in range(2):
        for oi in range(n):
            for sj in range(2):
                for oj in range(m):
                    out[si * n + oi, sj * m + oj] = interleaved[2 * oi + si, 2 * oj + sj]
    return out


def _chain_poscar(path):
    path.write_text("chain\n1.0\n3 0 0\n0 20 0\n0 0 20\nH\n1\nCartesian\n0 0 0\n")


def _random_hermitian(n, generator):
    a = generator.normal(size=(n, n)) + 1j * generator.normal(size=(n, n))
    return 0.5 * (a + a.conj().T)


@pytest.fixture
def spinor_chain(tmp_path):
    """A chain of one atom with two orbitals, H(k) known in the interleaved basis."""
    generator = np.random.default_rng(7)
    onsite = _random_hermitian(4, generator)
    hop = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))

    path = tmp_path / "hamiltonians.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("0")
        group["0_0_0_0_0"] = _spin_blocked(onsite).astype(np.complex64)
        # One direction only: R = -1 has to come back as the conjugate transpose.
        group["0_0_1_0_0"] = _spin_blocked(hop).astype(np.complex64)
    _chain_poscar(tmp_path / "chain.vasp")
    return tmp_path, onsite.astype(np.complex64), hop.astype(np.complex64)


def _chain_model(path, spin="spinorbit", norb=4):
    return TightBindingFileModel(hamiltonian=HamiltonianFile(str(path), format="deeptb"),
                                 type_symbols=["H"], type_norb=[norb], cutoff_radius=[1.6], spin=spin)


def test_a_spinor_h_is_interleaved_and_hermitian(spinor_chain):
    directory, onsite, hop = spinor_chain
    k = 0.137
    calc = RSATB(Structure.from_file(str(directory / "chain.vasp")),
                 _chain_model(directory / "hamiltonians.h5"),
                 SetupOptions(kpoints=[[k, 0.0, 0.0]], compute_bounds=False, energy_bounds=[-20.0, 20.0]))
    phase = np.exp(-2j * np.pi * k)     # rsatb's Bloch phase over the image R
    expected = onsite + hop * phase + hop.conj().T / phase
    assert np.abs(_dense(calc, "H") - expected).max() < 1e-5
    assert calc.n_states() == 4     # a spinor basis holds both spins: no factor 2


def test_a_spinor_file_read_without_spin_is_refused(spinor_chain):
    directory, _, _ = spinor_chain
    # A real reader over complex data with an imaginary part: an error, not the real part.
    with pytest.raises(RuntimeError):
        RSATB(Structure.from_file(str(directory / "chain.vasp")),
              _chain_model(directory / "hamiltonians.h5", spin="none"),
              SetupOptions(kpoints=[[0.0, 0.0, 0.0]], compute_bounds=False, energy_bounds=[-20.0, 20.0]))


def test_a_spinor_run_counts_both_spins_from_the_input(spinor_chain, tmp_path, capsys):
    directory, _, _ = spinor_chain
    document = {
        "structure": {"file": str(directory / "chain.vasp"), "supercell": [8, 1, 1]},
        "model": {"type": "tb_file", "spin": "spinorbit", "type_symbols": ["H"], "type_norb": [4],
                  "cutoff_radius": [1.6],
                  "hamiltonian": {"path": str(directory / "hamiltonians.h5"), "format": "deeptb"}},
        "run": {"nsamples": 8, "kpoints": {"mesh": [4, 1, 1]}},
        "tasks": [{"type": "dos", "dE": 0.2}, {"type": "ac", "nt": 40, "n_kpm": 200, "temp": 300.0,
                                               "mu": 0.0, "axis": "a"}],
        "output": {"dir": str(tmp_path / "out")},
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))
    assert main(["run", str(path)]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert next(r for r in summary["results"] if r["type"] == "dos")["scalars"]["n_states"] == 32
    arrays = np.load(str(tmp_path / "out" / "dos.npz"))
    assert np.trapezoid(arrays["dos"].mean(axis=0), arrays["energy"]) == pytest.approx(32, rel=1e-4)


def _diamond_overlap(path, hop_value=0.0):
    """An overlap that is plainly not op2c's: unit onsite blocks and one small hop."""
    n = 13
    with h5py.File(path, "w") as file:
        group = file.create_group("0")
        group["0_0_0_0_0"] = np.eye(n, dtype=np.float32)
        group["1_1_0_0_0"] = np.eye(n, dtype=np.float32)
        hop = hop_value * np.arange(n * n, dtype=np.float32).reshape(n, n) / (n * n)
        group["0_1_0_0_0"] = hop
        group["1_0_0_0_0"] = hop.T


def test_the_overlap_comes_from_the_file(tmp_path):
    path = tmp_path / "overlaps.h5"
    _diamond_overlap(path, hop_value=0.05)
    model = LcaoModel(hamiltonian=HamiltonianFile(str(DIAMOND / "hamiltonians.h5"), format="deeptb"),
                      orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir=str(DIAMOND)),
                      overlap=HamiltonianFile(str(path), format="deeptb"))
    calc = RSATB(Structure.from_file(str(DIAMOND / "conf.vasp")), model,
                 SetupOptions(kpoints=[[0.0, 0.0, 0.0]], compute_bounds=False, energy_bounds=[-50.0, 300.0]))
    n = 13
    hop = 0.05 * np.arange(n * n, dtype=np.float32).reshape(n, n) / (n * n)
    expected = np.block([[np.eye(n), hop], [hop.T, np.eye(n)]])
    assert np.abs(_dense(calc, "S") - expected).max() < 1e-7


def test_an_overlap_file_without_an_onsite_block_is_refused(tmp_path):
    path = tmp_path / "overlaps.h5"
    with h5py.File(path, "w") as file:
        file.create_group("0")["0_0_0_0_0"] = np.eye(13, dtype=np.float32)
    model = LcaoModel(hamiltonian=HamiltonianFile(str(DIAMOND / "hamiltonians.h5"), format="deeptb"),
                      orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir=str(DIAMOND)),
                      overlap=HamiltonianFile(str(path), format="deeptb"))
    with pytest.raises(RuntimeError, match="onsite block for atom 1"):
        RSATB(Structure.from_file(str(DIAMOND / "conf.vasp")), model,
              SetupOptions(kpoints=[[0.0, 0.0, 0.0]], compute_bounds=False, energy_bounds=[-50.0, 300.0]))


def test_a_spinor_basis_takes_the_spatial_overlap_spin_diagonal(tmp_path):
    # Ga then As in the structure's atom order; 2s2p2d is 18 orbitals, 2s2p1d is 13.
    atoms = [("As", 13), ("Ga", 18)]
    sizes = [norb for _, norb in atoms]
    generator = np.random.default_rng(3)
    onsite = [np.eye(n) + 0.01 * (lambda a: a + a.T)(generator.normal(size=(n, n))) for n in sizes]
    hop = 0.02 * generator.normal(size=(sizes[0], sizes[1]))
    path = tmp_path / "overlaps.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("0")
        group["0_0_0_0_0"] = onsite[0].astype(np.float32)
        group["1_1_0_0_0"] = onsite[1].astype(np.float32)
        group["0_1_0_0_0"] = hop.astype(np.float32)
        group["1_0_0_0_0"] = hop.T.astype(np.float32)

    model = LcaoModel(hamiltonian=HamiltonianFile(str(GAAS / "nextham" / "HR_atom_pairs.h5"), format="nextham"),
                      orbitals=OrbitalFiles(["Ga_gga_9au_60Ry_2s2p2d.orb", "As_gga_8au_60Ry_2s2p1d.orb"], dir=str(GAAS)),
                      overlap=HamiltonianFile(str(path), format="deeptb"), spin="spinorbit")
    calc = RSATB(Structure.from_file(str(GAAS / "conf.vasp")), model,
                 SetupOptions(kpoints=[[0.0, 0.0, 0.0]], compute_bounds=False, energy_bounds=[-200.0, 400.0]))
    spatial = np.block([[onsite[0], hop], [hop.T, onsite[1]]])
    expected = np.kron(spatial, np.eye(2))      # index 2 orb + s
    assert np.abs(_dense(calc, "S") - expected).max() < 1e-6


# --- a fitted model with an overlap and no orbitals --------------------------


def _nonorthogonal_chain(directory, spinor):
    """Two orbitals per atom, S(R) and H(R) known, one direction of each pair stored."""
    generator = np.random.default_rng(11)
    onsite_s = np.eye(2) + 0.02 * (lambda a: a + a.T)(generator.normal(size=(2, 2)))
    hop_s = 0.12 * generator.normal(size=(2, 2))
    dim = 4 if spinor else 2
    if spinor:
        onsite_h = _random_hermitian(4, generator)
        hop_h = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
        store = lambda block: _spin_blocked(block).astype(np.complex64)     # noqa: E731
    else:
        onsite_h = (lambda a: a + a.T)(generator.normal(size=(2, 2)))
        hop_h = generator.normal(size=(2, 2))
        store = lambda block: block.astype(np.float64)      # noqa: E731
    with h5py.File(directory / "h.h5", "w") as file:
        group = file.create_group("0")
        group["0_0_0_0_0"] = store(onsite_h)
        group["0_0_1_0_0"] = store(hop_h)
    with h5py.File(directory / "s.h5", "w") as file:
        group = file.create_group("0")
        group["0_0_0_0_0"] = onsite_s.astype(np.float64)
        group["0_0_1_0_0"] = hop_s.astype(np.float64)
    return dim


def _chain_at(directory, x, name="c.vasp"):
    path = directory / name
    path.write_text("chain\n1.0\n3 0 0\n0 20 0\n0 0 20\nH\n1\nCartesian\n{} 0.5 0.7\n".format(x))
    return path


def _mulliken_calc(directory, spinor, kpoints, x=0.0, position="mulliken"):
    model = TightBindingFileModel(
        hamiltonian=HamiltonianFile(str(directory / "h.h5"), format="deeptb"),
        type_symbols=["H"], type_norb=[4 if spinor else 2], cutoff_radius=[1.6],
        spin="spinorbit" if spinor else "none",
        overlap=HamiltonianFile(str(directory / "s.h5"), format="deeptb"), position=position)
    return RSATB(Structure.from_file(str(_chain_at(directory, x))), model,
                 SetupOptions(kpoints=kpoints, compute_bounds=False, energy_bounds=[-30.0, 30.0]))


def _velocity(calc, k):
    H, S, bond, xbar, delta = (_dense_axis(calc, w, k) for w in ("H", "S", "bond", "xbar", "delta"))
    Sinv = np.linalg.inv(S)
    return H, S, 1j * (bond - xbar @ Sinv @ H + H @ Sinv @ delta), xbar, delta


def _dense_axis(calc, which, k):
    rows, cols, values = calc._cpp_obj.dump_operator_coo(which, 0, k)
    size = int(max(rows.max(), cols.max())) + 1
    out = np.zeros((size, size), dtype=complex)
    np.add.at(out, (rows, cols), values)
    return out


@pytest.mark.parametrize("spinor", [False, True])
def test_the_mulliken_velocity_gives_the_band_velocity(tmp_path, spinor):
    """V_nn = -dE/dk, from H(R) and S(R) alone; -, as rsatb's Bloch phase is e^{-ikR}."""
    import scipy.linalg as la
    _nonorthogonal_chain(tmp_path, spinor)
    a, k, step = 3.0, 0.137, 1e-4
    calc = _mulliken_calc(tmp_path, spinor, [[k - step, 0, 0], [k, 0, 0], [k + step, 0, 0]])

    energies = [la.eigh(_dense_axis(calc, "H", i), _dense_axis(calc, "S", i), eigvals_only=True) for i in (0, 2)]
    derivative = (energies[1] - energies[0]) / (2 * step) * a / (2 * np.pi)

    H, S, velocity, xbar, delta = _velocity(calc, 1)
    # Nothing but displacements in the pieces: xbar = M[S] / 2 = -delta.
    assert np.abs(xbar + delta).max() < 1e-13
    e, c = la.eigh(H, S)
    in_bands = c.conj().T @ velocity @ c
    assert np.abs(in_bands - in_bands.conj().T).max() < 1e-12
    assert np.abs(np.diag(in_bands).real + derivative).max() < 1e-5    # the difference's truncation


@pytest.mark.parametrize("spinor", [False, True])
def test_the_mulliken_velocity_does_not_depend_on_the_origin(tmp_path, spinor):
    _nonorthogonal_chain(tmp_path, spinor)
    _, _, here, _, _ = _velocity(_mulliken_calc(tmp_path, spinor, [[0.137, 0, 0]], x=0.0), 0)
    _, _, far, _, _ = _velocity(_mulliken_calc(tmp_path, spinor, [[0.137, 0, 0]], x=41.3), 0)
    assert np.abs(here - far).max() < 1e-10


def test_a_tb_overlap_without_a_position_has_no_velocity(tmp_path):
    _nonorthogonal_chain(tmp_path, False)
    model = TightBindingFileModel(
        hamiltonian=HamiltonianFile(str(tmp_path / "h.h5"), format="deeptb"), type_symbols=["H"],
        type_norb=[2], cutoff_radius=[1.6], overlap=HamiltonianFile(str(tmp_path / "s.h5"), format="deeptb"))
    assert model.capabilities.overlap
    assert not model.capabilities.velocity
    assert not model.capabilities.position
    with pytest.raises(RuntimeError, match="built from the overlap"):
        TightBindingFileModel(hamiltonian=HamiltonianFile(str(tmp_path / "h.h5"), format="deeptb"),
                              type_symbols=["H"], type_norb=[2], cutoff_radius=[1.6], position="mulliken")


def test_a_mulliken_run_from_the_input(tmp_path, capsys):
    _nonorthogonal_chain(tmp_path, False)
    _chain_at(tmp_path, 0.0)
    document = {
        "structure": {"file": str(tmp_path / "c.vasp"), "supercell": [8, 1, 1]},
        "model": {"type": "tb_file", "type_symbols": ["H"], "type_norb": [2], "cutoff_radius": [1.6],
                  "hamiltonian": {"path": str(tmp_path / "h.h5"), "format": "deeptb"},
                  "overlap": {"path": str(tmp_path / "s.h5"), "format": "deeptb"},
                  "position": "mulliken"},
        "run": {"nsamples": 4, "kpoints": {"mesh": [4, 1, 1]}},
        "tasks": [{"type": "dos", "dE": 0.5},
                  {"type": "dc", "n_moments": 100, "mu": 0.0, "temp": 300.0, "axis": "a"}],
        "output": {"dir": str(tmp_path / "out")},
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))
    assert main(["run", str(path)]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert next(r for r in summary["results"] if r["type"] == "dos")["scalars"]["n_states"] == 32
    assert np.isfinite(np.load(str(tmp_path / "out" / "dc.npz"))["sigma"]).all()
