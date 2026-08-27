"""Tests for the JSON input interface and the CLI that reads it.

Everything here runs without the compiled extension and without MPI, which is
the property `rsatb check`, `rsatb doc`, `rsatb schema` and `rsatb template`
are supposed to have. The one test that computes is marked and skipped when
the extension is not importable.

    pytest test/test_json_input.py
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsatb.cli import main
from rsatb.inputs import check, report, spec, templates
from rsatb.inputs.nodes import ERROR, WARNING


def errors(issues):
    return [issue for issue in issues if issue.is_error]


def warnings(issues):
    return [issue for issue in issues if not issue.is_error]


def paths(issues):
    return [issue.path for issue in issues]


# --- templates -------------------------------------------------------------


@pytest.mark.parametrize("name", templates.names())
def test_every_template_validates_against_the_schema(name):
    """A template that no longer parses is a stale example, so this must hold."""
    _, issues = check.resolve(copy.deepcopy(templates.get(name)), check_paths=False)
    assert errors(issues) == [], [str(i) for i in errors(issues)]


@pytest.mark.parametrize("name", templates.names())
def test_every_template_has_a_description(name):
    assert templates.DESCRIPTIONS[name]


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples")
EXAMPLES = sorted(f for f in os.listdir(EXAMPLE_DIR) if f.endswith(".json"))


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_input_is_well_formed(name):
    """Structure only: 06 deliberately points at a moire cell the user supplies."""
    document = check.load(os.path.join(EXAMPLE_DIR, name))
    _, issues = check.resolve(document, check_paths=False)
    assert errors(issues) == [], [str(i) for i in errors(issues)]


@pytest.mark.parametrize("name", [n for n in EXAMPLES if not n.startswith("06")])
def test_the_self_contained_examples_pass_a_full_check(name):
    """Everything 01-05 read is in examples/data, so they run as written."""
    _, issues = check.load_and_resolve(os.path.join(EXAMPLE_DIR, name))
    assert errors(issues) == [], [str(i) for i in errors(issues)]


def test_a_template_is_carried_through_the_cli_and_back(tmp_path, capsys):
    target = str(tmp_path / "input.json")
    assert main(["template", "sktb", "-o", target]) == 0
    assert main(["check", target]) == 0


# --- schema and defaults ---------------------------------------------------


MINIMAL = {
    "structure": {"positions": [[0.0, 0.0, 0.0]], "symbols": ["C"],
                  "cell": [[2.0, 0, 0], [0, 20, 0], [0, 0, 20]]},
    "model": {
        "type": "sktb",
        "basis": {"C": ["pz"]},
        "hamiltonian": {"onsite": {"C": {"pz": 0.0}},
                        "pairs": {"C-C": {"pp_pi": {"kind": "constant",
                                                    "value": -1.0, "r_cut": 2.5}}}},
    },
}


def resolve(overrides=None, **kwargs):
    document = copy.deepcopy(MINIMAL)
    for key, value in (overrides or {}).items():
        document[key] = value
    return check.resolve(document, check_paths=False, **kwargs)


def test_defaults_are_filled_in():
    resolved, issues = resolve()
    assert errors(issues) == []
    assert resolved["version"] == spec.INPUT_VERSION
    assert resolved["run"]["npar"] == 1
    assert resolved["run"]["kpoints"] == [[0.0, 0.0, 0.0]]
    assert resolved["run"]["bounds"]["mode"] == "auto"
    assert resolved["run"]["preconditioner"]["maxiter"] == 20
    assert resolved["output"]["format"] == "npz"
    assert resolved["tasks"] == []


def test_an_unknown_key_is_an_error_not_a_default():
    _, issues = resolve({"run": {"kpoint": [[0, 0, 0]]}})
    assert "run.kpoint" in paths(errors(issues))
    assert "kpoints" in errors(issues)[0].hint


def test_a_misspelled_enum_suggests_the_right_one():
    _, issues = resolve({"model": dict(MINIMAL["model"], spin="colinear")})
    assert "collinear" in errors(issues)[0].hint


def test_an_underscore_key_is_carried_through_uninterpreted():
    resolved, issues = resolve({"_comment": "why this run exists"})
    assert errors(issues) == []
    assert resolved["_comment"] == "why this run exists"


def test_a_number_where_a_list_belongs_is_reported_by_path():
    _, issues = resolve({"run": {"kpoints": [[0.0, 0.0]]}})
    assert paths(errors(issues)) == ["run.kpoints[0]"]


def test_required_keys_are_named_when_missing():
    document = copy.deepcopy(MINIMAL)
    del document["model"]["basis"]
    _, issues = check.resolve(document, check_paths=False)
    assert "basis" in errors(issues)[0].message


# --- structure -------------------------------------------------------------


def test_a_structure_is_either_a_file_or_inline_but_not_both():
    _, issues = resolve({"structure": dict(MINIMAL["structure"], file="a.vasp")})
    assert any("mutually exclusive" in issue.message for issue in errors(issues))


def test_inline_positions_need_a_cell():
    structure = {k: v for k, v in MINIMAL["structure"].items() if k != "cell"}
    _, issues = resolve({"structure": structure})
    assert any("cell" in issue.message for issue in errors(issues))


def test_symbol_and_position_counts_must_agree():
    _, issues = resolve({"structure": dict(MINIMAL["structure"], symbols=["C", "C"])})
    assert "structure.symbols" in paths(errors(issues))


def test_a_cell_key_is_rejected_alongside_a_structure_file():
    _, issues = resolve({"structure": {"file": "a.vasp", "cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}})
    assert "structure.cell" in paths(errors(issues))


# --- Slater-Koster parameters ---------------------------------------------


def test_a_pair_key_must_name_two_species():
    model = copy.deepcopy(MINIMAL["model"])
    model["hamiltonian"]["pairs"] = {"CC": model["hamiltonian"]["pairs"]["C-C"]}
    _, issues = resolve({"model": model})
    assert any("species pair" in issue.message for issue in errors(issues))


def test_a_channel_name_must_be_a_shell_pair_and_bond_type():
    model = copy.deepcopy(MINIMAL["model"])
    model["hamiltonian"]["pairs"]["C-C"] = {"pppi": {"kind": "constant", "value": -1.0,
                                                     "r_cut": 2.5}}
    _, issues = resolve({"model": model})
    assert any("Slater-Koster channel" in issue.message for issue in errors(issues))


def test_a_species_in_the_parameters_must_be_in_the_basis():
    model = copy.deepcopy(MINIMAL["model"])
    model["hamiltonian"]["pairs"]["C-H"] = model["hamiltonian"]["pairs"]["C-C"]
    _, issues = resolve({"model": model})
    assert any("'H' is not in 'basis'" in issue.message for issue in errors(issues))


def test_a_spline_needs_matching_r_and_values():
    model = copy.deepcopy(MINIMAL["model"])
    model["hamiltonian"]["pairs"]["C-C"]["pp_pi"] = {
        "kind": "spline", "r": [1.0, 2.0], "values": [-1.0], "r_cut": 2.5}
    _, issues = resolve({"model": model})
    assert any("entries" in issue.message for issue in errors(issues))


def test_a_spline_held_constant_out_to_its_cutoff_warns():
    """The clamp beyond the last tabulated point is silent in C++, so it warns here."""
    model = copy.deepcopy(MINIMAL["model"])
    model["hamiltonian"]["pairs"]["C-C"]["pp_pi"] = {
        "kind": "spline", "r": [1.0, 1.5], "values": [-2.7, -2.7], "r_cut": 5.0}
    _, issues = resolve({"model": model})
    assert errors(issues) == []
    assert any("held at" in issue.message for issue in warnings(issues))


# --- capabilities and tasks ------------------------------------------------


def test_capabilities_follow_the_model_family():
    assert check.capabilities({"type": "sktb"})["velocity"] is True
    assert check.capabilities({"type": "sktb", "overlap": {}})["velocity"] is False
    assert check.capabilities({"type": "dftb"}) == {"overlap": True, "velocity": False,
                                                    "position": False}
    assert check.capabilities({"type": "lcao"})["velocity"] is True
    assert check.capabilities(
        {"type": "lcao", "pseudopotentials": {"names": ["C.upf"]}})["velocity"] is True


def test_a_task_needing_velocity_is_refused_when_the_model_has_none():
    model = copy.deepcopy(MINIMAL["model"])
    model["overlap"] = {"onsite": {"C": {"pz": 1.0}}}
    _, issues = resolve({"model": model, "tasks": [{"type": "ac"}]})
    assert any("velocity operator" in issue.message for issue in errors(issues))


def test_msd_is_refused_on_a_non_orthogonal_model_without_position():
    # The periodic MSD now runs on non-orthogonal bases through the commutator
    # route -- but that route needs the position/dipole integrals, which a
    # non-orthogonal Slater-Koster model does not have. So the refusal moves
    # from "orthogonal only" to the position capability.
    model = copy.deepcopy(MINIMAL["model"])
    model["overlap"] = {"onsite": {"C": {"pz": 1.0}}}
    _, issues = resolve({"model": model, "tasks": [{"type": "msd"}]})
    assert any("position operator" in issue.message for issue in errors(issues))


def test_msd_is_allowed_on_lcao():
    # LCAO carries the dipole integrals, so the non-orthogonal periodic MSD is
    # available (doc/position_commutator.md section 8).
    lcao = {"type": "lcao", "hamiltonian": {"path": "h.h5"},
            "orbitals": {"names": ["C.orb"]}}
    _, issues = resolve({"model": lcao, "tasks": [{"type": "msd"}]})
    assert not any("position operator" in issue.message or "orthogonal" in issue.message
                   for issue in errors(issues))


def test_msd_spectral_method_is_refused_without_velocity():
    # The spectral estimator integrates velocity correlations, so the velocity
    # requirement follows the method, not the task kind; 'auto' on the same
    # model is fine because it falls back to the stepwise route.
    model = copy.deepcopy(MINIMAL["model"])
    model["overlap"] = {"onsite": {"C": {"pz": 1.0}}}
    _, issues = resolve({"model": model,
                         "tasks": [{"type": "msd", "method": "spectral"}]})
    assert any("commutator velocity" in issue.message for issue in errors(issues))

    _, issues = resolve({"model": model,
                         "tasks": [{"type": "msd", "method": "auto"}]})
    assert not any("commutator velocity" in issue.message for issue in errors(issues))


def test_duplicate_result_names_are_refused():
    _, issues = resolve({"tasks": [{"type": "dos"}, {"type": "dos", "dE": 0.05}]})
    assert any("already used" in issue.message for issue in errors(issues))


def test_labels_separate_two_runs_of_the_same_task():
    _, issues = resolve({"tasks": [{"type": "dos", "label": "coarse", "dE": 0.4},
                                   {"type": "dos", "label": "fine", "dE": 0.04}]})
    assert errors(issues) == []


def test_deeptb_tiling_is_allowed_for_a_file_backed_hamiltonian():
    document = {
        "structure": {"file": "a.vasp", "supercell": [2, 2, 1]},
        "model": {"type": "tb_file",
                  "hamiltonian": {"path": "h.h5"},
                  "type_symbols": ["C"], "type_norb": [1], "cutoff_radius": [1.1]},
    }
    _, issues = check.resolve(document, check_paths=False)
    assert errors(issues) == []


def test_bounds_mode_fixed_needs_an_energy_window():
    _, issues = resolve({"run": {"bounds": {"mode": "fixed"}}})
    assert any("emin, emax" in issue.message for issue in errors(issues))


def test_bounds_energy_without_fixed_mode_warns_rather_than_being_ignored():
    _, issues = resolve({"run": {"bounds": {"energy": [-1.0, 1.0]}}})
    assert errors(issues) == []
    assert "run.bounds.energy" in paths(warnings(issues))


# --- paths -----------------------------------------------------------------


def test_paths_resolve_against_the_input_file(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "chain.h5").write_text("")
    (tmp_path / "cell.vasp").write_text("")
    document = {
        "structure": {"file": "cell.vasp"},
        "model": {"type": "tb_file", "hamiltonian": {"path": "data/chain.h5"},
                  "type_symbols": ["C"], "type_norb": [1], "cutoff_radius": [1.1]},
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))
    resolved, issues = check.load_and_resolve(str(path))
    assert errors(issues) == []
    assert resolved["model"]["hamiltonian"]["path"] == str(data / "chain.h5")


def test_a_missing_file_is_an_error_naming_the_key(tmp_path):
    path = tmp_path / "input.json"
    document = copy.deepcopy(MINIMAL)
    document["structure"] = {"file": "absent.vasp"}
    path.write_text(json.dumps(document))
    _, issues = check.load_and_resolve(str(path))
    assert "structure.file" in paths(errors(issues))


# --- generated k-points and energies ---------------------------------------


def test_a_mesh_expands_to_explicit_kpoints():
    from rsatb.inputs.build import build_kpoints
    points = build_kpoints({"mesh": [2, 2, 1], "shift": [0.0, 0.0, 0.0]})
    assert len(points) == 4
    assert [0.0, 0.0, 0.0] in points
    assert [0.5, 0.5, 0.0] in points


def test_a_line_does_not_repeat_its_shared_vertices():
    from rsatb.inputs.build import build_kpoints
    points = build_kpoints({"line": [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]],
                            "points_per_segment": 3})
    assert len(points) == 5
    assert points[0] == [0.0, 0.0, 0.0]
    assert points[-1] == [0.5, 0.5, 0.0]


def test_an_energy_grid_expands_to_its_endpoints():
    from rsatb.inputs.build import build_energies
    energies = build_energies({"min": -1.0, "max": 1.0, "n": 5})
    assert energies[0] == -1.0 and energies[-1] == 1.0 and len(energies) == 5


# --- documentation ---------------------------------------------------------


def test_every_topic_renders():
    for name, node in spec.topics().items():
        text = report.render(name, node)
        assert name in text and len(text) > 100


def test_the_overview_lists_every_topic():
    text = report.overview()
    for name in spec.topics():
        assert name in text


def test_the_exported_schema_is_valid_json_schema():
    schema = report.json_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    json.dumps(schema)  # must be serializable
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for name in templates.names():
        assert list(validator.iter_errors(templates.get(name))) == [], name


def test_short_numeric_rows_stay_on_one_line():
    text = report.pretty_json({"cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]})
    assert "[1.0, 0.0, 0.0]" in text


# --- CLI -------------------------------------------------------------------


def test_check_exits_non_zero_on_an_invalid_input(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"structure": {"file": "a.vasp"}}))
    assert main(["check", str(path)]) == 1


def test_check_json_reports_machine_readably(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": {"type": "nope"}}))
    assert main(["check", "--json", str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert {"severity", "path", "message", "hint"} <= set(payload["issues"][0])
    assert "model.type" in [issue["path"] for issue in payload["issues"]]


def test_show_prints_the_resolved_input(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(MINIMAL))
    assert main(["show", str(path)]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["run"]["nsamples"] == 1


def test_doc_names_an_unknown_topic(capsys):
    assert main(["doc", "model.sktbb"]) == 1
    assert "model.sktb" in capsys.readouterr().err


def test_info_runs_without_the_solver(capsys):
    assert main(["info"]) == 0
    assert "input format version" in capsys.readouterr().out


def test_star_import_resolves_every_lazy_name():
    """`from rsatb import *` walks __all__ through the lazy hook; it must not recurse."""
    pytest.importorskip("_rsatb")
    namespace = {}
    exec("from rsatb import *", namespace)  # noqa: S102 - that is the thing under test
    import rsatb
    for name in rsatb.__all__:
        assert name in namespace, name


# --- end to end ------------------------------------------------------------


@pytest.mark.order(-1)
def test_a_graphene_dos_runs_from_json(tmp_path, capsys):
    """The analytic check: three neighbours at 1.42 A give 3|t| = 8.1 eV."""
    pytest.importorskip("_rsatb")
    document = copy.deepcopy(templates.get("sktb"))
    document["run"]["nsamples"] = 8
    document["tasks"] = [{"type": "energy_bounds"}, {"type": "dos", "dE": 0.27}]
    document["output"] = {"dir": str(tmp_path / "out")}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))

    assert main(["run", str(path)]) == 0

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    bounds = summary["setup"]["energy_bounds"]
    assert bounds[0] == pytest.approx(-8.6, abs=1e-6)   # -(3|t| + 0.5)
    assert bounds[1] == pytest.approx(8.6, abs=1e-6)

    numpy = pytest.importorskip("numpy")
    arrays = numpy.load(str(tmp_path / "out" / "dos.npz"))
    assert arrays["dos"].shape[0] == 1                  # one k-point
    assert arrays["energy"].shape[0] == arrays["dos"].shape[1]


@pytest.mark.parametrize("family,example", [
    ("sktb", "01_sktb_graphene.json"),
    ("dftb", "03_dftb_skf.json"),
    ("tb_file", "04_tb_file_chain.json"),
    pytest.param("lcao", "02_lcao_diamond.json", marks=pytest.mark.mpi),
])
def test_capabilities_match_the_compiled_models(family, example):
    """The Python capability table must agree with each model's own declaration.

    `check.capabilities` restates, in pure Python, the operator-source table
    every C++ model declares -- it has to, because `rsatb check` runs where the
    extension does not. This is what keeps the restatement honest: build the
    model and compare.

    LCAO is marked `mpi` because `LcaoModel::Create` calls MPI_Comm_rank, which
    aborts if MPI was never initialized; importing mpi4py does that.
    """
    pytest.importorskip("_rsatb")
    if family == "lcao":
        pytest.importorskip("mpi4py").MPI  # initializes MPI
    from rsatb.inputs.build import build_model

    resolved, issues = check.load_and_resolve(os.path.join(EXAMPLE_DIR, example))
    assert errors(issues) == [], example
    assert resolved["model"]["type"] == family

    compiled = build_model(resolved["model"]).capabilities
    assert check.capabilities(resolved["model"]) == {
        "overlap": compiled.overlap,
        "velocity": compiled.velocity,
        "position": compiled.position,
    }, family


def test_an_sktb_overlap_turns_off_velocity_in_both_places():
    """The orthogonality hinge, checked against the compiled model too.

    An SKTB model with an overlap is non-orthogonal, so the generic commutator
    velocity and diagonal position are no longer correct and the model stops
    offering them. Both the table and `operator_sources()` have to say so.
    """
    pytest.importorskip("_rsatb")
    from rsatb.inputs.build import build_model

    model = copy.deepcopy(MINIMAL["model"])
    model["overlap"] = {
        "onsite": {"C": {"pz": 1.0}},
        "pairs": {"C-C": {"pp_pi": {"kind": "constant", "value": 0.1, "r_cut": 2.5}}},
    }
    resolved, issues = check.resolve(dict(MINIMAL, model=model), check_paths=False)
    assert errors(issues) == []

    compiled = build_model(resolved["model"]).capabilities
    assert compiled.overlap and not compiled.velocity and not compiled.position
    assert check.capabilities(resolved["model"]) == {
        "overlap": True, "velocity": False, "position": False}
