"""The `rsatb` command line.

Six verbs, and they line up with how an input is actually written:

    rsatb doc [TOPIC]     what an input may contain, and what each key means
    rsatb template NAME   a working input to start from
    rsatb check INPUT     validate: schema, files, and cross-section consistency
    rsatb show INPUT      the input with every default filled in
    rsatb run INPUT       set up and compute
    rsatb info            what this build supports

`doc`, `template`, `check` and `show` import nothing compiled, so an input can
be written and validated on a machine that cannot run the solver -- a laptop, a
login node, or an agent with no MPI. Only `run` needs the extension.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .inputs import check as check_module
from .inputs import report, spec, templates
from .inputs.nodes import Issue


def _print_issues(issues: List[Issue], stream=sys.stderr) -> None:
    errors = [issue for issue in issues if issue.is_error]
    warnings = [issue for issue in issues if not issue.is_error]
    for issue in errors + warnings:
        print(str(issue), file=stream)
    if errors or warnings:
        print("", file=stream)
    print("{} error(s), {} warning(s)".format(len(errors), len(warnings)), file=stream)


def _fail(message: str) -> int:
    print("rsatb: {}".format(message), file=sys.stderr)
    return 1


# --- doc / schema / template ----------------------------------------------


def cmd_doc(args) -> int:
    if args.all:
        sys.stdout.write(report.render_all(args.depth))
        return 0
    if not args.topic:
        sys.stdout.write(report.overview())
        return 0

    index = spec.topics()
    if args.topic not in index:
        from .inputs.nodes import suggest
        return _fail("no documentation topic {!r}. {}".format(
            args.topic, suggest(args.topic, index.keys())))
    sys.stdout.write(report.render(args.topic, index[args.topic], args.depth))
    return 0


def cmd_schema(args) -> int:
    text = json.dumps(report.json_schema(), indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
        print("wrote {}".format(args.output))
    else:
        print(text)
    return 0


def cmd_template(args) -> int:
    if args.list or not args.name:
        print("templates\n")
        for name in templates.names():
            print("  {:<16} {}".format(name, templates.DESCRIPTIONS[name]))
        print("\n  rsatb template <name> -o input.json")
        return 0

    try:
        document = templates.get(args.name)
    except KeyError as error:
        return _fail(str(error))

    document = dict(document)
    document = {"_comment": templates.DESCRIPTIONS[args.name], **document}
    text = report.pretty_json(document)
    if args.output:
        if os.path.exists(args.output) and not args.force:
            return _fail("{} exists; pass --force to overwrite".format(args.output))
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
        print("wrote {}".format(args.output))
        print("next: rsatb check {}".format(args.output))
    else:
        print(text)
    return 0


# --- check / show ----------------------------------------------------------


def _load(path: str, check_paths: bool):
    try:
        return check_module.load_and_resolve(path, check_paths=check_paths)
    except (IOError, OSError) as error:
        raise SystemExit(_fail(str(error)))
    except ValueError as error:
        raise SystemExit(_fail(str(error)))


def cmd_check(args) -> int:
    resolved, issues = _load(args.input, check_paths=not args.no_files)

    if args.json:
        print(json.dumps({
            "input": args.input,
            "ok": not any(issue.is_error for issue in issues),
            "issues": [issue.as_dict() for issue in issues],
        }, indent=2))
        return 1 if any(issue.is_error for issue in issues) else 0

    if any(issue.is_error for issue in issues):
        print("{}: not valid\n".format(args.input), file=sys.stderr)
        _print_issues(issues)
        print("\ntry: rsatb doc <topic>", file=sys.stderr)
        return 1

    for line in check_module.summarize(resolved):
        print(line)
    print("")
    if issues:
        _print_issues(issues, sys.stdout)
        print("")
    print("{}: valid".format(args.input))
    return 0


def cmd_show(args) -> int:
    resolved, issues = _load(args.input, check_paths=False)
    if any(issue.is_error for issue in issues):
        print("{}: not valid\n".format(args.input), file=sys.stderr)
        _print_issues(issues)
        return 1
    text = report.pretty_json(resolved)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
        print("wrote {}".format(args.output))
    else:
        print(text)
    return 0


# --- run -------------------------------------------------------------------


_RULE_WIDTH = 62


def _rule(title: str) -> str:
    """A thin section rule: -- title ------- . Decoration, kept quiet."""
    head = "\u2500\u2500 {} ".format(title)
    return head + "\u2500" * max(4, _RULE_WIDTH - len(head))


def cmd_run(args) -> int:
    resolved, issues = _load(args.input, check_paths=True)
    if any(issue.is_error for issue in issues):
        print("{}: not valid\n".format(args.input), file=sys.stderr)
        _print_issues(issues)
        print("\nnothing was run. try: rsatb check {}".format(args.input), file=sys.stderr)
        return 1

    rank = 0
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        n_rank = MPI.COMM_WORLD.Get_size()
    except ImportError:
        n_rank = 1

    def log(message):
        if rank == 0:
            print(message, flush=True)

    if issues and rank == 0:
        _print_issues(issues, sys.stdout)
        print("")

    if args.dry_run:
        for line in check_module.summarize(resolved):
            log(line)
        log("\n--dry-run: validated, nothing computed")
        return 0

    from .inputs import build, tasks as task_module

    base_dir = os.path.dirname(os.path.abspath(args.input))
    if args.output_dir:
        resolved.setdefault("output", {})["dir"] = args.output_dir

    # The flag beats the file: how loud a run is belongs to the person watching
    # it, not to the input -- a "verbose": true left in a shared input makes
    # every future run noisy (which is exactly how it was found). Writing into
    # the resolved dict rather than a side channel keeps build_calculation the
    # single reader of run.log_level.
    cli_level = "debug" if args.verbose else args.log_level
    if cli_level:
        run_section = resolved.setdefault("run", {})
        run_section["log_level"] = cli_level
        run_section["verbose"] = False  # the file's shorthand must not out-shout the flag

    import time as _time
    t_start = _time.time()

    log("rsatb {}".format(_version()))
    log("")
    for line in check_module.summarize(resolved):
        log("  " + line)
    log("  ranks         {}".format(n_rank))
    log("")
    log(_rule("setup"))
    # No blank line here: the C++ profile table opens with one.

    calculation, model, options = build.build_calculation(resolved)

    basis = model.basis
    setup = {
        "model": model.name,
        "type_symbols": list(basis.type_symbols),
        "type_norb": list(basis.type_norb),
        "cutoff_radius": [float(r) for r in basis.cutoff_radius],
        "capabilities": {
            "overlap": model.capabilities.overlap,
            "velocity": model.capabilities.velocity,
            "position": model.capabilities.position,
        },
        "n_kpoint": len(options.kpoints),
        "n_rank": n_rank,
    }
    emin, emax = calculation.energy_bounds()
    setup["energy_bounds"] = [float(emin), float(emax)]

    log("")
    species = "; ".join(
        "{} \u00b7 {} orbitals \u00b7 cutoff {:.3f} \u00c5".format(symbol, norb, radius)
        for symbol, norb, radius in zip(
            basis.type_symbols, basis.type_norb, basis.cutoff_radius))
    log("  basis         {}".format(species))
    log("  bounds        [{:.6f}, {:.6f}] eV".format(emin, emax))

    task_list = resolved.get("tasks", [])
    if not task_list:
        log("\nno tasks: setup completed and the spectral window is reported above")
        return 0

    log("")
    log(_rule("tasks"))
    log("")
    def write_now(result):
        # Each task's arrays hit disk the moment it finishes: a scheduler
        # time limit during a later task must not cost the finished ones.
        if rank == 0:
            task_module.write_single_result(result, resolved, base_dir, log=log)

    try:
        results = task_module.run_tasks(calculation, task_list, options, log=log,
                                        on_task_done=write_now)
    except KeyboardInterrupt:
        # Raised from a step boundary inside the C++ loops (every rank throws
        # together, so MPI state is clean). Results of the interrupted task
        # are lost; completed tasks were not yet written, so nothing partial
        # lands on disk to be mistaken for a finished run.
        log("")
        log("interrupted \u2014 completed tasks are on disk, the one in flight is lost")
        return 130

    if rank == 0:
        log("")
        log(_rule("results"))
        log("")
        task_module.write_results(results, resolved, setup, base_dir, log=log)
        for result in results:
            if result.scalars:
                log("  {}: {}".format(result.label, ", ".join(
                    "{}={}".format(key, _short(value))
                    for key, value in result.scalars.items())))
    log("")
    log("done in {}".format(task_module.format_duration(_time.time() - t_start)))
    return 0


def _short(value):
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return value


# --- info ------------------------------------------------------------------


def cmd_info(args) -> int:
    print("rsatb {}".format(_version()))
    print("input format version {}".format(spec.INPUT_VERSION))
    print("")
    print("models        {}".format(", ".join(sorted(spec.MODEL.variants))))
    print("tasks         {}".format(", ".join(sorted(spec.TASK.variants))))
    print("radial forms  {}".format(", ".join(sorted(spec.RADIAL.variants))))
    print("H(R) formats  {}".format(", ".join(
        spec.HAMILTONIAN_FILE.fields["format"].node.choices)))
    print("")
    try:
        import _rsatb  # noqa: F401
        print("solver        available ({})".format(_rsatb.__file__))
    except ImportError as error:
        print("solver        not importable: {}".format(error))
        print("              doc, template, check and show still work")
        return 0
    try:
        from mpi4py import MPI
        print("MPI           {} rank(s) in this launch".format(MPI.COMM_WORLD.Get_size()))
    except ImportError:
        print("MPI           mpi4py not importable; runs will be serial")
    return 0


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("rsatb")
    except Exception:
        return "0.1.0"


# --- entry point -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsatb",
        description="Real-space ab-initio tight binding, driven by a JSON input.",
        epilog="start with: rsatb doc",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version="rsatb " + _version())
    subparsers = parser.add_subparsers(dest="command")

    doc = subparsers.add_parser(
        "doc", help="explain the input format, in full or one topic at a time")
    doc.add_argument("topic", nargs="?", help="e.g. model.sktb, tasks.dc, run.kpoints")
    doc.add_argument("--all", action="store_true", help="every topic at once")
    doc.add_argument("--depth", type=int, default=1,
                     help="how many levels of nesting to expand inline")
    doc.set_defaults(func=cmd_doc)

    schema = subparsers.add_parser(
        "schema", help="export the input schema as JSON Schema")
    schema.add_argument("-o", "--output", help="write here instead of stdout")
    schema.set_defaults(func=cmd_schema)

    template = subparsers.add_parser(
        "template", help="write a working input to start from")
    template.add_argument("name", nargs="?", help="which template")
    template.add_argument("-o", "--output", help="write here instead of stdout")
    template.add_argument("--list", action="store_true", help="list the templates")
    template.add_argument("--force", action="store_true", help="overwrite an existing file")
    template.set_defaults(func=cmd_template)

    check = subparsers.add_parser(
        "check", help="validate an input without running it")
    check.add_argument("input", help="the JSON input file")
    check.add_argument("--json", action="store_true",
                       help="machine-readable report on stdout")
    check.add_argument("--no-files", action="store_true",
                       help="skip the does-this-file-exist checks")
    check.set_defaults(func=cmd_check)

    show = subparsers.add_parser(
        "show", help="print the input with every default filled in")
    show.add_argument("input", help="the JSON input file")
    show.add_argument("-o", "--output", help="write here instead of stdout")
    show.set_defaults(func=cmd_show)

    run = subparsers.add_parser("run", help="set up and compute")
    run.add_argument("input", help="the JSON input file")
    run.add_argument("--dry-run", action="store_true",
                     help="validate and report what would run, then stop")
    run.add_argument("-o", "--output-dir", help="override output.dir")
    run.add_argument("--log-level", choices=("silent", "summary", "progress", "debug"),
                     help="override run.log_level from the input; verbosity is a "
                          "property of this invocation, not of the calculation, "
                          "which is why the flag wins over the file")
    run.add_argument("-v", "--verbose", action="store_true",
                     help="shorthand for --log-level debug")
    run.set_defaults(func=cmd_run)

    info = subparsers.add_parser("info", help="what this build supports")
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
