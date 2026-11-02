---
description: Every rsatb subcommand and flag — doc, template, check, show, schema, run, info — with exit codes and logging levels.
---

<p class="rs-kicker">Reference</p>

# Command line

<div class="rs-lead" markdown>
One program, seven subcommands. Where the `rsatb` command is not on `PATH`,
`python -m rsatb.cli` is the same program.
</div>

```text
rsatb [-h] [--version] {doc,schema,template,check,show,run,info} ...
```

## `rsatb doc`

Explain the input format, in full or one topic at a time.

```text
rsatb doc [--all] [--depth DEPTH] [topic]
```

| Argument | |
|---|---|
| `topic` | e.g. `model.sktb`, `tasks.dc`, `run.kpoints`; omit for the overview and topic index |
| `--all` | every topic at once |
| `--depth DEPTH` | how many levels of nesting to expand inline |

Topics: `input`, `structure`, `model`, `model.sktb`, `model.sktb.parameters`,
`model.lcao`, `model.dftb`, `model.tb_file`, `radial`, `radial.constant`,
`radial.spline`, `radial.exp_poly`, `run`, `run.kpoints`, `run.bounds`,
`run.preconditioner`, `tasks`, `tasks.<type>` for every task, `output`.

## `rsatb template`

Write a working input to start from.

```text
rsatb template [-o OUTPUT] [--list] [--force] [name]
```

| Template | |
|---|---|
| `minimal` | The shortest input that validates: a structure file, one SK hopping, a DOS |
| `sktb` | Graphene written inline with a single $p_z$ hopping; no data files, runs as written |
| `sktb-transport` | An 8×8 graphene supercell with the DC conductivity spectrum and the MSD |
| `lcao` | A precomputed Hamiltonian plus `.orb` and `.upf` files on a 2×2×2 mesh |
| `dftb` | A `.skf` parameter set from a released DFTB directory |
| `tb_file` | An orthogonal H(R) straight from a file, along a k-point line |

`-o` refuses to overwrite an existing file unless `--force` is given.

## `rsatb check`

Validate an input without running it.

```text
rsatb check [--json] [--no-files] input
```

| Flag | |
|---|---|
| `--json` | machine-readable report on stdout: `{"input", "ok", "issues": [{"severity", "path", "message", "hint"}]}` |
| `--no-files` | skip the does-this-file-exist checks |

Exit status is **0** if the input is valid (warnings allowed) and **1** if any
error was found.

## `rsatb show`

Print the input with every default filled in, exactly as `run` will see it.

```text
rsatb show [-o OUTPUT] input
```

## `rsatb schema`

Export the input schema as JSON Schema (draft 2020-12), for editors and
generators. The same file is published at
[`/rsatb.schema.json`](../rsatb.schema.json).

```text
rsatb schema [-o OUTPUT]
```

!!! tip "Editor completion"
    Point your editor's JSON-schema setting at `rsatb.schema.json` — in VS
    Code, a `"$schema"` key is not accepted by RSATB (unknown keys are
    errors), so map it through `json.schemas` in the settings instead.

## `rsatb run`

Set up and compute.

```text
rsatb run [--dry-run] [-o OUTPUT_DIR] [--log-level {silent,summary,progress,debug}] input
```

| Flag | |
|---|---|
| `--dry-run` | validate and report what would run, then stop |
| `-o`, `--output-dir` | override `output.dir` |
| `--log-level` | how much to report (default `summary`) |

Run under MPI with `mpirun -n N rsatb run input.json` or `srun rsatb run
input.json`. A run refuses to start if `check` finds an error, rather than
failing partway through setup. An empty `tasks` list sets the calculation up
and reports the derived basis and spectral window, as a quick end-to-end test.

### Log levels

| Level | Adds |
|---|---|
| `silent` | nothing (faults and input warnings still print) |
| `summary` | one line per phase, and the profile of where wall time and memory went |
| `progress` | bounded per-iteration reporting: Newton–Schulz steps, task batches |
| `debug` | everything else |

Levels are cumulative. Output goes to stdout; only rank 0 reports. From
Python, call `rsatb.set_log_level("progress")` before building a calculation.

## `rsatb info`

What this build supports: models, tasks, radial forms, Hamiltonian file
formats, whether the solver loads, and the MPI rank count.
