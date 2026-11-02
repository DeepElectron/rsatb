---
description: Every rsatb subcommand and flag — doc, template, check, show, schema, run, info — with exit codes and logging levels.
---

<p class="rs-kicker">Reference</p>

# Command line

<div class="rs-lead" markdown>
The `rsatb` command has seven subcommands. Where `rsatb` is not on `PATH`,
`python -m rsatb.cli` runs the same program.
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

Write an example input to start from.

```text
rsatb template [-o OUTPUT] [--list] [--force] [name]
```

| Template | |
|---|---|
| `minimal` | A minimal input: a structure file, one Slater-Koster hopping and a density of states |
| `sktb` | Graphene written in the input, with a single $p_z$ hopping; no data files are needed |
| `sktb-transport` | An 8×8 graphene supercell with the DC conductivity spectrum and the MSD |
| `lcao` | A DFT Hamiltonian file and its `.orb` file, on a 2×2×2 k-point mesh |
| `dftb` | A DFTB parameter set (`.skf` files) for diamond |
| `tb_file` | An orthogonal H(R) file for a 1D chain, along a k-point line |

`-o` refuses to overwrite an existing file unless `--force` is given.

## `rsatb check`

Validate an input without running it.

```text
rsatb check [--json] [--no-files] input
```

| Flag | |
|---|---|
| `--json` | machine-readable report on stdout: `{"input", "ok", "issues": [{"severity", "path", "message", "hint"}]}` |
| `--no-files` | skip the checks that files exist |

Exit status is **0** if the input is valid (warnings allowed) and **1** if any
error was found.

## `rsatb show`

Print the input with all defaults filled in, as `run` uses it.

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
    Point your editor's JSON-schema setting at `rsatb.schema.json`. In VS
    Code, use the `json.schemas` setting rather than a `"$schema"` key in the
    input, because RSATB rejects unknown keys.

## `rsatb run`

Set up the calculation and compute the tasks.

```text
rsatb run [--dry-run] [-o OUTPUT_DIR] [--log-level {silent,summary,progress,debug}] input
```

| Flag | |
|---|---|
| `--dry-run` | validate and report what would run, then stop |
| `-o`, `--output-dir` | override `output.dir` |
| `--log-level` | how much to report (default `summary`) |

Run under MPI with `mpirun -n N rsatb run input.json` or `srun rsatb run
input.json`. A run does not start if `check` finds an error. With an empty
`tasks` list, the run sets up the calculation and reports the basis and the
energy window.

### Log levels

| Level | Adds |
|---|---|
| `silent` | nothing (faults and input warnings still print) |
| `summary` | one line per phase, and the time and memory of each phase |
| `progress` | the progress of iterations: Newton–Schulz steps, task batches |
| `debug` | everything else |

Levels are cumulative. Output goes to stdout, from MPI process 0 only. From
Python, call `rsatb.set_log_level("progress")` before building a calculation.

## `rsatb info`

Print what this build supports: models, tasks, radial forms and Hamiltonian
file formats, whether the compiled solver loads, and the number of MPI
processes.
