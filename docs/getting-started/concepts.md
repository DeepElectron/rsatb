---
description: The five sections of an RSATB input file, what each controls, and the few rules every input follows.
---

<p class="rs-kicker">Get started</p>

# How an input works

<div class="rs-lead" markdown>
An RSATB calculation is described by one JSON file with up to five
sections.
</div>

## The five sections

```json
{
  "structure": { },   // atomic structure
  "model":     { },   // Hamiltonian
  "run":       { },   // k-points, random vectors, numerical settings
  "tasks":     [ ],   // calculations, in order
  "output":    { }    // output directory and format
}
```

Only `structure` and `model` are required; the other sections have default values.

<div class="grid cards rs-grid-2" markdown>

-   :material-atom-variant:{ .lg } **[`structure`](../reference/input/structure.md)**

    The atoms are read from a structure file (POSCAR, CIF, XYZ) or written
    in the input. `supercell` repeats the cell.

-   :material-function-variant:{ .lg } **[`model`](../reference/input/model/index.md)**

    The Hamiltonian is given by Slater-Koster parameters, a DFT Hamiltonian
    file, DFTB parameter files or a tight-binding file.

-   :material-tune-variant:{ .lg } **[`run`](../reference/input/run.md)**

    This section sets the k-points, the number of random vectors and other
    numerical settings.

-   :material-format-list-checks:{ .lg } **[`tasks`](../reference/input/tasks/index.md)**

    The calculations to run, in order, for example a density of states, a
    conductivity or a real-space map.

</div>

A run validates the input, builds the Hamiltonian once, and then computes
the tasks in order, so several tasks in one input share the setup.

## Rules

- **Unknown keys are errors.** `rsatb check` reports them with a suggested
  correction.
- **Comments are allowed.** Any key starting with `_`, such as `"_comment"`,
  is ignored.
- **Paths are relative to the input file,** so an input and its data can be
  moved together.
- **Units are fixed:** eV, Å, K and fs.

## Tasks available for each model

Some tasks need operators that not every model type provides. `rsatb check`
reports a task that the model cannot run.

| Model | Density of states | Conductivity & optics | Real-space maps |
|---|:---:|:---:|:---:|
| `sktb` (no overlap) | ✓ | ✓ | — |
| `lcao` (DFT) | ✓ | ✓ | ✓ |
| `dftb` | ✓ | — | — |
| `tb_file` | ✓ | ✓ | — |

The reasons are given in [Choose a model](../guides/choose-model.md).

## Commands

| Command | What it does |
|---|---|
| `rsatb template NAME -o FILE` | write an example input |
| `rsatb check FILE` | validate an input |
| `rsatb show FILE` | print the input with every default filled in |
| `rsatb run FILE` | run the calculation |
| `rsatb doc [TOPIC]` | describe the input options |

All options are listed in [Command line](../reference/cli.md).
