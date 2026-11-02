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
  "structure": { },   // where the atoms are
  "model":     { },   // what the Hamiltonian is
  "run":       { },   // how to run: k-points, sampling, ...
  "tasks":     [ ],   // what to compute
  "output":    { }    // where to write results
}
```

Only `structure` and `model` are required; the others have sensible defaults.

<div class="grid cards rs-grid-2" markdown>

-   :material-atom-variant:{ .lg } **[`structure`](../reference/input/structure.md)**

    A structure file (POSCAR, CIF, XYZ) or atoms written inline. Add
    `supercell` to repeat the cell.

-   :material-function-variant:{ .lg } **[`model`](../reference/input/model/index.md)**

    The Hamiltonian: hand-written parameters, DFT output, DFTB files, or a
    tight-binding file.

-   :material-tune-variant:{ .lg } **[`run`](../reference/input/run.md)**

    Which k-points, how many random vectors, and a few numerical settings.

-   :material-format-list-checks:{ .lg } **[`tasks`](../reference/input/tasks/index.md)**

    A list of what to compute, in order — a density of states, a
    conductivity, a map…

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

## Which model can compute what

Some observables need extra ingredients from the model. `rsatb check` tells
you if a request isn't possible.

| Model | Density of states | Conductivity & optics | Real-space maps |
|---|:---:|:---:|:---:|
| `sktb` (no overlap) | ✓ | ✓ | — |
| `lcao` (DFT) | ✓ | ✓ | ✓ |
| `dftb` | ✓ | — | — |
| `tb_file` | ✓ | ✓ | — |

More detail: [Choose a model](../guides/choose-model.md).

## Commands at a glance

| Command | What it does |
|---|---|
| `rsatb template NAME -o FILE` | write an example input |
| `rsatb check FILE` | validate an input |
| `rsatb show FILE` | print the input with every default filled in |
| `rsatb run FILE` | run the calculation |
| `rsatb doc [TOPIC]` | describe the input options |

All flags: [Command line](../reference/cli.md).
