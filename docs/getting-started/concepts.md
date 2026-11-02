---
description: The five sections of an RSATB input file, what each controls, and the few rules every input follows.
---

<p class="rs-kicker">Get started</p>

# How an input works

<div class="rs-lead" markdown>
An RSATB calculation is described by one JSON file with up to five
sections. All examples, templates and tutorials follow this layout.
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

## What happens when you run

1. **Check.** The input is validated; a run with errors does not start.
2. **Setup.** RSATB builds the Hamiltonian (and overlap) for the structure
   and estimates its energy range.
3. **Tasks.** Each task is computed in order, and its result file is written
   when it finishes.

The setup is done once and shared by all tasks in the input.

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

??? deep "Under the hood: the setup in detail"
    During setup RSATB builds the atomic neighbour graph from the model's
    orbital ranges and distributes it across processes; assembles $H$ (and,
    for overlapping orbitals, $S$) at each k-point; builds $S^{-1/2}$ when
    needed; and estimates the energy range that every calculation is scaled
    to. A profile table at the end of each run shows how long each phase
    took. Background: [Theory](../theory/index.md).

## Commands at a glance

| Command | What it does |
|---|---|
| `rsatb template NAME -o FILE` | write a working input to start from |
| `rsatb check FILE` | validate an input (no solver or MPI needed) |
| `rsatb show FILE` | print the input with every default filled in |
| `rsatb run FILE` | run the calculation |
| `rsatb doc [TOPIC]` | explain any part of the input format |

All flags: [Command line](../reference/cli.md).
