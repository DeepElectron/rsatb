---
description: Install RSATB, run a first calculation, and learn the five sections every input has.
---

<p class="rs-kicker">Getting started</p>

# Getting started

<div class="rs-lead" markdown>
Three short pages take you from nothing to a working calculation and a mental
model of every input you will write afterwards.
</div>

<div class="grid cards" markdown>

-   **1 · [Installation](installation.md)**

    ---

    A conda environment with MPI, HDF5 and the compiled solver. About ten
    minutes, most of it compiling.

-   **2 · [Quick start](quickstart.md)**

    ---

    Write, check and run a graphene calculation from a template, then read the
    result in Python.

-   **3 · [Anatomy of an input](concepts.md)**

    ---

    `structure`, `model`, `run`, `tasks`, `output`: what each section answers,
    and the rules every input follows.

</div>

!!! tip "Working without the solver"
    Writing and validating inputs needs only Python: `rsatb doc`, `rsatb
    template`, `rsatb check`, `rsatb show` and `rsatb schema` import nothing
    compiled and need no MPI. You can prepare a calculation on a laptop or a
    login node and only compile where it will run.
