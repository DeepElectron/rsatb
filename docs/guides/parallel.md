---
description: Lay out MPI ranks, k-point groups (npar) and OpenMP threads for RSATB on a cluster, with a SLURM script and the two common traps.
---

<p class="rs-kicker">How-to guide</p>

# Run on a cluster

<div class="rs-lead" markdown>
RSATB has three levels of parallelism. Fill them from the outside in, because
communication cost grows inward.
</div>

| Level | Set by | Splits | Communication |
|---|---|---|---|
| 1 · k-point groups | `run.npar` | the k-points | none between groups |
| 2 · MPI ranks in a group | `mpirun -n` / `npar` | the atoms | halo exchange every product |
| 3 · OpenMP threads | `OMP_NUM_THREADS` | block rows of a rank | shared memory |

1. **k-points first.** Groups never talk to each other, so this axis scales
   perfectly. Set `npar` to the number of k-points or a divisor of it.
2. **Then ranks.** Within a group, ranks divide the atoms. Every rank must own
   at least one atom; setup refuses a layout where one would own none.
3. **Then threads.** Threads split the block rows of a rank. Use physical
   cores only — the Chebyshev loop is memory-bandwidth bound, and SMT
   siblings only contend.

## A SLURM job

```bash title="job.sh"
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4          # one rank per NUMA domain
#SBATCH --cpus-per-task=12           # physical cores per rank
#SBATCH --time=04:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores OMP_PROC_BIND=close

srun --cpu-bind=cores rsatb run input.json --log-level progress
```

With Open MPI's `mpirun` instead of `srun`, state the mapping explicitly:

```bash
OMP_NUM_THREADS=12 mpirun -n 16 --map-by numa:PE=12 --bind-to core \
    rsatb run input.json
```

stdout goes to `slurm-<jobid>.out`; only rank 0 reports.

## Two traps

!!! warning "Open MPI binds each rank to one core"
    Open MPI 5 binds every rank to a single core by default, so
    `mpirun -n 4` without `--map-by …:PE=N` runs four serial ranks no matter
    what `OMP_NUM_THREADS` says. Always state the layout.

!!! warning "Idle threads spin"
    OpenMP threads without work spin at the barriers instead of sleeping. A
    2-atom cell has 2 block rows, so 96 threads burn ~100% CPU each while 2 do
    the work. If the profile table shows far more CPU than wall time for a
    task, the system is too small for the thread count — reduce it.

## Memory

The largest transient on non-orthogonal models is the $S^{-1/2}$ seed during
setup, which scales with *threads* × (neighbours × orbitals per atom)². When a
setup runs out of memory, lower `OMP_NUM_THREADS` and raise the rank count
over more nodes; see [Non-orthogonal bases](../theory/overlap.md#the-graph-local-s-12).

## Before you queue

```bash
rsatb check input.json                          # on the login node: no MPI needed
python examples/estimate_matvecs.py input.json  # how much work the tasks are
```

`rsatb check` also warns when `npar` leaves groups idle:

```text
warning: run.npar: npar is 8 but there are only 4 k-points, so 4 sub-communicators have nothing to do
```
