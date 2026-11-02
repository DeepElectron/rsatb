---
description: Lay out k-point groups (npar), MPI processes and OpenMP threads for RSATB on a cluster, with a SLURM example.
---

<p class="rs-kicker">Guide</p>

# Run on a cluster

<div class="rs-lead" markdown>
RSATB parallelizes over k-points, over atoms (MPI processes) and within each
process (OpenMP threads). Assign resources in that order.
</div>

| Level | Set by | Divides |
|---|---|---|
| k-point groups | `run.npar` | the k-points |
| MPI processes in a group | `mpirun -n` together with `npar` | the atoms |
| OpenMP threads | `OMP_NUM_THREADS` | the work of one process |

1. **k-points.** Groups work on different k-points and do not communicate.
   Set `npar` to the number of k-points or a divisor of it.
2. **Processes.** Within a group, processes divide the atoms; each process
   must own at least one atom.
3. **Threads.** Use one thread per physical core.

## SLURM example

```bash title="job.sh"
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4          # one MPI process per NUMA domain
#SBATCH --cpus-per-task=12           # physical cores per process
#SBATCH --time=04:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores OMP_PROC_BIND=close

srun --cpu-bind=cores rsatb run input.json
```

With Open MPI's `mpirun`, give the process placement explicitly:

```bash
OMP_NUM_THREADS=12 mpirun -n 16 --map-by numa:PE=12 --bind-to core \
    rsatb run input.json
```

Output is written by process 0 only.

## Common problems

!!! warning "No speed-up with threads under Open MPI"
    By default Open MPI binds each process to a single core, so additional
    threads cannot run. Use `--map-by …:PE=N` as above.

!!! warning "Too many threads for a small system"
    A cell with few atoms has little work to divide. If a small test runs
    slower with more threads, reduce `OMP_NUM_THREADS`.

!!! warning "Out of memory during setup"
    For non-orthogonal models, building $S^{-1/2}$ during setup uses memory
    that grows with the number of threads per process. Use fewer threads per
    process and more processes.

## Before submitting

```bash
rsatb check input.json                          # no MPI or solver needed
python examples/estimate_matvecs.py input.json  # amount of work per task
```

`rsatb check` also warns when `npar` leaves k-point groups without work:

```text
warning: run.npar: npar is 8 but there are only 4 k-points, so 4 sub-communicators have nothing to do
```
