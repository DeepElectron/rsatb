#!/bin/bash
# The two references drawn under every curve of fig_transport, per system:
#   1. gen_reference_ops.py  dense generalized diagonalization of the SAME
#                            supercell with the production operators
#                            (results/ref_<run>_eig.npz; minutes, ~1.5 GB
#                            per matrix at 13,312 basis functions);
#   2. make_ref_curves.py    the estimators' trace limits from those
#                            eigenpairs with the code's own kernels, so the
#                            deviation is sampling error alone
#                            (results/ref_<run>_curves.npz);
#   3. make_bulk_ref.py      the bulk limit: primitive cell on a dense
#                            k-mesh, textbook Kubo sums, matched broadening
#                            (results/bulk_<run>_curves.npz; hours).
# Steps 2 and 3 read the run's outputs (its energy grids), so run
# run_production.sh first. Idempotent per file. Run with the rsatb environment
# active (`python` must be the one rsatb is installed in).
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"; mkdir -p results
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
for run in ${@:-diamond silicon aluminum}; do
  [ -f "results/ref_${run}_eig.npz" ]    || python gen_reference_ops.py "$run"
  [ -f "results/$run/msd.npz" ] || { echo "$run: no run outputs yet, skipping curves"; continue; }
  [ -f "results/ref_${run}_curves.npz" ] || python make_ref_curves.py "$run"
  [ -f "results/bulk_${run}_curves.npz" ] || python make_bulk_ref.py "$run" --mesh 24 --batch 512
done
echo "REFERENCES DONE"
