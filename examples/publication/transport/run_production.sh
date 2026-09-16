#!/bin/bash
# The three production runs behind fig_transport: dc_spectrum, dc, ac, msd
# and quantum_metric from one setup each, for diamond and silicon (8^3 of the
# 2-atom cell, 1,024 atoms) and aluminium (9^3 of the 1-atom cell, 729
# atoms). Sequential (two of these do not fit a 256 GB node together) and
# idempotent: a system whose last output (qmetric.npz) exists is skipped.
#
# Cost on 48 cores: diamond and silicon about a day each; aluminium several
# days, because its 4s4p1d basis spans ~640 eV and every moment count is 4x
# theirs (see aluminum.json). Status lines go to production_status.txt.
#
#   OMP_NUM_THREADS=48 ./run_production.sh            # all three
#   OMP_NUM_THREADS=48 ./run_production.sh silicon    # one system
D="$(cd "$(dirname "$0")" && pwd)"; S="$D/production_status.txt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48} OMP_PROC_BIND=close OMP_PLACES=cores
mkdir -p "$D/results"
for job in ${@:-diamond silicon aluminum}; do
  [ -f "$D/results/$job/qmetric.npz" ] && { echo "=== $job cached" >> "$S"; continue; }
  echo "=== $job start $(date '+%F %H:%M:%S')" >> "$S"
  rsatb run "$D/$job.json" > "$D/results/$job.log" 2>&1
  echo "=== $job exit=$? $(date '+%F %H:%M:%S')" >> "$S"
done
echo "PRODUCTION DONE $(date '+%F %H:%M:%S')" >> "$S"
