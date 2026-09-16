#!/bin/bash
# The nine production runs behind fig_ldos_gallery: DOS/PDOS, charge density
# and LDOS maps for diamond, silicon (12^3 of the 2-atom cell, 3,456 atoms)
# and aluminium (12^3 of the 1-atom cell, 1,728 atoms).
#
# Sequential and idempotent: a job whose output exists is skipped, so a
# re-run resumes wherever it stopped. Sequential because one 12^3 run peaks
# at ~100 GB. Each job is hours on 48 cores (silicon's LDOS ~5.5 h, most of
# it the S^-1/2 factor; aluminium's density ~10 h). Status lines go to
# production_status.txt beside this script.
#
#   OMP_NUM_THREADS=48 ./run_production.sh            # everything
#   OMP_NUM_THREADS=48 ./run_production.sh dia_pdos   # one job
D="$(cd "$(dirname "$0")" && pwd)"; S="$D/production_status.txt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48} OMP_PROC_BIND=close OMP_PLACES=cores
mkdir -p "$D/results"

done_job() { # job -> 0 if its output exists
  case $1 in
    *pdos*)    [ -f "$D/results/$1/pdos_slot.npz" ];;
    *density*) [ -f "$D/results/$1/density.rank00000.npz" ];;
    *ldos*)    [ -f "$D/results/$1/ldos.rank00000.npz" ];;
  esac
}

JOBS=${@:-dia_pdos si_pdos al_pdos dia_density si_density al_density dia_ldos si_ldos al_ldos}
for job in $JOBS; do
  done_job "$job" && { echo "=== $job cached" >> "$S"; continue; }
  echo "=== $job start $(date '+%F %H:%M:%S')" >> "$S"
  rsatb run "$D/$job.json" > "$D/results/$job.log" 2>&1
  echo "=== $job exit=$? $(date '+%F %H:%M:%S')" >> "$S"
done
echo "PRODUCTION DONE $(date '+%F %H:%M:%S')" >> "$S"
