#!/bin/bash
# Diagonalization references for fig_ldos_gallery, compare_spectrum.py and
# compare_ldos.py. diag_kpoint diagonalizes the PRIMITIVE cell on a
# Gamma-centred k-mesh and folds the result onto the supercell: a 12^3 mesh
# of the primitive cell is the same eigenproblem as the 12^3 supercell at
# Gamma, so the deviation of a run from its *_rs_k12 reference is the
# stochastic estimator's error alone. The 24^3 meshes are the converged
# (dense-k) spectrum the figure's second deviation column quotes.
#
# diag_kpoint is built by the normal build (`pip install .`, CMake target
# diag_kpoint under build/<wheel-tag>/); set DIAG_KPOINT to point elsewhere.
# The DOS-only meshes take seconds to minutes; each --realspace fold takes
# ~10 min. Idempotent.
D="$(cd "$(dirname "$0")" && pwd)"; ROOT="$D/../../.."; T="$ROOT/test/data"
DIAG_KPOINT=${DIAG_KPOINT:-$(ls -t "$ROOT"/build/*/diag_kpoint 2>/dev/null | head -1)}
[ -x "$DIAG_KPOINT" ] || { echo "diag_kpoint not found: build the package, or set DIAG_KPOINT" >&2; exit 1; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}

run() { # name system orb mesh [extra flags]
  local name=$1 sys=$2 orb=$3 mesh=$4; shift 4
  local out="$D/results/$name"
  [ -f "$out/k_eigenvalues.f64" ] && { echo "cached $name"; return; }
  mkdir -p "$out"
  "$DIAG_KPOINT" --poscar "$T/$sys/conf.vasp" --ham "$T/$sys/hamiltonians.h5" \
      --orb "$orb" --orbdir "$T/$sys" --kmesh "$mesh" --out "$out" "$@" > "$out/log" 2>&1
  echo "done $name exit=$?"
}

# converged spectra (DOS/PDOS only)
run dia_k24 diamond C_gga_7au_100Ry_2s2p1d.orb  24
run si_k24  silicon Si_gga_7au_100Ry_2s2p1d.orb 24
run al_k24  Al      Al_gga_7au_100Ry_4s4p1d.orb 24
# same-cell references: spectrum, density and LDOS at the runs' energies.
# --electrons is spin-summed (4 per C/Si atom, 11 per Al atom) and fixes mu;
# the LDOS energies and eta match the *_ldos.json inputs.
run dia_rs_k12 diamond C_gga_7au_100Ry_2s2p1d.orb  12 --realspace --electrons 13824 --temp 300 --eta 0.25 --ldos -0.419,13.026,17.824
run si_rs_k12  silicon Si_gga_7au_100Ry_2s2p1d.orb 12 --realspace --electrons 13824 --temp 300 --eta 0.25 --ldos 8.8,16.8,18.3
run al_rs_k12  Al      Al_gga_7au_100Ry_4s4p1d.orb 12 --realspace --electrons 19008 --temp 300 --eta 0.25 --ldos 7.112081,11.112081,15.112081
echo "REFERENCES DONE"
