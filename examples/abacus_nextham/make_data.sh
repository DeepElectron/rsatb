#!/bin/bash
# Regenerates data/ : runs ABACUS for fcc Al (nspin=1) and zincblende GaAs
# (nspin=4, spin-orbit), keeps what rsatb reads, and writes each H(R) a second
# time as a NextHam HR_atom_pairs.h5 through NextHam's own writer.
#
# usage: ./make_data.sh <ABACUS PP_ORB dir> <NextHam pre_post_process dir> [ranks]
#   PP_ORB: holds the .upf/.orb files named in abacus/*/STRU
#           (abacus-develop/tests/PP_ORB has them all)
#   ABACUS: the abacus binary, taken from PATH unless $ABACUS is set
set -euo pipefail
PP_ORB=$(realpath "$1")
NEXTHAM=$(realpath "$2")
RANKS=${3:-8}
ABACUS=${ABACUS:-abacus}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}

run() {  # run <system> <nspin> <symbol:orb>...
    local system=$1 nspin=$2; shift 2
    local work=$HERE/_abacus_run/$system data=$HERE/data/$system
    rm -rf "$work"; mkdir -p "$work" "$data"
    cp "$HERE"/abacus/$system/{INPUT,STRU,KPT} "$work"/
    for f in $(awk '/upf|\.orb/ {for (i = 1; i <= NF; i++) if ($i ~ /\.(upf|txt|orb)$/) print $i}' "$work/STRU"); do
        cp "$PP_ORB/$f" "$work/"
    done
    (cd "$work" && OMP_NUM_THREADS=1 mpirun -np "$RANKS" "$ABACUS" > abacus.log 2>&1)
    local out=$(ls -d "$work"/OUT.*)
    cp "$work"/*.orb "$data"/
    mkdir -p "$data/abacus"
    for f in hrs1_nao.csr data-HR-sparse_SPIN0.csr BANDS_1.dat; do
        [ -f "$out/$f" ] && cp "$out/$f" "$data/abacus/"
    done
    # The SCF k-points (fractional) BANDS_1.dat is listed at, and E_Fermi.
    awk '/K-POINTS DIRECT COORDINATES/ {on = 1; getline; next}
         on && NF >= 5 && $1 ~ /^[0-9]+$/ {print $2, $3, $4; next} on {exit}' \
        "$out/running_scf.log" > "$data/abacus/kpoints.dat"
    grep E_Fermi "$out/running_scf.log" | tail -1 | awk '{print $3}' > "$data/abacus/E_Fermi_eV"
    "$PYTHON" "$HERE/to_nextham.py" --nextham "$NEXTHAM" --abacus-out "$out" \
        --poscar "$data/conf.vasp" --orbitals "$@" --nspin "$nspin" --save-path "$data/nextham"
    rm -f "$data/nextham/SR_atom_pairs.h5"  # rsatb builds S from the orbital files
}

run al 1 Al:"$PP_ORB/Al_gga_8au_100Ry_4s4p1d.orb"
run gaas_soc 4 As:"$PP_ORB/As_gga_8au_60Ry_2s2p1d.orb" Ga:"$PP_ORB/Ga_gga_9au_60Ry_2s2p2d.orb"
rm -rf "$HERE/_abacus_run"
