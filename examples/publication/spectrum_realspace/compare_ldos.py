"""Survey-sampling LDOS maps against the exact k-point fold of the same cell.

    python compare_ldos.py [dia_ldos] [si_ldos] [al_ldos]

Per energy: the rms error relative to the reference map's rms, the mean
(level) error, the realised capture c and sketch rank k, and the run's own
prediction sqrt(2 (1 - c) / k). The reference is the k-point diagonalization
folded onto the run's cell (results/*_rs_k12, written by make_references.sh),
so the deviation is the estimator's alone.
"""
import glob, json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
REF = {"dia_ldos": "results/dia_rs_k12", "si_ldos": "results/si_rs_k12",
       "al_ldos": "results/al_rs_k12"}


def load_map(run):
    parts = sorted(glob.glob(os.path.join(D, "results", run, "ldos.rank*.npz")))
    if not parts:
        return None
    first = np.load(parts[0])
    n_point = json.load(open(os.path.join(D, "results", run, "summary.json")))["results"][0]["scalars"]["n_point"]
    ldos = np.zeros((first["ldos"].shape[0], n_point))
    for p in parts:
        d = np.load(p)
        ldos[:, d["point_index"]] = d["ldos"]
    return ldos, first["capture"], first["sketch_rank"], first["predicted_rel_error"]


def main(runs):
    for run in runs:
        got = load_map(run)
        if got is None:
            print(run, "(missing)")
            continue
        ldos, capture, rank, predicted = got
        refdir = os.path.join(D, REF[run])
        npt = json.load(open(os.path.join(refdir, "k_grid.json")))["n_point"]
        ref = np.fromfile(os.path.join(refdir, "k_ldos.f64")).reshape(-1, npt)
        s = json.load(open(os.path.join(D, "results", run, "summary.json")))["results"][0]
        energies = json.load(open(os.path.join(D, run + ".json")))["tasks"][0]["energies"]
        print(f"{run}: {s['seconds'] / 3600:.2f} h, n_states {s['scalars']['n_states']}, "
              f"filter degree {s['scalars']['filter_degree']}")
        print("   E (eV)   rel rms   mean err   capture   rank   predicted")
        for k, e in enumerate(energies):
            d = ldos[k] - ref[k]
            rms = np.sqrt((d ** 2).mean()) / np.sqrt((ref[k] ** 2).mean())
            mean = d.mean() / ref[k].mean()
            print(f"  {e:7.3f}   {100 * rms:6.2f} %   {100 * mean:+6.2f} %   {capture[k]:7.3f}   {rank[k]:4d}   {100 * predicted[k]:6.2f} %")


if __name__ == "__main__":
    main(sys.argv[1:] or ["dia_ldos", "si_ldos", "al_ldos"])
