#!/usr/bin/env python3
"""Conventional (bulk, k-mesh) references for the transport figure.

The same-cell references of make_ref_curves.py reproduce the stochastic
estimators' trace limits on the supercell at Gamma, with the code's own
Jackson kernel, so they measure sampling error alone. This script builds the
other reference a reader expects: the primitive cell on a dense k-mesh, the
Kubo formulas summed over eigenpairs, and a Gaussian broadening of matched
width -- what pyatb, Wannier90 or BoltzTraP would produce, but from OUR
operators, so the velocity convention is the production one.

Operators. For each k the production setup provides H(k), S(k) and the three
pieces of the commutator velocity (bond, xbar, delta; cheby.hpp) through
`dump_operator_coo`; the velocity in the eigenbasis is
    V_a(k) = i (bond_a - xbar_a S^-1 H + H S^-1 delta_a),   v_nm = <n|V_a|m>,
the same composite the Chebyshev sweeps apply. Units: V in eV Angstrom, i.e.
hbar times the velocity.

Formulas (g_s = 2 spin, Omega the primitive-cell volume, N_k mesh points):
  sigma_ab(E)   = (pi e^2 / hbar Omega) g_s (1/N_k) sum_k sum_nm
                  v^a_nm v^b_mn delta(E - e_n) delta(E - e_m)            [Kubo-Greenwood, T = 0]
  Re sigma(w)   = (pi e^2 / hbar Omega) g_s (1/N_k) sum_k sum_nm
                  (f_n - f_m)/(e_m - e_n) |v_nm|^2 delta(hbar w - (e_m - e_n))
                  stored as the REGULAR part (pairs with e_m != e_n); the
                  n = m limit -df/de |v_nn|^2 is the Drude weight D of the
                  D delta(w) term, reported separately
  Im sigma(w)   by Kramers-Kronig from Re sigma, as the driver does.
  MSD(E, t)     = [sum_k sum_nm delta(E - e_n) |v^x_nm|^2 g(e_n - e_m, t)]
                  / [sum_k sum_n delta(E - e_n)],   g = 4 sin^2(de t / 2 hbar) / de^2
  sigma_dc(mu,T)= integral sigma(E) (-df/dE) dE
with the run's own Jackson kernel (its moment count on its window) for
sigma(E) and the MSD, and for sigma(w) a Gaussian of the line width of the
run's windowed sine transform (exact for that window).
Pair sums are accumulated as weighted histograms on fine grids and broadened
once, so the cost is one diagonalization per k.

    python make_bulk_ref.py silicon [--mesh 24] [--batch 256]
writes results/bulk_<run>_curves.npz on the run's own energy/omega/time
grids, plus the f-sum and the mesh/width metadata.
"""
import argparse
import json
import os
import sys

from mpi4py import MPI  # noqa: F401
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from rsatb.inputs.build import build_calculation
from rsatb.inputs.check import load_and_resolve

from make_ref_curves import HBAR_FS, KB, SIG_SI, delta_matrix, window_from_cheb_grid

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, "results", *p)
G_S = 2.0
E_CM1 = 8065.56
EV_CM1 = E_CM1

# mu and temperature of the production inputs (dc point, ac Fermi operator)
SYSTEMS = {"diamond": ("diamond.json", 15.418, 300.0),
           "silicon": ("silicon.json", 6.528, 300.0),
           "aluminum": ("aluminum.json", 11.112081, 300.0)}


def fermi(e, mu, T):
    if T <= 0:
        return (e < mu).astype(float)
    x = np.clip((e - mu) / (KB * T), -700, 700)
    return 1.0 / (1.0 + np.exp(x))


def dfermi(e, mu, T):
    """-df/de, 1/eV."""
    if T <= 0:
        raise ValueError("finite T only")
    x = np.clip((e - mu) / (KB * T), -700, 700)
    f = 1.0 / (1.0 + np.exp(x))
    return f * (1.0 - f) / (KB * T)


def ac_line_width(run):
    """Standard deviation (eV) of the line the run's ac transform makes of a
    single pair: the transform windows the correlator with exp(-2 (t/T)^2),
    T = (ns + 1) t_step, whose transform is a Gaussian of width 2 / T in the
    pair energy. Checked numerically on one pair through the literal sums."""
    za = np.load(R(run, "ac.npz"))
    w_ev = za["omega_cm1"] / EV_CM1
    nt = w_ev.size - 1
    a = w_ev[1] * (nt + 1)
    ns = nt * 2
    t_step = np.pi / (a * 2.0)
    T = (ns + 1) * t_step
    sig = 2.0 / T
    # numerical check around one pair
    t = np.arange(ns + 1) * t_step
    d0 = 0.3 * a
    wgt = np.exp(-2.0 * ((np.arange(ns + 1) + 1.0) / (ns + 1.0)) ** 2)
    near = w_ev[np.abs(w_ev - d0) < 3.0]
    line = np.array([np.sum(2.0 * np.sin(w * t) * wgt * np.sin(d0 * t)) for w in near])
    m = np.clip(line, 0, None); m /= m.sum()
    mean = (near * m).sum()
    sig_num = float(np.sqrt(((near - mean) ** 2 * m).sum()))
    print(f"[ac line] analytic width {sig:.4f} eV, measured {sig_num:.4f} eV", flush=True)
    return float(sig)


def mesh_points(n):
    g = (np.arange(n) / n)
    kx, ky, kz = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)


def dense(cpp, which, axis, k):
    r, c, v = cpp.dump_operator_coo(which, axis, k)
    n = int(max(r.max(), c.max())) + 1
    return sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mesh", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--assemble-only", action="store_true",
                    help="reuse results/bulk_<run>_hist.npz from an earlier k loop")
    args = ap.parse_args()
    run = args.run
    inp, mu, temp = SYSTEMS[run]

    # ---- the run's grids and widths -----------------------------------------
    zs = np.load(R(run, "dc_spectrum.npz"))
    E = zs["energy"]
    c_win, a_win = window_from_cheb_grid(E)
    # Jackson kernel width: measured on the run's own grid (a delta at the
    # window centre through make_ref_curves' delta_matrix), not the pi a / N
    # rule of thumb, which is 2x too wide.
    dj = delta_matrix(E, np.array([c_win]), c_win, a_win, E.size)[:, 0]
    mj = dj / np.trapezoid(dj, E)
    ej = np.trapezoid(E * mj, E)
    sig_j = float(np.sqrt(np.trapezoid((E - ej) ** 2 * mj, E)))
    za = np.load(R(run, "ac.npz"))
    w_ev = za["omega_cm1"] / EV_CM1
    sig_w = ac_line_width(run)
    zm = np.load(R(run, "msd.npz"))
    Em, t_fs = zm["energy"], zm["time_fs"]
    print(f"[{run}] Jackson width {sig_j:.3f} eV, ac line width {sig_w:.3f} eV, "
          f"mesh {args.mesh}^3", flush=True)

    # ---- fine histogram grids -----------------------------------------------
    lo, hi = E[0] - 5 * sig_j, E[-1] + 5 * sig_j
    de = 0.05                                  # eV, << sig_j
    e_bins = np.arange(lo, hi + de, de)
    n_e = e_bins.size - 1
    e_ctr = 0.5 * (e_bins[1:] + e_bins[:-1])
    span = hi - lo
    dd = 0.01                                  # eV, resolves sin^2(de t/2hbar) at t = 40 fs
    d_bins = np.arange(-span, span + dd, dd)
    n_d = d_bins.size - 1
    d_ctr = 0.5 * (d_bins[1:] + d_bins[:-1])

    H_ee = np.zeros((n_e, n_e))               # sum_nm W_nm at (e_n, e_m)      -> sigma(E)
    H_w = np.zeros(n_d)                        # sum_nm (f_n-f_m)/de |v_nm|^2 at de -> sigma(w)
    H_ed = np.zeros((n_e, n_d))                # sum_nm |v^x_nm|^2 at (e_n, e_m - e_n) -> MSD
    N_e = np.zeros(n_e)                        # sum_n at e_n                    -> MSD denominator
    fsum = 0.0
    n_occ = 0.0

    # ---- k loop in batches (each batch is one setup) ------------------------
    kpts = mesh_points(args.mesh)
    n_k = kpts.shape[0]
    hist_path = R(f"bulk_{run}_hist.npz")
    if args.assemble_only:
        h = np.load(hist_path)
        assert int(h["mesh"]) == args.mesh, "histogram mesh differs"
        H_ee, H_w, H_ed, N_e = h["H_ee"], h["H_w"], h["H_ed"], h["N_e"]
        fsum, n_occ, omega = float(h["fsum"]), float(h["n_occ"]), float(h["omega"])
        kpts = kpts[:0]
    resolved, _ = load_and_resolve(os.path.join(HERE, inp))
    resolved["structure"]["supercell"] = [1, 1, 1]
    resolved["run"]["npar"] = 1
    resolved["run"]["preconditioner"]["operator_form"] = "solve"
    resolved["tasks"] = []
    omega = None
    for b0 in range(0, n_k, args.batch):
        kb = kpts[b0:b0 + args.batch]
        resolved["run"]["kpoints"] = [[float(x) for x in k] for k in kb]
        calc, model, options = build_calculation(resolved, log_level="silent")
        cpp = calc._cpp_obj
        if omega is None:
            cell = np.array(cpp.cell()).reshape(3, 3)
            omega = abs(np.linalg.det(cell))
        for ik in range(kb.shape[0]):
            H = dense(cpp, "H", 0, ik)
            S = dense(cpp, "S", 0, ik)
            eps, C = sla.eigh(H, S)
            Sinv = np.linalg.inv(S)
            W = None
            for ax in range(3):
                V = 1j * (dense(cpp, "bond", ax, ik) - dense(cpp, "xbar", ax, ik) @ Sinv @ H
                          + H @ Sinv @ dense(cpp, "delta", ax, ik))
                Ve = C.conj().T @ V @ C
                Ve = 0.5 * (Ve + Ve.conj().T)
                Wa = np.abs(Ve) ** 2
                if ax == 0:
                    Wx = Wa
                W = Wa / 3.0 if W is None else W + Wa / 3.0
            f = fermi(eps, mu, temp)
            n_occ += f.sum()
            ie = np.clip(((eps - lo) / de).astype(int), 0, n_e - 1)
            # sigma(E): 2D histogram over (e_n, e_m)
            np.add.at(H_ee, (ie[:, None].repeat(eps.size, 1), ie[None, :].repeat(eps.size, 0)), W)
            # sigma(w): pair energies with occupation weights
            dmat = eps[None, :] - eps[:, None]          # e_m - e_n
            fdiff = f[:, None] - f[None, :]
            small = np.abs(dmat) < 1e-9
            P = np.where(small, dfermi(eps, mu, temp)[:, None] * np.ones_like(dmat),
                         fdiff / np.where(small, 1.0, dmat)) * W
            idd = np.clip(((dmat + span) / dd).astype(int), 0, n_d - 1)
            np.add.at(H_w, idd.ravel(), P.ravel())
            # f-sum: sum_n f_n sum_m 2 |v_nm|^2 / (e_m - e_n)  (hbar^2/m_e = 7.62 eV A^2)
            with np.errstate(divide="ignore", invalid="ignore"):
                fs = np.where(small, 0.0, 2.0 * Wx / np.where(small, 1.0, dmat))
            fsum += (f[:, None] * fs).sum() / 7.6199682
            # MSD: (e_n, e_m - e_n) histogram of |v^x_nm|^2, and the bra density
            np.add.at(H_ed, (ie[:, None].repeat(eps.size, 1), idd), Wx)
            np.add.at(N_e, ie, 1.0)
        print(f"[{run}] k {min(b0 + args.batch, n_k)}/{n_k}", flush=True)
        del calc, model, options, cpp
    if not args.assemble_only:
        np.savez(hist_path, H_ee=H_ee, H_w=H_w, H_ed=H_ed, N_e=N_e, e_bins=e_bins, d_bins=d_bins,
                 fsum=fsum, n_occ=n_occ, omega=omega, mesh=args.mesh)

    # ---- broaden and assemble ------------------------------------------------
    pref = np.pi * SIG_SI * G_S / omega / n_k
    # sigma(E): the run's own Jackson delta on both energies (a Gaussian of
    # the same width is not enough for a PRODUCT of two kernels: on the 8^3
    # mesh, the supercell's own k-set, it gave 0.54 of the exact same-cell
    # value while the Jackson pair gives 1.00; sigma(omega), a single
    # Gaussian line, matched to 0.1 % either way).
    G = delta_matrix(E, e_ctr, c_win, a_win, E.size)
    sigma_e = pref * np.einsum("ki,ij,kj->k", G, H_ee, G)
    # dc point
    wdc = dfermi(E, mu, temp)
    sigma_dc = np.trapezoid(sigma_e * wdc, E)
    # Re sigma(w): pair histogram broadened by the ac line width. Evaluated
    # on a FINE grid (the histogram step): the run's omega step (0.13-0.17 eV)
    # is coarser than the line width (0.08-0.11 eV), and a Drude line
    # sampled that coarsely, then Kramers-Kronig transformed on the coarse
    # grid, rings (seen on aluminium). The run-grid values are interpolated
    # from the fine curves for the deviation numbers.
    # The pair energy 0 (n = m and exactly degenerate pairs) is the Drude
    # weight: in a pristine crystal it is D delta(omega), which no finite-time
    # transform can resolve and which the code's sine transform of Im C(t)
    # excludes by construction (omega = 0 is set to zero; a constant C has no
    # imaginary part). The comparable quantity is therefore the REGULAR part
    # sigma_reg(omega) = sigma(omega) - D delta(omega); D is reported
    # separately (S/m x eV, the integral of the Drude line).
    central = np.abs(d_ctr) < 0.5 * dd
    drude_weight = pref * H_w[central].sum()
    H_reg = H_w.copy()
    H_reg[central] = 0.0
    Hs = gaussian_filter1d(H_reg, sig_w / dd) / dd        # per eV
    w_fine = np.arange(0.0, w_ev[-1] + dd, dd)
    re_fine = pref * np.interp(w_fine, d_ctr, Hs)
    # Im by Kramers-Kronig, as the driver does (even extension, Hilbert)
    nf = w_fine.size - 1
    full = np.zeros(2 * (nf + 1))
    full[nf + 1:] = re_fine
    full[1:nf + 2] = re_fine[::-1]
    im_fine = np.imag(hilbert(full))[nf + 1:2 * (nf + 1)]
    re_sig = np.interp(w_ev, w_fine, re_fine)
    im_sig = np.interp(w_ev, w_fine, im_fine)
    # MSD(E, t): the run's Jackson delta on the bra energy, time kernel on the pair energy
    cm, am = window_from_cheb_grid(Em)
    Gm = delta_matrix(Em, e_ctr, cm, am, Em.size)
    dens = Gm @ N_e
    small = np.abs(d_ctr) < 1e-9
    msd = np.zeros((t_fs.size, Em.size))
    for it, tv in enumerate(t_fs):
        if tv == 0:
            continue
        g = np.where(small, (tv / HBAR_FS) ** 2,
                     4.0 * np.sin(d_ctr * tv / (2 * HBAR_FS)) ** 2 / np.where(small, 1.0, d_ctr) ** 2)
        msd[it] = (Gm @ (H_ed @ g)) / dens
    diff = 0.5 * np.gradient(msd, t_fs, axis=0, edge_order=1)

    n_eff = fsum / n_k
    print(f"[{run}] f-sum n_eff = {n_eff:.3f} electrons per cell (occupied {n_occ / n_k:.2f} states), "
          f"sigma_dc({mu}, {temp:.0f} K) = {sigma_dc / 1e6:.4f} MS/m, Drude weight D = {drude_weight / 1e6:.4f} MS/m eV "
          f"(sigma(omega) stored is the regular part)", flush=True)
    np.savez(R(f"bulk_{run}_curves.npz"), energy=E, sigma_e=sigma_e, sigma_dc=sigma_dc,
             omega_ev=w_ev, sigma_w=re_sig, sigma_w_imag=im_sig,
             omega_fine=w_fine, sigma_w_fine=re_fine, sigma_w_imag_fine=im_fine,
             drude_weight=drude_weight,
             msd_energy=Em, time_fs=t_fs, msd=msd, diffusion=diff,
             mesh=args.mesh, sigma_jackson=sig_j, sigma_ac=sig_w, omega_cell=omega,
             f_sum=n_eff, mu=mu, temp=temp)
    print(f"[{run}] wrote results/bulk_{run}_curves.npz", flush=True)


if __name__ == "__main__":
    main()
