"""Plot the DOS and AC conductivity written by 07_lcao_diamond_dos_ac.json.

    rsatb run examples/07_lcao_diamond_dos_ac.json
    python examples/plot_dos_ac.py results/07_lcao_diamond_dos_ac

Reads only the .npz files the CLI writes, so it works on any run directory
that has a dos and an ac task in it.
"""

import sys
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt

# sigma is stored as the 6 independent components of a symmetric tensor, in the
# order the C++ side packs them: (-i*i + 5*i)/2 + j for i <= j.
XX, YY, ZZ = 0, 3, 5


def gaussian_broaden(x, y, fwhm):
    """Convolve y(x) with a normalised Gaussian of the given FWHM.

    A single k-point of a finite supercell has a genuinely discrete spectrum,
    so the raw curve is a comb of resolution-limited peaks with Gibbs ringing
    between them (it can dip below zero). Broadening is what turns that into
    the band picture one expects -- it stands in for the finite lifetime and
    the k-point sampling that a single Gamma evaluation does not have. It is a
    presentation choice, not a correction, so the caller is told the width and
    the unbroadened curve stays on the plot underneath.
    """
    if fwhm <= 0.0:
        return y
    dx = float(np.mean(np.diff(x)))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(1, int(np.ceil(4.0 * sigma / dx)))
    grid = np.arange(-half, half + 1) * dx
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    # 'same' with edge padding, so the ends are not dragged toward zero.
    padded = np.pad(y, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main(run_dir, dos_fwhm=1.0, ac_fwhm=None):
    run = pathlib.Path(run_dir)
    dos = np.load(run / "dos.npz")
    ac = np.load(run / "ac.npz")

    energy = dos["energy"]
    dos_values = dos["dos"].mean(axis=0)   # (n_k, n_energy) -> average over k

    omega = ac["omega_cm1"]
    # Average the diagonal: diamond is cubic, so the three should agree to
    # within the stochastic noise, and their spread is a free error bar.
    sigma = ac["sigma"]                      # (n_k, nt+1, 6)
    sigma_k = sigma.mean(axis=0)             # average over k-points
    diagonal = np.real(sigma_k[:, [XX, YY, ZZ]])
    sigma_avg = diagonal.mean(axis=1)
    sigma_spread = diagonal.std(axis=1)

    absorption = np.real(ac["absorption_coefficient"].mean(axis=0)[:, [XX, YY, ZZ]]).mean(axis=1)

    fig, (ax_dos, ax_ac, ax_abs) = plt.subplots(1, 3, figsize=(15, 4.2))

    dos_smooth = gaussian_broaden(energy, dos_values, dos_fwhm)
    ax_dos.plot(energy, dos_values, lw=0.5, color="#9db8d2", alpha=0.8,
                label="raw (discrete levels)")
    ax_dos.plot(energy, dos_smooth, lw=1.4, color="#1f4e79",
                label=f"{dos_fwhm:g} eV FWHM")
    ax_dos.fill_between(energy, 0.0, dos_smooth, alpha=0.15, color="#1f4e79")
    ax_dos.legend(frameon=False, fontsize=8)
    ax_dos.set_xlabel("Energy (eV)")
    ax_dos.set_ylabel("DOS (states / eV / cell)")
    ax_dos.set_title("Density of states")
    ax_dos.set_xlim(energy.min(), energy.max())
    ax_dos.set_ylim(bottom=min(0.0, dos_smooth.min()))
    ax_dos.grid(alpha=0.25)

    if ac_fwhm is None:                       # default: 2% of the plotted range
        ac_fwhm = 0.02 * float(omega.max())
    sigma_smooth = gaussian_broaden(omega, sigma_avg, ac_fwhm)
    absorption_smooth = gaussian_broaden(omega, absorption, ac_fwhm)

    positive = omega > 0
    ax_ac.plot(omega[positive], sigma_avg[positive], lw=0.5, color="#d99a9a", alpha=0.8)
    ax_ac.plot(omega[positive], sigma_smooth[positive], lw=1.4, color="#a03030",
               label=r"$\overline{\mathrm{Re}\,\sigma_{\alpha\alpha}}$")
    ax_ac.fill_between(omega[positive],
                       (sigma_avg - sigma_spread)[positive],
                       (sigma_avg + sigma_spread)[positive],
                       alpha=0.2, color="#a03030",
                       label="spread over xx, yy, zz")
    ax_ac.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax_ac.set_ylabel(r"Re $\sigma(\omega)$")
    ax_ac.set_title("AC conductivity")
    ax_ac.set_xlim(0.0, omega[positive].max())
    ax_ac.grid(alpha=0.25)
    ax_ac.legend(frameon=False, fontsize=9)

    ax_abs.plot(omega[positive], absorption[positive], lw=0.5, color="#a8c9b6", alpha=0.8)
    ax_abs.plot(omega[positive], absorption_smooth[positive], lw=1.4, color="#2d6a4f")
    ax_abs.axhline(0.0, lw=0.6, color="0.5", ls=":")
    ax_abs.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax_abs.set_ylabel(r"$\alpha(\omega)$")
    ax_abs.set_title("Absorption coefficient")
    ax_abs.set_xlim(0.0, omega[positive].max())
    ax_abs.grid(alpha=0.25)

    fig.suptitle(f"Diamond, 128-atom supercell, LCAO  "
                 f"(Gaussian broadening: DOS {dos_fwhm:g} eV, "
                 f"optics {ac_fwhm:.3g} cm$^{{-1}}$)", fontsize=11)
    fig.tight_layout()
    out = run / "dos_ac.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # A couple of numbers worth eyeballing against the physics.
    gap_mask = np.abs(dos_values) < 0.02 * dos_values.max()
    if gap_mask.any():
        gap_energies = energy[gap_mask]
        interior = gap_energies[(gap_energies > energy.min() + 1.0)
                                & (gap_energies < energy.max() - 1.0)]
        if interior.size:
            print(f"low-DOS window: {interior.min():.2f} .. {interior.max():.2f} eV")
    print(f"sigma diagonal anisotropy (max spread / max value): "
          f"{sigma_spread.max() / max(abs(sigma_avg).max(), 1e-30):.3f}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "results/07_lcao_diamond_dos_ac"
    fwhm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    main(run, dos_fwhm=fwhm)
