#ifndef RSATB_CALC_AC_HPP
#define RSATB_CALC_AC_HPP

#include <algorithm>
#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "cheby.hpp"
#include "fermi_dirac.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace vbcsr;

namespace rsatb {
namespace calc {

// Time-sampling density of the AC correlator, shared by both evolution
// routes and the sine transform. The correlator carries transition
// frequencies up to the FULL spectral span (occupied-to-unoccupied energy
// differences, 2 half-widths), so its Nyquist step is pi/(2*half_width) --
// half the DOS step. Coarser sampling folds content in (S, 2S) back onto
// (0, S) with a sign flip (sin((2S-w) t_k) = -sin(w t_k) on that grid):
// spurious negative low-frequency absorption. This is a constant, not a
// knob: the content bound 2S is exact, so 1 aliases, 2 is Nyquist-exact,
// and more adds nothing.
inline constexpr int kAcOversample = 2;

template <typename T, typename VectorType>
void apply_operator_dual(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& op,
    const VectorType& psi_in,
    VectorType& psi_out,
    ChebyMode mode = ChebyMode::NORMAL,
    double scale = 1.0
) {
    auto construct_vec = [&](DistGraph* g, int nv) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            return VectorType(g, nv);
        } else {
            return VectorType(g);
        }
    };
    int nv = 1;
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        nv = psi_in.num_vectors;
    }

    if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            op.mult_dense(const_cast<VectorType&>(psi_in), psi_out);
        } else {
            op.mult(const_cast<VectorType&>(psi_in), psi_out);
        }

        if (cheby.S) {
            VectorType tmp = construct_vec(psi_in.graph, nv);
            tmp.copy_from(psi_out);
            if (cheby.M) {
                cheby.solver.solve(*cheby.S, tmp, psi_out, *cheby.M, cheby.ptype);
            } else {
                cheby.solver.solve(*cheby.S, tmp, psi_out);
            }
        }
    } else if (mode == ChebyMode::ADJOINT) {
        VectorType tmp = construct_vec(psi_in.graph, nv);
        if (cheby.S) {
            if (cheby.M) {
                cheby.solver.solve(*cheby.S, psi_in, tmp, *cheby.M, cheby.ptype);
            } else {
                cheby.solver.solve(*cheby.S, psi_in, tmp);
            }
        } else {
            tmp.copy_from(psi_in);
        }

        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            op.mult_dense(tmp, psi_out);
        } else {
            op.mult(tmp, psi_out);
        }
    } else {
        throw std::runtime_error("apply_operator_dual only supports NORMAL, CONJUGATE, and ADJOINT modes");
    }

    psi_out.scale(T(scale));
}

// Compute Current: J = -i S^{-1} [H, R] = -S^{-1} V (if V = -i[H,R])
// We assume Vx, Vy, Vz are the velocity operators V.
// J = - S^{-1} V Psi
template <typename T, typename VectorType>
void current(Chebyshev<T>& cheby, 
             BlockSpMat<T>& Vx, 
             BlockSpMat<T>& Vy, 
             BlockSpMat<T>& Vz, 
             const VectorType& Psi_in, 
             VectorType* J_Psi, // Array of 3
             double scale = 1.0) {
    apply_operator_dual(cheby, Vx, Psi_in, J_Psi[0], ChebyMode::NORMAL, -scale);
    apply_operator_dual(cheby, Vy, Psi_in, J_Psi[1], ChebyMode::NORMAL, -scale);
    apply_operator_dual(cheby, Vz, Psi_in, J_Psi[2], ChebyMode::NORMAL, -scale);
}

// Compute Adjoint Current: J = - V S^{-1} Psi
template <typename T, typename VectorType>
void current_adjoint(Chebyshev<T>& cheby, 
                     BlockSpMat<T>& Vx, 
                     BlockSpMat<T>& Vy, 
                     BlockSpMat<T>& Vz, 
                     const VectorType& Psi_in, 
                     VectorType* J_Psi, // Array of 3
                     double scale = 1.0) {
    apply_operator_dual(cheby, Vx, Psi_in, J_Psi[0], ChebyMode::ADJOINT, -scale);
    apply_operator_dual(cheby, Vy, Psi_in, J_Psi[1], ChebyMode::ADJOINT, -scale);
    apply_operator_dual(cheby, Vz, Psi_in, J_Psi[2], ChebyMode::ADJOINT, -scale);
}

// AC Conductivity Correlation
//
// Fermi-operator degree selection:
//   n_kpm > 0  — legacy fixed-degree Jackson expansion (needs n_kpm ~
//                pi*W/kT at low temperature to be pointwise converged).
//   n_kpm <= 0 — adaptive tapered ACE-FD expansion (fermi_options); the
//                degree is chosen per vector from its own spectral measure
//                and is essentially temperature independent on extended
//                systems. Valid here because f(H) enters the Kubo correlator
//                linearly and sigma(omega) carries the time-window
//                broadening; see benchmarks/fermi_dirac/METHOD.md.
template <typename T, typename VectorType>
void corr_ac(Chebyshev<T>& cheby,
             BlockSpMat<T>& Vx,
             BlockSpMat<T>& Vy,
             BlockSpMat<T>& Vz,
             const VectorType& Psi_in,
             int nt,
             double temp,
             double mu,
             double scale, // Unused, relying on cheby
             double shift, // Unused, relying on cheby
             int n_kpm,
             double eps = 1e-14,
             std::vector<std::complex<double>>* cac = nullptr,
             FermiExpansionOptions fermi_options = {}) {

    DistGraph* graph = Psi_in.graph;
    int rank = graph->rank;

    if (n_kpm > 0) {
        // Legacy call style: a positive n_kpm pins the degree and asks for
        // the classic Jackson expansion.
        fermi_options.n_fixed = n_kpm;
        fermi_options.summation = KpmKernel::Jackson;
    }
    auto apply_fermi = [&](const VectorType& in, VectorType& out, ChebyMode fd_mode) {
        fermi_dirac_kpm(cheby, in, out, temp, mu, fermi_options, fd_mode);
    };
    
    // 1. Generate Time Propagation Coefficients. The step is the correlator's
    // Nyquist step pi/(2S) (see kAcOversample above); the original code's
    // dt = pi aliased occupied-to-unoccupied transitions above the
    // half-width onto the reported window with a sign flip.
    const double dt = M_PI / kAcOversample;
    const int ns = nt * kAcOversample;

    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(dt, eps, true, coeffs);
    int n_bes = coeffs.size();

    // Initialize cac: ns + 1 samples at spacing dt.
    if (cac!=nullptr) {
        if (cac->size() < static_cast<size_t>(ns + 1) * 6) {
            cac->resize(static_cast<size_t>(ns + 1) * 6);
        }
        std::fill(cac->data(), cac->data() + (ns + 1) * 6, 0.0);
    }

    // Prepare Vectors
    // Helper to construct
    auto construct_vec = [&](DistGraph* g, int nv) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            return VectorType(g, nv);
        } else {
            return VectorType(g);
        }
    };

    int nv = 1;
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        nv = Psi_in.num_vectors;
    }

    VectorType Psi0 = construct_vec(graph, nv);
    VectorType Psi_out = construct_vec(graph, nv);
    VectorType buff = construct_vec(graph, nv);
    
    std::vector<VectorType> J_Psi_vec;
    J_Psi_vec.reserve(3);
    J_Psi_vec.emplace_back(construct_vec(graph, nv));
    J_Psi_vec.emplace_back(construct_vec(graph, nv));
    J_Psi_vec.emplace_back(construct_vec(graph, nv));
    VectorType* J_Psi = J_Psi_vec.data();

    std::vector<VectorType> J_Psi0_vec;
    J_Psi0_vec.reserve(3);
    J_Psi0_vec.emplace_back(construct_vec(graph, nv));
    J_Psi0_vec.emplace_back(construct_vec(graph, nv));
    J_Psi0_vec.emplace_back(construct_vec(graph, nv));
    VectorType* J_Psi0 = J_Psi0_vec.data();

    // 2. Compute Psi0 = f(H) Psi_in (Adjoint)
    // Original code: VecConjugate(Psi_in); fermi...; VecConjugate(Psi_in);
    // Why conjugate?
    // If Psi_in is a ket |psi>, we want <psi| f(H) ...
    // So we want <psi f(H)|.
    // If we represent bra as conjugated ket, then yes.
    // But `fermi_dirac_kpm` applies f(H) to a vector.
    // If we want f(H) |psi>, we just call it.
    // If we want <psi| f(H), and f(H) is Hermitian, it is <f(H) psi|.
    // So we compute Psi0 = f(H) Psi_in.
    // Original code:
    // VecConjugate(Psi_in);
    // fermi_dirac_kpm(..., ADJOINT, ...);
    // VecConjugate(Psi_in);
    // This suggests Psi_in is conjugated before KPM.
    // Maybe Psi_in is treated as bra?
    // Let's follow the original code exactly.
    
    // We need a non-const copy of Psi_in to conjugate
    // VectorType Psi_in_copy(graph);
    // if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
    //     Psi_in_copy = VectorType(graph, Psi_in.num_vectors);
    // }
    // Psi_in_copy.copy_from(Psi_in);
    
    // Psi_in_copy.conjugate();
    apply_fermi(Psi_in, Psi0, ChebyMode::ADJOINT);
    // Psi_in_copy.conjugate(); // Restore (though we use copy)

    // 3. Compute J_Psi = current(Psi_in)
    current(cheby, Vx, Vy, Vz, Psi_in, J_Psi, 1.0);

    // 4. Apply (1-f) to J_Psi
    // (1-f) J Psi = J Psi - f(J Psi)
    for (int i = 0; i < 3; ++i) {
        apply_fermi(J_Psi[i], buff, ChebyMode::NORMAL);
        J_Psi[i].axpy(-1.0, buff);
    }

    // 5. Compute J_Psi0 = current_adjoint(Psi0)
    current_adjoint(cheby, Vx, Vy, Vz, Psi0, J_Psi0, 1.0);

    // 6. Initial Correlation (t=0)
    for (int i = 0; i < 3; ++i) { // Right (J_Psi)
        for (int j = i; j < 3; ++j) { // Left (J_Psi0)
            std::complex<double> dot_val;
            if constexpr (std::is_same_v<VectorType, DistVector<T>>) {
                dot_val = J_Psi0[j].dot(J_Psi[i]);
            } else {
                std::vector<typename VectorType::value_type> dots;
                J_Psi0[j].dot(J_Psi[i], dots);
                std::complex<double> sum = 0;
                for (auto d : dots) sum += d;
                dot_val = sum / static_cast<double>(Psi_in.num_vectors);
            }
            if (cac!=nullptr) (*cac)[int((-i * i + 5 * i) / 2 + j)] = dot_val;
        }
    }

    // 7. Time Propagation
    for (int it = 0; it < ns; ++it) {
        ::rsatb::util::throw_if_interrupted(graph->comm);
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
            ::rsatb::util::every_nth(it, std::max(1, ns / 20))) {
            std::cout << "ac: step " << it + 1 << " / " << ns << std::endl;
        }

        // Propagate J_Psi (Right) - NORMAL
        for (int i = 0; i < 3; ++i) {
            cheby.propagate(J_Psi[i], Psi_out, coeffs, ChebyMode::NORMAL);
            J_Psi[i].copy_from(Psi_out);
        }

        // Propagate Psi0 (Left) - ADJOINT
        cheby.propagate(Psi0, Psi_out, coeffs, ChebyMode::ADJOINT);
        Psi0.copy_from(Psi_out);

        // Update J_Psi0
        current_adjoint(cheby, Vx, Vy, Vz, Psi0, J_Psi0, 1.0);

        // Correlation
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                std::complex<double> dot_val;
                if constexpr (std::is_same_v<VectorType, DistVector<T>>) {
                    dot_val = J_Psi0[j].dot(J_Psi[i]); // <J_Psi0 | J_Psi>
                } else {
                    std::vector<typename VectorType::value_type> dots;
                    J_Psi0[j].dot(J_Psi[i], dots);
                    std::complex<double> sum = 0;
                    for (auto d : dots) sum += d;
                    dot_val = sum / static_cast<double>(Psi_in.num_vectors);
                }
                if (cac!=nullptr) (*cac)[(it + 1) * 6 + int((-i * i + 5 * i) / 2 + j)] = dot_val;
            }
        }
    }
}

// Concurrent (shared-Chebyshev-state) AC correlator.
//
// Mathematically the same estimator as corr_ac: both sides of
//   C_ij(t) = < J_j U(t) f(H) psi ,  U(t) (1 - f(H)) J_i psi >
// are linear combinations of the Chebyshev states T_n(Op) of their
// block-start seeds, so one recurrence of ~b*pi + tail matvecs per side per
// block replaces b sequential propagations of ~19 matvecs each (the
// state-based scheme of arXiv:2510.17432: expansion load R(pi) ~ 6 falls
// towards 1 for long steps). Reconstructing the state at each sample time is
// axpys over those shared states -- cheap next to the preconditioned matvec,
// which is why the per-sample current applications and dots, not the
// reconstruction, are the non-accelerable cost here.
//
// Sampling follows kAcOversample (see its note at the top of this file):
// dt = pi/(2*half_width), alias-free for the full transition band. Here the
// finer sampling is essentially free -- half-step states are just different
// coefficient combinations of the same stored Chebyshev states -- whereas
// the sequential route pays real matvecs for it (two pi/2 expansions cost
// ~32 terms where one pi step costs ~20).
//
// `block` bounds memory: 4 sides x (block*pi + Bessel tail) stored
// states of Nbasis (x nsamples columns for the multivector case). The
// matvec saving saturates like 4*pi + 4*N(pi)/block, so block ~ 128
// already captures ~95% of the full-block gain.
//
// cac receives nt*kAcOversample + 1 samples x 6 components, sample spacing
// pi/(kAcOversample*half_width) in physical time -- the same contract as
// corr_ac, so the two routes are interchangeable sample for sample.
template <typename T, typename VectorType>
void corr_ac_concurrent(Chebyshev<T>& cheby,
                        BlockSpMat<T>& Vx,
                        BlockSpMat<T>& Vy,
                        BlockSpMat<T>& Vz,
                        const VectorType& Psi_in,
                        int nt,
                        double temp,
                        double mu,
                        int n_kpm,
                        double eps = 1e-14,
                        std::vector<std::complex<double>>* cac = nullptr,
                        int block = 0,
                        FermiExpansionOptions fermi_options = {}) {

    DistGraph* graph = Psi_in.graph;
    int rank = graph->rank;

    if (n_kpm > 0) {
        fermi_options.n_fixed = n_kpm;
        fermi_options.summation = KpmKernel::Jackson;
    }
    auto apply_fermi = [&](const VectorType& in, VectorType& out, ChebyMode fd_mode) {
        fermi_dirac_kpm(cheby, in, out, temp, mu, fermi_options, fd_mode);
    };

    const int os = kAcOversample;
    if (block <= 0 || block > nt) block = nt;
    const int ns = nt * os;

    if (cac != nullptr) {
        if (cac->size() < static_cast<size_t>(ns + 1) * 6) {
            cac->resize(static_cast<size_t>(ns + 1) * 6);
        }
        std::fill(cac->data(), cac->data() + (ns + 1) * 6, 0.0);
    }

    // Coefficient table: expansion of e^{-i Op (k pi/os)} for every sample
    // offset within a block. Truncation length grows monotonically with the
    // time, so ctab.back() sets the number of stored Chebyshev states.
    std::vector<std::vector<std::complex<double>>> ctab(block * os + 1);
    for (int k = 1; k <= block * os; ++k) {
        timeprop_coeff(k * M_PI / os, eps, true, ctab[k]);
    }
    const int n_states = static_cast<int>(ctab[block * os].size());

    auto construct_vec = [&](DistGraph* g, int nv) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            return VectorType(g, nv);
        } else {
            return VectorType(g);
        }
    };
    int nv = 1;
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        nv = Psi_in.num_vectors;
    }

    // Seeds: identical to corr_ac steps 2-5.
    VectorType Psi0 = construct_vec(graph, nv);
    VectorType buff = construct_vec(graph, nv);
    std::vector<VectorType> Phi_vec;
    Phi_vec.reserve(3);
    for (int i = 0; i < 3; ++i) Phi_vec.emplace_back(construct_vec(graph, nv));
    VectorType* Phi = Phi_vec.data();

    apply_fermi(Psi_in, Psi0, ChebyMode::ADJOINT);
    current(cheby, Vx, Vy, Vz, Psi_in, Phi, 1.0);
    for (int i = 0; i < 3; ++i) {
        apply_fermi(Phi[i], buff, ChebyMode::NORMAL);
        Phi[i].axpy(-1.0, buff);
    }

    std::vector<VectorType> J0_vec;
    J0_vec.reserve(3);
    for (int i = 0; i < 3; ++i) J0_vec.emplace_back(construct_vec(graph, nv));
    VectorType* J0 = J0_vec.data();

    auto record = [&](int sample, VectorType* left_J, VectorType* right) {
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                std::complex<double> dot_val;
                if constexpr (std::is_same_v<VectorType, DistVector<T>>) {
                    dot_val = left_J[j].dot(right[i]);
                } else {
                    std::vector<typename VectorType::value_type> dots;
                    left_J[j].dot(right[i], dots);
                    std::complex<double> sum = 0;
                    for (auto d : dots) sum += d;
                    dot_val = sum / static_cast<double>(nv);
                }
                if (cac != nullptr) {
                    (*cac)[static_cast<size_t>(sample) * 6 + ((-i * i + 5 * i) / 2 + j)] = dot_val;
                }
            }
        }
    };

    // t = 0.
    current_adjoint(cheby, Vx, Vy, Vz, Psi0, J0, 1.0);
    record(0, J0, Phi);

    // Shared Chebyshev states: sides 0-2 are the Phi_i (NORMAL evolution),
    // side 3 is Psi0 (ADJOINT evolution, as in the sequential loop).
    std::vector<std::vector<VectorType>> X(4);
    for (int s = 0; s < 4; ++s) {
        X[s].reserve(n_states);
        for (int n = 0; n < n_states; ++n) X[s].emplace_back(construct_vec(graph, nv));
    }
    std::vector<VectorType> R_vec;
    R_vec.reserve(4);
    for (int s = 0; s < 4; ++s) R_vec.emplace_back(construct_vec(graph, nv));
    VectorType* R = R_vec.data();

    if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
        std::cout << "ac: concurrent, block " << block << " of " << nt
                  << " steps, oversample " << os << ", " << 4 * n_states
                  << " stored states" << std::endl;
    }

    const int n_blocks = (nt + block - 1) / block;
    int done = 0;
    int iblock = 0;
    while (done < nt) {
        ::rsatb::util::throw_if_interrupted(graph->comm);
        const int b = std::min(block, nt - done);
        const int nk = static_cast<int>(ctab[b * os].size());
        ++iblock;

        // Chebyshev recurrence from the block-start seeds (the matvecs).
        // Progress counts applies across all four sides, so long blocks
        // never go quiet while the later sides expand.
        const int total_applies = 4 * std::max(1, nk - 1);
        for (int s = 0; s < 4; ++s) {
            const ChebyMode mode = (s == 3) ? ChebyMode::ADJOINT : ChebyMode::NORMAL;
            const VectorType& seed = (s == 3) ? Psi0 : Phi[s];
            X[s][0].copy_from(seed);
            if (nk > 1) cheby.apply_op(X[s][0], X[s][1], mode);
            for (int n = 2; n < nk; ++n) {
                ::rsatb::util::throw_if_interrupted(graph->comm);
                const int applies_done = s * (nk - 1) + n - 1;
                if (rank == 0 &&
                    ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
                    ::rsatb::util::every_nth(applies_done, std::max(1, total_applies / 20))) {
                    std::cout << "ac: block " << iblock << " / " << n_blocks
                              << ", expansion " << applies_done << " / " << total_applies
                              << std::endl;
                }
                cheby.apply_op(X[s][n - 1], X[s][n], mode);
                X[s][n].axpby(-1.0, X[s][n - 2], 2.0);
            }
        }

        // Reconstruct every sample in the block from the shared states.
        for (int k = 1; k <= b * os; ++k) {
            ::rsatb::util::throw_if_interrupted(graph->comm);
            const auto& c = ctab[k];
            const int n_terms = static_cast<int>(c.size());
            for (int s = 0; s < 4; ++s) {
                R[s].copy_from(X[s][0]);
                R[s].scale(CoeffCast<T>::apply(c[0]));
                for (int n = 1; n < n_terms; ++n) {
                    R[s].axpy(CoeffCast<T>::apply(c[n]), X[s][n]);
                }
            }
            current_adjoint(cheby, Vx, Vy, Vz, R[3], J0, 1.0);
            record(done * os + k, J0, R);
            if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
                ::rsatb::util::every_nth(done * os + k - 1, std::max(1, ns / 20))) {
                std::cout << "ac: sample " << done * os + k << " / " << ns << std::endl;
            }
        }

        // Block-end states seed the next block. The reconstruction at
        // k = b*os is the same expansion of e^{-i Op b pi} the sequential
        // path would compose stepwise, accurate to the 1e-14 truncation.
        for (int i = 0; i < 3; ++i) Phi[i].copy_from(R[i]);
        Psi0.copy_from(R[3]);
        done += b;
    }
}

} // namespace calc
} // namespace rsatb

#endif
