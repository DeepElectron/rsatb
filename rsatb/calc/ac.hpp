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
    
    // 1. Generate Time Propagation Coefficients
    double dt = M_PI; // Fixed in original code?
    // Let's stick to `dt = M_PI` as argument to `timeprop_coeff` as in original code.
    
    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(dt, eps, true, coeffs);
    int n_bes = coeffs.size();

    // Initialize cac
    if (cac!=nullptr) {
        if (cac->size()<(nt+1)*6) {
            cac->resize((nt+1)*6);
        }
        std::fill(cac->data(), cac->data() + (nt + 1) * 6, 0.0);
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
    for (int it = 0; it < nt; ++it) {
        ::rsatb::util::throw_if_interrupted(graph->comm);
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
            ::rsatb::util::every_nth(it, std::max(1, nt / 20))) {
            std::cout << "ac: step " << it + 1 << " / " << nt << std::endl;
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

} // namespace calc
} // namespace rsatb

#endif
