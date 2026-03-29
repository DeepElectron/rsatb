#ifndef RSATB_CALC_AC_HPP
#define RSATB_CALC_AC_HPP

#include "cheby.hpp"
#include "fermi_dirac.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

using namespace vbcsr;

namespace rsatb {
namespace calc {

// Compute Current: J = -i S^{-1} [H, R] = -S^{-1} V (if V = -i[H,R])
// Original code scales by -1.0 (assuming scale=1.0).
// We assume Vx, Vy, Vz are the velocity operators V.
// J = - S^{-1} V Psi
template <typename T, typename Kernel, typename VectorType>
void current(Chebyshev<T, Kernel>& cheby, 
             BlockSpMat<T, Kernel>& Vx, 
             BlockSpMat<T, Kernel>& Vy, 
             BlockSpMat<T, Kernel>& Vz, 
             const VectorType& Psi_in, 
             VectorType* J_Psi, // Array of 3
             double scale = 1.0) {
    
    // J_Psi[0] = Vx * Psi_in
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        Vx.mult_dense(const_cast<VectorType&>(Psi_in), J_Psi[0]);
        Vy.mult_dense(const_cast<VectorType&>(Psi_in), J_Psi[1]);
        Vz.mult_dense(const_cast<VectorType&>(Psi_in), J_Psi[2]);
    } else {
        Vx.mult(const_cast<VectorType&>(Psi_in), J_Psi[0]);
        Vy.mult(const_cast<VectorType&>(Psi_in), J_Psi[1]);
        Vz.mult(const_cast<VectorType&>(Psi_in), J_Psi[2]);
    }
    
    // Apply S^{-1} if needed
    if (cheby.S) {
        // Use temporary buffer for solve?
        // solver.solve(S, b, x)
        // We can solve in-place if supported, or use a temp.
        // LinearSolver::solve(A, b, x).
        // Let's use a temp.
        // Let's use a temp.
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
        VectorType tmp = construct_vec(Psi_in.graph, nv);

        for (int i = 0; i < 3; ++i) {
            tmp.copy_from(J_Psi[i]);
            if (cheby.M) cheby.solver.solve(*cheby.S, tmp, J_Psi[i], *cheby.M, cheby.ptype);
            else cheby.solver.solve(*cheby.S, tmp, J_Psi[i]);
        }
    }
    
    // Scale by -scale
    // Original code: nI(-scale, 0.0) -> -scale
    T val = static_cast<typename T::value_type>(-scale);
    J_Psi[0].scale(val);
    J_Psi[1].scale(val);
    J_Psi[2].scale(val);
}

// Compute Adjoint Current: J = - V S^{-1} Psi
template <typename T, typename Kernel, typename VectorType>
void current_adjoint(Chebyshev<T, Kernel>& cheby, 
                     BlockSpMat<T, Kernel>& Vx, 
                     BlockSpMat<T, Kernel>& Vy, 
                     BlockSpMat<T, Kernel>& Vz, 
                     const VectorType& Psi_in, 
                     VectorType* J_Psi, // Array of 3
                     double scale = 1.0) {
    
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
    VectorType tmp = construct_vec(Psi_in.graph, nv);

    // tmp = S^{-1} Psi_in
    if (cheby.S) {
        if (cheby.M) cheby.solver.solve(*cheby.S, Psi_in, tmp, *cheby.M, cheby.ptype);
        else cheby.solver.solve(*cheby.S, Psi_in, tmp);
    } else {
        tmp.copy_from(Psi_in);
    }
    
    // J_Psi = V * tmp
    // J_Psi = V * tmp
    // here we assume the velocity is strictly hermitian, so to use mult instead of mult_adjoint.
    if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
        Vx.mult_dense(tmp, J_Psi[0]);
        Vy.mult_dense(tmp, J_Psi[1]);
        Vz.mult_dense(tmp, J_Psi[2]);
    } else {
        Vx.mult(tmp, J_Psi[0]);
        Vy.mult(tmp, J_Psi[1]);
        Vz.mult(tmp, J_Psi[2]);
    }
    
    // Scale by -scale
    T val = static_cast<typename T::value_type>(-scale);
    J_Psi[0].scale(val);
    J_Psi[1].scale(val);
    J_Psi[2].scale(val);
}

// AC Conductivity Correlation
template <typename T, typename Kernel, typename VectorType>
void corr_ac(Chebyshev<T, Kernel>& cheby, 
             BlockSpMat<T, Kernel>& Vx, 
             BlockSpMat<T, Kernel>& Vy, 
             BlockSpMat<T, Kernel>& Vz, 
             const VectorType& Psi_in, 
             int nt, 
             double temp, 
             double mu, 
             double scale, // Unused, relying on cheby
             double shift, // Unused, relying on cheby
             int n_kpm, 
             double eps = 1e-14, 
             std::vector<std::complex<double>>* cac = nullptr, 
             bool verbose = true) {
    
    DistGraph* graph = Psi_in.graph;
    int rank = graph->rank;
    
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
    fermi_dirac_kpm(cheby, Psi_in, Psi0, temp, mu, n_kpm, eps, ChebyMode::ADJOINT, verbose);
    // Psi_in_copy.conjugate(); // Restore (though we use copy)

    // 3. Compute J_Psi = current(Psi_in)
    current(cheby, Vx, Vy, Vz, Psi_in, J_Psi, 1.0);

    // 4. Apply (1-f) to J_Psi
    // (1-f) J Psi = J Psi - f(J Psi)
    for (int i = 0; i < 3; ++i) {
        fermi_dirac_kpm(cheby, J_Psi[i], buff, temp, mu, n_kpm, eps, ChebyMode::NORMAL, verbose);
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
        if (rank == 0 && verbose) std::cout << "corr_ac calculation: " << it + 1 << " / " << nt << "\n";

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
