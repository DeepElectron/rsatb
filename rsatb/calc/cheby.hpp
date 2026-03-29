#ifndef RSATB_CALC_CHEBY_HPP
#define RSATB_CALC_CHEBY_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "preconditioner.hpp"
#include "solver.hpp"
#include "bessel.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <memory>

using namespace vbcsr;

namespace rsatb {
namespace calc {

enum class ChebyMode { NORMAL, ADJOINT, CONJUGATE };

// Helper to generate time propagation coefficients
inline void timeprop_coeff(double time_step, double precision, bool forward,
                         std::vector<std::complex<double>>& coeffs) {
    double *bes_vec;
    int n_bes;
    build_bessel_vector(time_step, precision, &bes_vec, &n_bes);

    coeffs.resize(n_bes);
    const std::complex<double> I(0.0, 1.0);
    
    if (forward) {
        for (int n = 0; n < n_bes; ++n) {
            coeffs[n] = 2.0 * std::pow(-I, n) * bes_vec[n];
        }
    } else {
        for (int n = 0; n < n_bes; ++n) {
            coeffs[n] = 2.0 * std::pow(I, n) * bes_vec[n];
        }
    }
    
    coeffs[0] *= 0.5;
    delete[] bes_vec;
}

// Chebyshev Recurrence
// Computes T_n(H_scaled) * v
// H_scaled = (H - center) / half_width
template <typename T, typename Kernel = DefaultKernel<T>>
class Chebyshev {
public:
    BlockSpMat<T, Kernel>& H;
    BlockSpMat<T, Kernel>* S = nullptr;
    Preconditioner<T, Kernel>* M = nullptr;
    rsatb::calc::PreconditionerType ptype = rsatb::calc::PreconditionerType::Left;
    LinearSolver<T, Kernel> solver;
    double center;
    double half_width;

    Chebyshev(BlockSpMat<T, Kernel>& ham, double emin, double emax, bool verbose = false) 
        : H(ham), S(nullptr), M(nullptr), solver(1e-8, 1000, verbose) {
        center = (emax + emin) / 2.0;
        half_width = (emax - emin) / 2.0;
    }

    Chebyshev(BlockSpMat<T, Kernel>& ham, Preconditioner<T, Kernel>& pre, rsatb::calc::PreconditionerType type, double emin, double emax, bool verbose = false)
        : H(ham), S(nullptr), M(&pre), ptype(type), solver(1e-8, 1000, verbose) {
        center = (emax + emin) / 2.0;
        half_width = (emax - emin) / 2.0;
    }

    Chebyshev(BlockSpMat<T, Kernel>& ham, BlockSpMat<T, Kernel>& ovp, double emin, double emax, bool verbose = false) 
        : H(ham), S(&ovp), M(nullptr), solver(1e-8, 1000, verbose) {
        center = (emax + emin) / 2.0;
        half_width = (emax - emin) / 2.0;
    }

    Chebyshev(BlockSpMat<T, Kernel>& ham, BlockSpMat<T, Kernel>& ovp, Preconditioner<T, Kernel>& pre, rsatb::calc::PreconditionerType type, double emin, double emax, bool verbose = false) 
        : H(ham), S(&ovp), M(&pre), ptype(type), solver(1e-8, 1000, verbose) {
        center = (emax + emin) / 2.0;
        half_width = (emax - emin) / 2.0;
    }

    // Apply scaled operator Op * x -> y
    // Op = (S^-1 H - center) / half_width  (NORMAL)
    // Op = (H^*S^-1* - center) / half_width  (CONJUGATE)
    // Op = (H S^-1 - center) / half_width  (ADJOINT)
    void apply_op(const DistVector<T>& x, DistVector<T>& y, ChebyMode mode = ChebyMode::NORMAL) {
        if (!v_tmp) v_tmp = std::make_unique<DistVector<T>>(x.graph);
        if (mode == ChebyMode::CONJUGATE) {
            if (!v_x_conj) v_x_conj = std::make_unique<DistVector<T>>(x.graph);
            v_x_conj->copy_from(x);
            v_x_conj->conjugate();
            apply_op_internal(*v_x_conj, y, mode);
            y.conjugate();
        } else {
            apply_op_internal(x, y, mode);
        }
    }

    // we assume H and S is hermitian here.
    void apply_op_internal(const DistVector<T>& x, DistVector<T>& y, ChebyMode mode) {
        if (S == nullptr && M == nullptr) {
            // Orthogonal case: y = (H - center*I) x / half_width
            H.mult(const_cast<DistVector<T>&>(x), y);
            y.axpy(-center, x);
            y.scale(1.0 / half_width);
        } else if (S == nullptr && M != nullptr) {
            // Direct Preconditioned case: y = (MHM - center*I) x / half_width (Symmetric)
            if (ptype == rsatb::calc::PreconditionerType::Symmetric) {
                M->apply(x, y);
                H.mult(y, *v_tmp);
                M->apply_adjoint(*v_tmp, y);
                y.axpy(-center, x);
            } else if (ptype == rsatb::calc::PreconditionerType::Left) {
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    H.mult(const_cast<DistVector<T>&>(x), *v_tmp);
                    M->apply(*v_tmp, y);
                    y.axpy(-center, x);
                } else {
                    M->apply(x, *v_tmp);
                    H.mult(*v_tmp, y);
                    y.axpy(-center, x);
                }
            } else { // Right
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    M->apply(x, *v_tmp);
                    H.mult(*v_tmp, y);
                    y.axpy(-center, x);
                } else {
                    H.mult(const_cast<DistVector<T>&>(x), *v_tmp);
                    M->apply(*v_tmp, y);
                    y.axpy(-center, x);
                }
            }
            y.scale(1.0 / half_width);
        } else {
            // Non-orthogonal case with solver
            if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                // y = S^-1 (H - center*S) x / half_width
                H.mult(const_cast<DistVector<T>&>(x), *v_tmp);
                if (M) solver.solve(*S, *v_tmp, y, *M, ptype);
                else solver.solve(*S, *v_tmp, y);
                
                y.axpy(-center, x);
            } else {
                // y = (H - center*S) S^-1 x / half_width
                if (M) solver.solve(*S, x, *v_tmp, *M, ptype);
                else solver.solve(*S, x, *v_tmp);
                
                H.mult_adjoint(*v_tmp, y);
                y.axpy(-center, x);
            }
            y.scale(1.0 / half_width);
        }
    }

    void apply_op(const DistMultiVector<T>& X, DistMultiVector<T>& Y, ChebyMode mode = ChebyMode::NORMAL) {
        if (!mv_tmp || mv_tmp->num_vectors != X.num_vectors) {
            mv_tmp = std::make_unique<DistMultiVector<T>>(X.graph, X.num_vectors);
        }
        if (mode == ChebyMode::CONJUGATE) {
            if (!mv_x_conj || mv_x_conj->num_vectors != X.num_vectors) {
                mv_x_conj = std::make_unique<DistMultiVector<T>>(X.graph, X.num_vectors);
            }
            mv_x_conj->copy_from(X);
            mv_x_conj->conjugate();
            apply_op_dense_internal(*mv_x_conj, Y, mode);
            Y.conjugate();
        } else {
            apply_op_dense_internal(X, Y, mode);
        }
    }

    void apply_op_dense_internal(const DistMultiVector<T>& X, DistMultiVector<T>& Y, ChebyMode mode) {
        if (S == nullptr && M == nullptr) {
            // Orthogonal case
            H.mult_dense(const_cast<DistMultiVector<T>&>(X), Y);
            Y.axpy(-center, X);
            Y.scale(1.0 / half_width);
        } else if (S == nullptr && M != nullptr) {
            // Direct Preconditioned case
            if (ptype == rsatb::calc::PreconditionerType::Symmetric) {
                M->apply(X, Y);
                H.mult_dense(Y, *mv_tmp);
                M->apply_adjoint(*mv_tmp, Y);
                Y.axpy(-center, X);
            } else if (ptype == rsatb::calc::PreconditionerType::Left) {
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    H.mult_dense(const_cast<DistMultiVector<T>&>(X), *mv_tmp);
                    M->apply(*mv_tmp, Y);
                    Y.axpy(-center, X);
                } else {
                    M->apply(X, *mv_tmp);
                    H.mult_dense(*mv_tmp, Y);
                    Y.axpy(-center, X);
                }
            } else { // Right
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    M->apply(X, *mv_tmp);
                    H.mult_dense(*mv_tmp, Y);
                    Y.axpy(-center, X);
                } else {
                    H.mult_dense(const_cast<DistMultiVector<T>&>(X), *mv_tmp);
                    M->apply(*mv_tmp, Y);
                    Y.axpy(-center, X);
                }
            }
            Y.scale(1.0 / half_width);
        } else {
            // Non-orthogonal case
            if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                H.mult_dense(const_cast<DistMultiVector<T>&>(X), *mv_tmp);
                if (M) solver.solve(*S, *mv_tmp, Y, *M, ptype);
                else solver.solve(*S, *mv_tmp, Y);
                Y.axpy(-center, X);
            } else {
                if (M) solver.solve(*S, X, *mv_tmp, *M, ptype);
                else solver.solve(*S, X, *mv_tmp);
                H.mult_dense_adjoint(*mv_tmp, Y);
                Y.axpy(-center, X);
            }
            Y.scale(1.0 / half_width);
        }
    }

    // Propagate vector: v_out = sum c_n T_n(Op) v_in
    void propagate(const DistVector<T>& v_in, DistVector<T>& v_out, const std::vector<std::complex<double>>& coeffs, ChebyMode mode = ChebyMode::NORMAL) {
        int N = coeffs.size();
        if (N == 0) return;

        if (!v_t0) v_t0 = std::make_unique<DistVector<T>>(v_in.graph);
        if (!v_t1) v_t1 = std::make_unique<DistVector<T>>(v_in.graph);
        if (!v_t2) v_t2 = std::make_unique<DistVector<T>>(v_in.graph);

        v_t0->copy_from(v_in); // T0
        
        // v_out = c0 * T0
        v_out.copy_from(*v_t0);
        v_out.scale(static_cast<T>(coeffs[0]));

        if (N > 1) {
            apply_op(*v_t0, *v_t1, mode); // t1 = T1
            v_out.axpy(static_cast<T>(coeffs[1]), *v_t1);
        }

        for (int n = 2; n < N; ++n) {
            apply_op(*v_t1, *v_t2, mode);
            v_t2->axpby(-1.0, *v_t0, 2.0); // t2 = 2*Op*t1 - t0 = Tn
            
            v_out.axpy(static_cast<T>(coeffs[n]), *v_t2);
            
            v_t0->swap(*v_t1);
            v_t1->swap(*v_t2);
        }
    }

    // Propagate multivector: V_out = sum c_n T_n(Op) V_in
    void propagate(const DistMultiVector<T>& V_in, DistMultiVector<T>& V_out, const std::vector<std::complex<double>>& coeffs, ChebyMode mode = ChebyMode::NORMAL) {
        int N = coeffs.size();
        if (N == 0) return;

        if (!mv_t0 || mv_t0->num_vectors != V_in.num_vectors) {
            mv_t0 = std::make_unique<DistMultiVector<T>>(V_in.graph, V_in.num_vectors);
        }
        if (!mv_t1 || mv_t1->num_vectors != V_in.num_vectors) {
            mv_t1 = std::make_unique<DistMultiVector<T>>(V_in.graph, V_in.num_vectors);
        }
        if (!mv_t2 || mv_t2->num_vectors != V_in.num_vectors) {
            mv_t2 = std::make_unique<DistMultiVector<T>>(V_in.graph, V_in.num_vectors);
        }

        mv_t0->copy_from(V_in); // T0
        
        // V_out = c0 * T0
        V_out.copy_from(*mv_t0);
        V_out.scale(static_cast<T>(coeffs[0]));

        if (N > 1) {
            apply_op(*mv_t0, *mv_t1, mode); // T1 = T1
            V_out.axpy(static_cast<T>(coeffs[1]), *mv_t1);
        }

        for (int n = 2; n < N; ++n) {
            apply_op(*mv_t1, *mv_t2, mode);
            mv_t2->axpby(-1.0, *mv_t0, 2.0); // T2 = 2*Op*T1 - T0 = Tn
            
            V_out.axpy(static_cast<T>(coeffs[n]), *mv_t2);
            
            mv_t0->swap(*mv_t1);
            mv_t1->swap(*mv_t2);
        }
    }

    // Apply H_scaled * x -> y (Legacy, for backward compatibility)
    void apply_scaled_H(const DistVector<T>& x, DistVector<T>& y) {
        apply_op(x, y, ChebyMode::NORMAL);
    }

    // Apply H_scaled * X -> Y (Legacy)
    void apply_scaled_H(const DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        apply_op(X, Y, ChebyMode::NORMAL);
    }

    // Compute moments <v| T_n(H) |v> for n=0..N
    // Uses KPM recurrence
    std::vector<T> compute_moments(const DistVector<T>& v, int N) {
        std::vector<T> moments(N);
        
        if (!v_t0) v_t0 = std::make_unique<DistVector<T>>(v.graph);
        if (!v_t1) v_t1 = std::make_unique<DistVector<T>>(v.graph);
        if (!v_t2) v_t2 = std::make_unique<DistVector<T>>(v.graph);

        v_t0->copy_from(v); // T_0 |v> = |v>
        
        // mu_0 = <v|v>
        moments[0] = v_t0->dot(*v_t0);
        
        if (N > 1) {
            // T_1 |v> = H_scaled |v>
            apply_op(*v_t0, *v_t1, ChebyMode::NORMAL);
            moments[1] = v.dot(*v_t1); // <v| T_1 |v>
        }
        
        for (int n = 2; n < N; ++n) {
            apply_op(*v_t1, *v_t2, ChebyMode::NORMAL);
            v_t2->axpby(-1.0, *v_t0, 2.0);
            
            moments[n] = v.dot(*v_t2);
            
            v_t0->swap(*v_t1);
            v_t1->swap(*v_t2);
        }
        
        return moments;
    }
    
    // Compute moments for multivector (Batched)
    std::vector<std::vector<T>> compute_moments(const DistMultiVector<T>& V, int N) {
        int num_vecs = V.num_vectors;
        std::vector<std::vector<T>> moments(N, std::vector<T>(num_vecs));
        
        if (!mv_t0 || mv_t0->num_vectors != V.num_vectors) {
            mv_t0 = std::make_unique<DistMultiVector<T>>(V.graph, V.num_vectors);
        }
        if (!mv_t1 || mv_t1->num_vectors != V.num_vectors) {
            mv_t1 = std::make_unique<DistMultiVector<T>>(V.graph, V.num_vectors);
        }
        if (!mv_t2 || mv_t2->num_vectors != V.num_vectors) {
            mv_t2 = std::make_unique<DistMultiVector<T>>(V.graph, V.num_vectors);
        }

        mv_t0->copy_from(V);
        
        // mu_0
        mv_t0->dot(*mv_t0, moments[0]);
        
        if (N > 1) {
            apply_op(*mv_t0, *mv_t1, ChebyMode::NORMAL);
            V.dot(*mv_t1, moments[1]);
        }
        
        for (int n = 2; n < N; ++n) {
            apply_op(*mv_t1, *mv_t2, ChebyMode::NORMAL);
            mv_t2->axpby(-1.0, *mv_t0, 2.0);
            
            V.dot(*mv_t2, moments[n]);
            
            mv_t0->swap(*mv_t1);
            mv_t1->swap(*mv_t2);
        }
        
        return moments;
    }

private:
    std::unique_ptr<DistVector<T>> v_tmp, v_x_conj, v_t0, v_t1, v_t2;
    std::unique_ptr<DistMultiVector<T>> mv_tmp, mv_x_conj, mv_t0, mv_t1, mv_t2;
};

} // namespace calc
} // namespace rsatb

#endif
