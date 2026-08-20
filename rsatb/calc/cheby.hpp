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
#include <stdexcept>
#include <type_traits>

using namespace vbcsr;

namespace rsatb {
namespace calc {

enum class ChebyMode { NORMAL, ADJOINT, CONJUGATE };

// Which operator the recurrence iterates.
//
// Chosen at construction, never inferred from which of H/S/M happen to be
// present. "S was released because the sandwich form does not reference it"
// and "the basis is orthogonal" leave the same nullptr behind, and reading
// the form off that pointer is what let the sandwich case be mistaken for an
// orthogonal one wherever a call site dresses an auxiliary operator
// (apply_dual) or gates on orthogonality (msd.hpp).
enum class ChebyOpForm {
    Orthogonal,  // H        -- orthogonal basis; there is no metric
    Solve,       // S^-1 H   -- S kept, one CG solve per matvec
    Sandwich,    // Z^H H Z  -- S is never referenced; Hermitian when H is
};

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

// Chebyshev coefficients are carried as complex (time propagation needs
// i^n factors). A real-scalar operator takes their real part: the cast
// std::complex<double> -> double does not exist, which silently made
// propagate() complex-only.
template <typename T>
struct CoeffCast {
    static T apply(const std::complex<double>& c) { return static_cast<T>(c.real()); }
};
template <typename R>
struct CoeffCast<std::complex<R>> {
    static std::complex<R> apply(const std::complex<double>& c) {
        return static_cast<std::complex<R>>(c);
    }
};

// Chebyshev Recurrence
// Computes T_n(H_scaled) * v
// H_scaled = (H - center) / half_width
template <typename T>
class Chebyshev {
public:
    BlockSpMat<T>& H;
    BlockSpMat<T>* S = nullptr;
    Preconditioner<T>* M = nullptr;
    rsatb::calc::PreconditionerType ptype = rsatb::calc::PreconditionerType::Left;
    /// What apply_op applies. Read it instead of testing S or M: see ChebyOpForm.
    ChebyOpForm form = ChebyOpForm::Orthogonal;
    LinearSolver<T> solver;
    double center;
    double half_width;

    /// Orthogonal: the recurrence iterates H itself.
    Chebyshev(BlockSpMat<T>& ham, double emin, double emax, bool verbose = false) 
        : H(ham), S(nullptr), M(nullptr), form(ChebyOpForm::Orthogonal),
          solver(1e-8, 1000, verbose) {
        set_window(emin, emax);
    }

    /// Sandwich: the recurrence iterates Z^H H Z, where `pre` applies Z. S is
    /// not a parameter because this form never references it.
    Chebyshev(BlockSpMat<T>& ham, Preconditioner<T>& pre, rsatb::calc::PreconditionerType type, double emin, double emax, bool verbose = false)
        : H(ham), S(nullptr), M(&pre), ptype(type), form(ChebyOpForm::Sandwich),
          solver(1e-8, 1000, verbose) {
        if (type != rsatb::calc::PreconditionerType::Symmetric) {
            throw std::runtime_error(
                "Chebyshev: a preconditioner-only operator is the Z^H H Z sandwich and "
                "needs a Symmetric preconditioner. One-sided M H or H M is not similar "
                "to a Hermitian matrix and has no spectral expansion; pass an overlap "
                "for the S^-1 H form instead.");
        }
        set_window(emin, emax);
    }

    /// Solve: the recurrence iterates S^-1 H, one unpreconditioned CG per matvec.
    Chebyshev(BlockSpMat<T>& ham, BlockSpMat<T>& ovp, double emin, double emax, bool verbose = false) 
        : H(ham), S(&ovp), M(nullptr), form(ChebyOpForm::Solve),
          solver(1e-8, 1000, verbose) {
        set_window(emin, emax);
    }

    /// Solve, with `pre` preconditioning the inner CG (`type` governs that
    /// solve only -- it does not change which operator is iterated).
    Chebyshev(BlockSpMat<T>& ham, BlockSpMat<T>& ovp, Preconditioner<T>& pre, rsatb::calc::PreconditionerType type, double emin, double emax, bool verbose = false) 
        : H(ham), S(&ovp), M(&pre), ptype(type), form(ChebyOpForm::Solve),
          solver(1e-8, 1000, verbose) {
        set_window(emin, emax);
    }

    // True when the operator applied by apply_op is Hermitian: the orthogonal
    // form (H alone) and the sandwich form (Z^H H Z, Hermitian for any Z when
    // H is). The solve form applies a similarity transform of H that is not
    // Hermitian elementwise, so product-identity shortcuts (two KPM moments
    // per matvec) must not be used there.
    //
    // Why the solver mode (S kept, A = S^-1 H) has no doubling either way,
    // even though A is self-adjoint in the S inner product:
    //   <x|T_n(A) T_n(A)|x> = <T_n(A^H) x | T_n(A) x>,   A^H = H S^-1 = S A S^-1
    //                       = <T_n(A) (S^-1 x) | S | T_n(A) x>
    // -- the bra needs a SECOND Chebyshev chain seeded from S^-1 x: two
    // recurrences for two moments, no gain. The single-chain alternative
    // doubles the S-weighted moments <x|S T_n(A)|x>, but those estimate the
    // DOS trace only for sample vectors drawn with covariance S^-1
    // (x = Z eta), and then <x|S T_n(A)|x> = <eta|T_n(ZHZ)|eta> identically:
    // the sandwich estimator again, evaluated through a CG solve per matvec
    // instead of two SpMVs. The doubling exists only when the operator, the
    // dot product, and the sampling share one inner product -- exactly the
    // Symmetric-sandwich configuration this gate accepts.
    bool op_is_hermitian() const { return form != ChebyOpForm::Solve; }

    // Apply the scaled operator Op * x -> y, Op = (G - center) / half_width
    // with G the form's operator: H, S^-1 H, or Z^H H Z.
    //
    // ADJOINT applies G^H, CONJUGATE applies G conjugated elementwise. Only
    // the solve form distinguishes them -- H and Z^H H Z are Hermitian, so
    // there G^H = G and the mode is not consulted:
    //   Op = (S^-1 H - center) / half_width   (NORMAL)
    //   Op = (H^* S^-1^* - center) / half_width   (CONJUGATE)
    //   Op = (H S^-1 - center) / half_width   (ADJOINT)
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
        switch (form) {
            case ChebyOpForm::Orthogonal:
                // y = (H - center) x
                H.mult(const_cast<DistVector<T>&>(x), y);
                y.axpy(-center, x);
                break;
            case ChebyOpForm::Sandwich:
                // y = (Z^H H Z - center) x. Hermitian for Hermitian H and any
                // Z, so it is its own adjoint and `mode` does not enter.
                M->apply(x, y);
                H.mult(y, *v_tmp);
                M->apply_adjoint(*v_tmp, y);
                y.axpy(-center, x);
                break;
            case ChebyOpForm::Solve:
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    // y = S^-1 (H - center*S) x
                    H.mult(const_cast<DistVector<T>&>(x), *v_tmp);
                    if (M) solver.solve(*S, *v_tmp, y, *M, ptype);
                    else solver.solve(*S, *v_tmp, y);
                } else {
                    // y = (H - center*S) S^-1 x
                    if (M) solver.solve(*S, x, *v_tmp, *M, ptype);
                    else solver.solve(*S, x, *v_tmp);
                    H.mult_adjoint(*v_tmp, y);
                }
                y.axpy(-center, x);
                break;
        }
        y.scale(1.0 / half_width);
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
        switch (form) {
            case ChebyOpForm::Orthogonal:
                H.mult_dense(const_cast<DistMultiVector<T>&>(X), Y);
                Y.axpy(-center, X);
                break;
            case ChebyOpForm::Sandwich:
                M->apply(X, Y);
                H.mult_dense(Y, *mv_tmp);
                M->apply_adjoint(*mv_tmp, Y);
                Y.axpy(-center, X);
                break;
            case ChebyOpForm::Solve:
                if (mode == ChebyMode::NORMAL || mode == ChebyMode::CONJUGATE) {
                    H.mult_dense(const_cast<DistMultiVector<T>&>(X), *mv_tmp);
                    if (M) solver.solve(*S, *mv_tmp, Y, *M, ptype);
                    else solver.solve(*S, *mv_tmp, Y);
                } else {
                    if (M) solver.solve(*S, X, *mv_tmp, *M, ptype);
                    else solver.solve(*S, X, *mv_tmp);
                    H.mult_dense_adjoint(*mv_tmp, Y);
                }
                Y.axpy(-center, X);
                break;
        }
        Y.scale(1.0 / half_width);
    }

    // Applies an auxiliary operator V in the representation this Chebyshev
    // iterates: y = scale * (dressed V) x. Was the free apply_operator_dual().
    //
    //   Orthogonal   V                 -- no metric to carry
    //   Solve        S^-1 V  (NORMAL)  /  V S^-1  (ADJOINT)
    //   Sandwich     Z^H V Z           -- both modes
    //
    // ADJOINT applies the conjugate transpose of what NORMAL applies, which is
    // what the left/bra sequences of the transport correlators propagate. The
    // sandwich representation is orthonormal, so a Hermitian V dresses to a
    // Hermitian operator and the two modes coincide -- the same reason
    // apply_op does not consult `mode` in that form.
    //
    // Dressing V by the form rather than by `S != nullptr` is the whole point:
    // the sandwich case reached the orthogonal branch under the old test and
    // applied a bare V, dropping the Z on either side.
    template <typename VectorType>
    void apply_dual(BlockSpMat<T>& V, const VectorType& x, VectorType& y,
                    ChebyMode mode = ChebyMode::NORMAL, double scale = 1.0) {
        const bool adjoint = (mode == ChebyMode::ADJOINT);
        switch (form) {
            case ChebyOpForm::Orthogonal:
                mult_by(V, x, y);
                break;
            case ChebyOpForm::Sandwich: {
                VectorType& tmp = dual_tmp(x);
                M->apply(x, y);        // y   = Z x
                mult_by(V, y, tmp);    // tmp = V Z x
                M->apply_adjoint(tmp, y);
                break;
            }
            case ChebyOpForm::Solve: {
                // The solve target doubles as the CG initial guess, as it did
                // when this lived in ac.hpp: V x on the way in, zero on the
                // way out. Only the allocation is cached, not the contents.
                VectorType& tmp = dual_tmp(x);
                if (!adjoint) {
                    mult_by(V, x, y);      // y = V x, and the initial guess
                    tmp.copy_from(y);
                    if (M) solver.solve(*S, tmp, y, *M, ptype);
                    else solver.solve(*S, tmp, y);
                } else {
                    tmp.set_constant(T(0.0));
                    if (M) solver.solve(*S, x, tmp, *M, ptype);
                    else solver.solve(*S, x, tmp);
                    mult_by(V, tmp, y);
                }
                break;
            }
        }
        y.scale(T(scale));
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
        v_out.scale(CoeffCast<T>::apply(coeffs[0]));

        if (N > 1) {
            apply_op(*v_t0, *v_t1, mode); // t1 = T1
            v_out.axpy(CoeffCast<T>::apply(coeffs[1]), *v_t1);
        }

        for (int n = 2; n < N; ++n) {
            apply_op(*v_t1, *v_t2, mode);
            v_t2->axpby(-1.0, *v_t0, 2.0); // t2 = 2*Op*t1 - t0 = Tn
            
            v_out.axpy(CoeffCast<T>::apply(coeffs[n]), *v_t2);
            
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
        V_out.scale(CoeffCast<T>::apply(coeffs[0]));

        if (N > 1) {
            apply_op(*mv_t0, *mv_t1, mode); // T1 = T1
            V_out.axpy(CoeffCast<T>::apply(coeffs[1]), *mv_t1);
        }

        for (int n = 2; n < N; ++n) {
            apply_op(*mv_t1, *mv_t2, mode);
            mv_t2->axpby(-1.0, *mv_t0, 2.0); // T2 = 2*Op*T1 - T0 = Tn
            
            V_out.axpy(CoeffCast<T>::apply(coeffs[n]), *mv_t2);
            
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
    void set_window(double emin, double emax) {
        center = (emax + emin) / 2.0;
        half_width = (emax - emin) / 2.0;
    }

    /// mult / mult_dense, picked by vector kind so apply_dual is written once.
    template <typename VectorType>
    static void mult_by(BlockSpMat<T>& A, const VectorType& in, VectorType& out) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            A.mult_dense(const_cast<VectorType&>(in), out);
        } else {
            A.mult(const_cast<VectorType&>(in), out);
        }
    }

    /// Scratch for apply_dual, shaped like `like`. Kept separate from v_tmp /
    /// mv_tmp so a dressing can never land in the recurrence's scratch.
    template <typename VectorType>
    VectorType& dual_tmp(const VectorType& like) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            if (!mv_dual_tmp || mv_dual_tmp->num_vectors != like.num_vectors) {
                mv_dual_tmp = std::make_unique<DistMultiVector<T>>(like.graph,
                                                                  like.num_vectors);
            }
            return *mv_dual_tmp;
        } else {
            if (!v_dual_tmp) v_dual_tmp = std::make_unique<DistVector<T>>(like.graph);
            return *v_dual_tmp;
        }
    }

    std::unique_ptr<DistVector<T>> v_tmp, v_x_conj, v_t0, v_t1, v_t2, v_dual_tmp;
    std::unique_ptr<DistMultiVector<T>> mv_tmp, mv_x_conj, mv_t0, mv_t1, mv_t2, mv_dual_tmp;
};

} // namespace calc
} // namespace rsatb

#endif
