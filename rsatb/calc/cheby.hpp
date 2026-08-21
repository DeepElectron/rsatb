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

// HOW the operator is applied -- an application style, not an operator.
//
// NORMAL applies A. ADJOINT applies A^H, which is what a left (bra) sequence
// propagates: the correlators store the left vector as an ordinary ket and
// conjugate only at the final contraction (fftcheb_transport.hpp,
// `conj(left) * right`), so advancing <L|A means advancing the stored ket by
// A^H. Applying A^T there instead would be wrong for any complex H, and the
// transport tests would catch it: their reference-checked fixture is complex
// Hermitian, where A^H = A but A^T = conj(A) differs.
//
// A^H is reached WITHOUT the transposed matrix kernel. H is Hermitian (see
// the class note), so every per-factor adjoint is an identity and only the
// ORDER of the factors turns around; see apply_form.
//
// That is worth doing because BlockSpMat::mult_adjoint is not uniformly
// cheap. Measured on a 32^3 lattice of 4-orbital sites, complex Hermitian,
// mult_dense_adjoint against the forward mult_dense (nv = 8), ms per apply:
//
//   threads          1      4      8     16
//   BSR  (uniform  15.5    9.4   11.5   17.8    <- gets SLOWER with threads
//         blocks)
//   VBCSR (varying 25.7    6.6    2.2    1.4
//         blocks)
//
// The VBCSR kernel gathers by column from a cached plan and scales; the BSR
// kernel gives each thread a private full-size copy of Y and reduces them
// afterwards, so both its time and its memory grow with the thread count
// (measured +threads x |Y|: 7.8 MB at 4 threads, 33.9 MB at 16, against a
// flat 3.8 MB for the VBCSR plan). Uniform orbital counts -- an all-carbon
// cell, say -- select BSR, so the large real-space runs land on the bad one.
// Until that kernel is fixed, a recurrence that applied the operator through
// it once per Chebyshev order would forfeit the thread scaling of the whole
// calculation.
//
enum class ChebyMode { NORMAL, ADJOINT };

// WHICH operator the recurrence iterates -- an operator shape, not a style.
//
// These three are the entire configurable space, and they are exactly the
// forms the input layer accepts (model/setup.hpp OperatorForm, which parses
// only "solve" and "sandwich"). A caller selects the physics -- is there a
// metric, and is it carried by a solve or by a congruence -- never an order
// of application.
//
// H S^-1 is deliberately NOT a fourth choice. It is what the solve form
// becomes under ChebyMode::ADJOINT and is reachable only that way; see
// apply_form, which derives it instead of naming it.
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
//
// H MUST BE HERMITIAN, and so must any auxiliary operator handed to
// apply_dual. This is a precondition of the method, not an assumption of
// convenience: the recurrence expands a function over the REAL interval
// [emin, emax], which presupposes a real spectrum, and the sandwich
// constructor below already refuses one-sided preconditioners on exactly
// that ground ("not similar to a Hermitian matrix and has no spectral
// expansion"). Every form is H itself, a congruence of H, or a similarity
// transform of H; the velocity operators V = -i[H, R] that apply_dual
// dresses are Hermitian whenever H is.
//
// The adjoint path leans on this: H^H = H, so ADJOINT reverses the order of
// the factors and nothing else. Handing this class a non-Hermitian H would
// not merely make ADJOINT wrong, it would leave the expansion itself
// undefined.
template <typename T>
class Chebyshev {
public:
    BlockSpMat<T>& H;
    BlockSpMat<T>* S = nullptr;
    Preconditioner<T>* M = nullptr;
    rsatb::calc::PreconditionerType ptype = rsatb::calc::PreconditionerType::Left;
    /// What apply_op applies. Read it instead of testing S or M: see ChebyOpForm.
    ///
    /// const because the constructors are what pair a form with the operands
    /// it needs -- Solve with an S to solve against, Sandwich with a
    /// Symmetric Z. A writable form let a caller move to Solve on an object
    /// built without an overlap, and the null S surfaced as a segfault inside
    /// CG rather than as the configuration error it was.
    const ChebyOpForm form = ChebyOpForm::Orthogonal;
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

    /// y = A x (NORMAL) or A^H x (ADJOINT), for the operator this was built for.
    ///
    /// `mode` is reduced to a single bool here and nowhere else, so the kernel
    /// below states each form once and reads no mode at all.
    void apply_op(const DistVector<T>& x, DistVector<T>& y, ChebyMode mode = ChebyMode::NORMAL) {
        apply_form(mode == ChebyMode::ADJOINT, x, y);
    }

    void apply_op(const DistMultiVector<T>& X, DistMultiVector<T>& Y,
                  ChebyMode mode = ChebyMode::NORMAL) {
        apply_form(mode == ChebyMode::ADJOINT, X, Y);
    }

    // Applies an auxiliary operator V in the representation this Chebyshev
    // iterates: y = scale * (dressed V) x. Was the free apply_operator_dual().
    //
    //   Orthogonal   V           Solve   S^-1 V      Sandwich   Z^H V Z
    //
    // ADJOINT reverses the factors by the same rule and in the same order as
    // apply_form -- V is Hermitian here, as the class note explains -- so the
    // operator and its dual cannot disagree about what "the adjoint of this
    // representation" means. Dressing V by the form rather than by
    // `S != nullptr` is the whole point: the sandwich case reached the
    // orthogonal branch under the old test and applied a bare V, dropping the
    // Z on either side.
    template <typename VectorType>
    void apply_dual(BlockSpMat<T>& V, const VectorType& x, VectorType& y,
                    ChebyMode mode = ChebyMode::NORMAL, double scale = 1.0) {
        const bool dagger = (mode == ChebyMode::ADJOINT);
        switch (form) {
            case ChebyOpForm::Orthogonal:
                // No scratch: taking dual_tmp() here would make every
                // orthogonal-basis caller (all of ac.hpp and msd.hpp on such a
                // basis) carry a full Nbasis x nv buffer it never reads.
                mult_by(V, x, y);
                break;
            case ChebyOpForm::Sandwich: {
                VectorType& tmp = dual_tmp(x);
                M->apply(x, y);      // y   = Z x
                mult_by(V, y, tmp);  // tmp = V Z x
                M->apply_adjoint(tmp, y);
                break;
            }
            case ChebyOpForm::Solve: {
                VectorType& tmp = dual_tmp(x);
                if (dagger) {
                    // y = V S^-1 x
                    tmp.set_constant(T(0.0));
                    solve_with_S(x, tmp);
                    mult_by(V, tmp, y);
                } else {
                    // y = S^-1 V x. The solve target doubles as the CG initial
                    // guess, as it did when this lived in ac.hpp: V x on the
                    // way in. Only the allocation is cached, not the contents.
                    mult_by(V, x, y);
                    tmp.copy_from(y);
                    solve_with_S(tmp, y);
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

    /// The scaled operator, applied to a vector or a multivector; A^H when
    /// `dagger`. Written once over both vector kinds, so the vector and
    /// multivector paths cannot drift.
    ///
    /// There is no table of adjoint shapes to keep in step with the forms:
    /// the adjoint of a product is the product of the adjoints in REVERSED
    /// order, and since H is Hermitian (see the class note) and S^-1 is
    /// self-adjoint for a Hermitian overlap, every per-factor adjoint is an
    /// identity. `dagger` collapses to the reversal alone:
    ///
    ///   (H)^H       = H         one factor; nothing to reverse
    ///   (S^-1 H)^H  = H S^-1    the solve moves to the other side
    ///   (Z^H H Z)^H = Z^H H Z   reverses onto itself
    ///
    /// Two consequences. Only the solve form has an adjoint distinct from
    /// itself, which is why the other two ignore `dagger` rather than
    /// checking it. And no branch applies H, S or V through the transposed
    /// kernel, so a left sequence costs exactly a right sequence and threads
    /// like one (see the ChebyMode note).
    ///
    /// The sandwich branch is the one exception, and it is not about `dagger`:
    /// Z^H H Z needs a genuine Z^H in BOTH modes, and Preconditioner::
    /// apply_adjoint reaches it through mult_adjoint (preconditioner.hpp,
    /// ISQRTMPreconditioner). So one of that form's three applies pays the
    /// scatter kernel per matvec whichever mode is asked for. isqrtm builds Z
    /// Hermitian by construction (isqrtm.hpp: "every route that forms it
    /// mirrors"), so Z^H = Z and that apply COULD use the forward kernel --
    /// an optimisation belonging to the preconditioner, not here, since this
    /// class must not assume a Preconditioner is self-adjoint.
    ///
    /// H S^-1 is derived here rather than named in ChebyOpForm: it is what
    /// asking the solve form for its adjoint means, not a shape a caller can
    /// configure.
    template <typename VectorType>
    void apply_form(bool dagger, const VectorType& x, VectorType& y) {
        VectorType& tmp = op_tmp(x);
        switch (form) {
            case ChebyOpForm::Orthogonal:
                // y = (H - center) x, its own adjoint
                mult_by(H, x, y);
                break;
            case ChebyOpForm::Sandwich:
                // y = (Z^H H Z - center) x, its own adjoint
                M->apply(x, y);
                mult_by(H, y, tmp);
                M->apply_adjoint(tmp, y);
                break;
            case ChebyOpForm::Solve:
                if (dagger) {
                    // y = (H - center S) S^-1 x
                    //
                    // tmp is the solve target, so it is also the unpreconditioned
                    // CG initial guess. Zeroed rather than left holding whatever
                    // the previous matvec parked there: that is scratch-reuse
                    // detail, and letting it steer CG makes the result depend on
                    // it. The NORMAL branch solves into the caller's y, whose
                    // prior contents are a genuine warm start.
                    tmp.set_constant(T(0.0));
                    solve_with_S(x, tmp);
                    mult_by(H, tmp, y);
                } else {
                    // y = S^-1 (H - center S) x
                    mult_by(H, x, tmp);
                    solve_with_S(tmp, y);
                }
                break;
        }
        y.axpy(-center, x);
        y.scale(1.0 / half_width);
    }

    /// A x, by the forward row-gather kernel. Picked by vector kind so the
    /// shape kernel and apply_dual are each written once over both.
    ///
    /// There is no adjoint counterpart on purpose: A^H = A for every matrix
    /// this class applies, so mult_adjoint would buy nothing and cost the
    /// thread scaling measured in the ChebyMode note.
    template <typename VectorType>
    static void mult_by(BlockSpMat<T>& A, const VectorType& in, VectorType& out) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            A.mult_dense(const_cast<VectorType&>(in), out);
        } else {
            A.mult(const_cast<VectorType&>(in), out);
        }
    }

    /// Solves S out = in, preconditioned when a preconditioner was supplied.
    /// `ptype` governs this solve alone -- it does not select an operator.
    template <typename VectorType>
    void solve_with_S(const VectorType& in, VectorType& out) {
        if (M) solver.solve(*S, const_cast<VectorType&>(in), out, *M, ptype);
        else solver.solve(*S, const_cast<VectorType&>(in), out);
    }

    /// Scratch for the shape kernel, shaped like `like`.
    template <typename VectorType>
    VectorType& op_tmp(const VectorType& like) {
        if constexpr (std::is_same_v<VectorType, DistMultiVector<T>>) {
            if (!mv_tmp || mv_tmp->num_vectors != like.num_vectors) {
                mv_tmp = std::make_unique<DistMultiVector<T>>(like.graph, like.num_vectors);
            }
            return *mv_tmp;
        } else {
            if (!v_tmp) v_tmp = std::make_unique<DistVector<T>>(like.graph);
            return *v_tmp;
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

    std::unique_ptr<DistVector<T>> v_tmp, v_t0, v_t1, v_t2, v_dual_tmp;
    std::unique_ptr<DistMultiVector<T>> mv_tmp, mv_t0, mv_t1, mv_t2, mv_dual_tmp;
};

} // namespace calc
} // namespace rsatb

#endif
