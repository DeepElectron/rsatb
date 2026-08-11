#ifndef RSATB_MODEL_SETUP_HPP
#define RSATB_MODEL_SETUP_HPP

// Run options: what the calculation does, independent of which model family
// produces the operators.
//
// Model-specific configuration lives with its model -- see lcao_model.hpp and
// sktb/sktb_model.hpp. Nothing here names a model family.

#include "vbcsr/core/atomic/periodic_table.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rsatb {
namespace model {

/// Which operator the iterative solvers are preconditioned against.
enum class PreconditionerTarget {
    kNone,      // no preconditioner
    kOverlap,   // approximate S^{-1/2}
};

/// Which form of the non-orthogonal operator the Chebyshev expansions apply.
enum class OperatorForm {
    kSolve,     // S^{-1} H: S is kept, each matvec CG-solves S y = H x with
                // Z = S^{-1/2} as the preconditioner. The fallback for
                // systems where Z densifies beyond what memory allows.
    kSandwich,  // Z H Z, exactly Hermitian: S is released after the build
                // (saves its memory; observables that need S become
                // unavailable). Two SpMVs per matvec, and the Hermiticity
                // enables the two-moments-per-matvec KPM shortcut -- see
                // Chebyshev::op_is_hermitian for why kSolve cannot have it.
};

struct PreconditionerOptions {
    PreconditionerTarget target = PreconditionerTarget::kOverlap;

    /// Selects between the S^{-1}H and Z H Z forms above. Formerly the bool
    /// release_overlap_after_build (false = kSolve), named for its memory
    /// side effect rather than for the operator it selects.
    OperatorForm operator_form = OperatorForm::kSolve;

    double tol = 1e-7;
    double sptol = 1e-5;
    int maxiter = 20;

};

/// Parses a user-facing preconditioner target name.
inline PreconditionerTarget ParsePreconditionerTarget(const std::string& raw_name) {
    const std::string name = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_name));
    if (name == "none") return PreconditionerTarget::kNone;
    if (name == "s" || name == "overlap") return PreconditionerTarget::kOverlap;
    throw std::runtime_error(
        "Unsupported preconditioner target '" + raw_name + "'. Supported: none, overlap. "
        "The former \"H\" mode is now preconditioner.operator_form = \"sandwich\".");
}

/// Parses a user-facing operator form name.
inline OperatorForm ParseOperatorForm(const std::string& raw_name) {
    const std::string name = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(raw_name));
    if (name == "solve") return OperatorForm::kSolve;
    if (name == "sandwich") return OperatorForm::kSandwich;
    throw std::runtime_error(
        "Unsupported operator form '" + raw_name + "'. Supported: sandwich (Z H Z, "
        "S released), solve (S^-1 H by preconditioned CG, S kept). This option was "
        "formerly the bool release_overlap_after_build (true = sandwich).");
}

struct SetupOptions {
    std::vector<std::vector<double>> kpoints = {{0.0, 0.0, 0.0}};
    int nsamples = 1;
    /// Seed of the stochastic sample vectors. Fixed by default, so two runs of
    /// the same calculation agree: the draw is counter-based, hence also
    /// identical across rank counts and thread counts. Change it to get an
    /// independent estimate, not to get "more random" -- averaging two seeds is
    /// the same as doubling nsamples.
    unsigned long sample_seed = 20240917UL;
    bool compute_bounds = true;
    std::optional<std::pair<double, double>> energy_bounds;
    double bounds_zero_tol = 1e-12;
    int bounds_max_matvecs = 10000;
    PreconditionerOptions preconditioner;
};

} // namespace model
} // namespace rsatb

#endif // RSATB_MODEL_SETUP_HPP
