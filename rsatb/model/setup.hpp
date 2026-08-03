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

struct PreconditionerOptions {
    PreconditionerTarget target = PreconditionerTarget::kOverlap;

    /// Frees the overlap matrix once the preconditioner is built.
    ///
    /// The preconditioner already carries the S^{-1/2} information the solvers
    /// need, so for a large system dropping S saves substantial memory. It
    /// also makes any observable that needs S explicitly unavailable, which is
    /// why this is a named option rather than a side effect of a mode string.
    bool release_overlap_after_build = false;

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
        "The former \"H\" mode is now preconditioner.release_overlap_after_build.");
}

struct SetupOptions {
    std::vector<std::vector<double>> kpoints = {{0.0, 0.0, 0.0}};
    int nsamples = 1;
    bool compute_bounds = true;
    std::optional<std::pair<double, double>> energy_bounds;
    double bounds_zero_tol = 1e-12;
    int bounds_max_matvecs = 10000;
    PreconditionerOptions preconditioner;
};

} // namespace model
} // namespace rsatb

#endif // RSATB_MODEL_SETUP_HPP
