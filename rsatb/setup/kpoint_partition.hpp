#ifndef RSATB_SETUP_KPOINT_PARTITION_HPP
#define RSATB_SETUP_KPOINT_PARTITION_HPP

// Which k-points this rank is responsible for.
//
// Split out of OperatorSet::initialize, which used to do the partitioning, the
// H(k)/S(k) sampling, the preconditioner build, and the spectral-bound
// estimate in one loop. The partition is pure arithmetic on ranks and counts,
// so having it alone makes it testable without building any operators.

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rsatb {
namespace setup {

// The same type op::KPoint names (operator/types.hpp), spelled locally so
// this header keeps its no-operator-includes property; it must stay a plain
// std::vector<double> for the two to interconvert.
using KPoint = std::vector<double>;

/// The k-points owned by one rank, and the global count they came from.
struct KPointPartition {
    int global_nk = 0;
    /// (global index, k-point) pairs owned by this rank, in ascending index.
    std::vector<std::pair<int, KPoint>> local;

    bool owns(int k_idx) const {
        for (const auto& entry : local) {
            if (entry.first == k_idx) return true;
        }
        return false;
    }
};

/// How the k-point set is spread over sub-communicators.
struct KPointPartitionContext {
    int npar = 1;
    int world_rank = 0;
    int world_size = 1;
};

/// Assigns k-points to sub-communicators round-robin.
///
/// Every rank of a sub-communicator gets the same k-points, because the
/// sub-communicator is the unit that cooperates on one k-point at a time; the
/// second level of parallelism is over atoms within it.
///
/// Args:
///   kpoints: The full k-point set, shared by all ranks.
///   context: Rank layout and the requested sub-communicator count.
/// Returns:
///   This rank's share.
inline KPointPartition PartitionKPoints(
    const std::vector<KPoint>& kpoints,
    const KPointPartitionContext& context
) {
    if (context.world_size < 1) {
        throw std::runtime_error("KPointPartitionContext.world_size must be at least 1.");
    }
    if (context.world_rank < 0 || context.world_rank >= context.world_size) {
        throw std::runtime_error("KPointPartitionContext.world_rank is outside the communicator.");
    }

    KPointPartition partition;
    partition.global_nk = static_cast<int>(kpoints.size());

    const int npar = std::max(1, std::min(context.npar, context.world_size));
    const int color = (context.world_rank * npar) / context.world_size;

    for (int i = 0; i < partition.global_nk; ++i) {
        if (i % npar == color) {
            partition.local.emplace_back(i, kpoints[static_cast<size_t>(i)]);
        }
    }
    return partition;
}

}  // namespace setup
}  // namespace rsatb

#endif  // RSATB_SETUP_KPOINT_PARTITION_HPP
