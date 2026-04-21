#ifndef RSATB_CALC_DISORDER_HPP
#define RSATB_CALC_DISORDER_HPP

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace calc {

struct DisorderConfig {
    std::string kind = "none";
    double strength_ev = 0.0;
    std::uint64_t seed = 0;
    bool per_atom = true;

    bool enabled() const {
        return kind != "none" && std::abs(strength_ev) > 0.0;
    }
};

inline std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline double uniform_from_hash(std::uint64_t value) {
    const std::uint64_t mantissa = splitmix64(value) >> 11;
    return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);
}

template <typename T, typename Kernel = vbcsr::DefaultKernel<T>>
void apply_anderson_disorder(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::BlockSpMat<T, Kernel>& hamiltonian,
    const DisorderConfig& config
) {
    int start = 0;
    int end = 0;
    atomic_data.get_vertex_range_local(&start, &end);
    for (int atom = start; atom < end; ++atom) {
        if (atomic_data.atom_is_ghost(atom)) {
            continue;
        }

        const int global_atom = atomic_data.get_global_index(atom);
        auto row_it = hamiltonian.graph->global_to_local.find(global_atom);
        if (row_it == hamiltonian.graph->global_to_local.end()) {
            continue;
        }
        const int local_row = row_it->second;
        const int norb = hamiltonian.graph->block_sizes[local_row];
        std::vector<T> block(static_cast<size_t>(norb) * static_cast<size_t>(norb), static_cast<T>(0.0));

        for (int orb = 0; orb < norb; ++orb) {
            const std::uint64_t orbital_key = config.per_atom ? 0ULL : static_cast<std::uint64_t>(orb + 1);
            const double disorder =
                config.strength_ev *
                (uniform_from_hash(config.seed ^ (static_cast<std::uint64_t>(global_atom + 1) << 32) ^ orbital_key) - 0.5);
            block[static_cast<size_t>(orb) * static_cast<size_t>(norb) + static_cast<size_t>(orb)] =
                static_cast<T>(disorder);
        }

        hamiltonian.update_local_block(
            local_row, local_row, block.data(), norb, norb, vbcsr::AssemblyMode::ADD, vbcsr::MatrixLayout::ColMajor);
    }
}

template <typename T, typename Kernel = vbcsr::DefaultKernel<T>>
void apply_disorder(
    vbcsr::atomic::AtomicData& atomic_data,
    vbcsr::BlockSpMat<T, Kernel>& hamiltonian,
    const DisorderConfig& config
) {
    if (!config.enabled()) {
        return;
    }
    if (config.kind == "anderson") {
        apply_anderson_disorder(atomic_data, hamiltonian, config);
        return;
    }
    throw std::runtime_error("Unsupported disorder kind: " + config.kind);
}

} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_DISORDER_HPP
