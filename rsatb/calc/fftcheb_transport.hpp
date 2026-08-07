#ifndef RSATB_CALC_FFTCHEB_TRANSPORT_HPP
#define RSATB_CALC_FFTCHEB_TRANSPORT_HPP

#include "../utils/interrupt.hpp"
#include "../utils/profile.hpp"
#include "ac.hpp"
#include "fft.hpp"
#include "kpm.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rsatb {
namespace calc {

/*
 * FFTCheb bulk transport implementation.
 *
 * The key idea is to avoid forming the full M x M moment tensor used by the
 * conventional spectral KPM transport formulas. Instead we:
 * 1. generate Chebyshev auxiliary sequences in real space,
 * 2. sweep those sequences row-by-row (or row-chunk-by-row-chunk),
 * 3. map each row to energy-space vectors with FFT-friendly transforms, and
 * 4. accumulate the needed rowwise dot products immediately.
 *
 * This keeps the implementation close to the paper while fitting the existing
 * rsatb conventions:
 * - `current(...)` / `current_adjoint(...)` provide the non-orthogonal
 *   velocity placement already used by `compute_ac`.
 * - `Chebyshev` owns the S^{-1}H / adjoint actions, so this module only
 *   orchestrates recurrences and energy-space post-processing.
 * - partitioning trades extra recurrence passes for lower peak memory by
 *   storing only a slice of the real-space rows at a time.
 */

enum class DCFormulation {
    Greenwood,
    Bastin
};

// Map user-facing strings onto the two transport formulas implemented here.
inline DCFormulation parse_dc_formulation(const std::string& formulation) {
    if (formulation == "greenwood") {
        return DCFormulation::Greenwood;
    }
    if (formulation == "bastin") {
        return DCFormulation::Bastin;
    }
    throw std::runtime_error("Unknown dc formulation: " + formulation);
}

// Upper-triangular conductivity-tensor storage order:
// (xx, xy, xz, yy, yz, zz).
inline int conductivity_component_index(int a, int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return int((-a * a + 5 * a) / 2 + b);
}

inline double fermi_dirac_energy(double energy, double mu, double temp) {
    if (temp <= 0.0) {
        return energy <= mu ? 1.0 : 0.0;
    }

    const double kb = 8.617333262e-5; // eV/K
    const double beta = 1.0 / (temp * kb);
    if (energy >= mu) {
        const double tmp = std::exp(beta * (mu - energy));
        return tmp / (1.0 + tmp);
    }

    const double tmp = std::exp(beta * (energy - mu));
    return 1.0 / (1.0 + tmp);
}

inline double fermi_dirac_derivative_energy(double energy, double mu, double temp) {
    if (temp <= 0.0) {
        return 0.0;
    }

    const double kb = 8.617333262e-5; // eV/K
    const double beta = 1.0 / (temp * kb);
    const double x = 0.5 * beta * (energy - mu);
    const double sech = 1.0 / std::cosh(x);
    return 0.25 * beta * sech * sech;
}

// Estimate the per-row footprint of the temporary
// [left/right] x [3 axes] x [samples] x [moments] complex workspaces.
inline size_t fftcheb_bytes_per_row(int n_moments, int sample_block_size) {
    return static_cast<size_t>(2) * 3 * static_cast<size_t>(sample_block_size) *
           static_cast<size_t>(n_moments) * sizeof(cd);
}

// Estimate how many row chunks are needed to keep the temporary
// [left/right] x [3 axes] x [samples] x [rows] x [moments] workspaces bounded.
inline int infer_partition_sections(
    int local_rows,
    int n_moments,
    int sample_block_size,
    int explicit_sections,
    size_t max_workspace_mb
) {
    if (local_rows <= 0 || n_moments <= 0) {
        return 1;
    }

    if (explicit_sections > 0) {
        return std::max(1, explicit_sections);
    }

    if (max_workspace_mb == 0) {
        return 1;
    }

    const size_t bytes_per_row = fftcheb_bytes_per_row(n_moments, sample_block_size);
    const size_t max_bytes = max_workspace_mb * 1024ULL * 1024ULL;
    const size_t rows_per_chunk = std::max<size_t>(1, max_bytes / std::max<size_t>(1, bytes_per_row));
    return std::max(1, static_cast<int>((static_cast<size_t>(local_rows) + rows_per_chunk - 1) / rows_per_chunk));
}

template <typename VectorType>
struct is_dist_multivector : std::false_type {};

template <typename T>
struct is_dist_multivector<DistMultiVector<T>> : std::true_type {};

template <typename VectorType>
inline constexpr bool is_dist_multivector_v = is_dist_multivector<std::remove_cv_t<VectorType>>::value;

template <typename VectorType, typename T>
VectorType make_transport_vector(DistGraph* graph, int sample_block_size) {
    if constexpr (is_dist_multivector_v<VectorType>) {
        return VectorType(graph, sample_block_size);
    } else {
        (void)sample_block_size;
        return VectorType(graph);
    }
}

template <typename VectorType>
int vector_block_width(const VectorType& vec) {
    if constexpr (is_dist_multivector_v<VectorType>) {
        return vec.num_vectors;
    } else {
        return 1;
    }
}

inline size_t fftcheb_store_offset(
    int row,
    int axis,
    int sample_idx,
    int moment_idx,
    int sample_block_size,
    int n_moments
) {
    return (((static_cast<size_t>(row) * 3 + static_cast<size_t>(axis)) *
             static_cast<size_t>(sample_block_size) + static_cast<size_t>(sample_idx)) *
            static_cast<size_t>(n_moments) + static_cast<size_t>(moment_idx));
}

inline const cd* fftcheb_row_axis_ptr(
    const std::vector<cd>& store,
    int row,
    int axis,
    int sample_block_size,
    int n_moments
) {
    return store.data() + fftcheb_store_offset(row, axis, 0, 0, sample_block_size, n_moments);
}

template <typename T>
DistMultiVector<T> extract_sample_block(
    const DistMultiVector<T>& samples,
    int sample_begin,
    int sample_count
) {
    DistMultiVector<T> block(samples.graph, sample_count);
    block.set_constant(T(0));
    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        for (int row = 0; row < samples.local_rows; ++row) {
            block(row, sample_idx) = samples(row, sample_begin + sample_idx);
        }
    }
    return block;
}

inline int choose_sample_block_size(
    int n_samples,
    int local_rows,
    int n_moments,
    int partition_sections,
    size_t max_workspace_mb
) {
    if (n_samples <= 1 || max_workspace_mb == 0) {
        return std::max(1, n_samples);
    }

    const size_t max_bytes = max_workspace_mb * 1024ULL * 1024ULL;
    if (partition_sections > 0) {
        const int section_rows = std::max(1, (local_rows + partition_sections - 1) / partition_sections);
        for (int block_size = n_samples; block_size >= 1; --block_size) {
            if (static_cast<size_t>(section_rows) * fftcheb_bytes_per_row(n_moments, block_size) <= max_bytes) {
                return block_size;
            }
        }
        return 1;
    }

    int best_block_size = 1;
    size_t best_cost = static_cast<size_t>(n_samples) *
                       static_cast<size_t>(infer_partition_sections(local_rows, n_moments, 1, 0, max_workspace_mb));

    for (int block_size = 2; block_size <= n_samples; ++block_size) {
        const int sections = infer_partition_sections(local_rows, n_moments, block_size, 0, max_workspace_mb);
        const size_t sample_passes =
            (static_cast<size_t>(n_samples) + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
        const size_t cost = sample_passes * static_cast<size_t>(sections);
        if (cost < best_cost || (cost == best_cost && block_size > best_block_size)) {
            best_block_size = block_size;
            best_cost = cost;
        }
    }

    return best_block_size;
}

template <typename T>
inline void set_multivector_cols(DistMultiVector<T>& mv, const std::array<DistVector<T>, 3>& vecs) {
    for (int axis = 0; axis < 3; ++axis) {
        mv.set_col(axis, vecs[axis]);
    }
}

// Copy one Chebyshev moment of one axis into the shared
// [row][axis][sample][moment] scratch layout used by the row FFT stage.
template <typename VectorType>
inline void store_axis_rows(
    const VectorType& vec,
    std::vector<cd>& store,
    int row_begin,
    int row_count,
    int axis,
    int moment_idx,
    int sample_block_size,
    int n_moments
) {
    if constexpr (is_dist_multivector_v<VectorType>) {
        for (int row = 0; row < row_count; ++row) {
            for (int sample_idx = 0; sample_idx < sample_block_size; ++sample_idx) {
                store[fftcheb_store_offset(row, axis, sample_idx, moment_idx, sample_block_size, n_moments)] =
                    static_cast<cd>(vec(row_begin + row, sample_idx));
            }
        }
    } else {
        const auto* data = vec.local_data();
        for (int row = 0; row < row_count; ++row) {
            store[fftcheb_store_offset(row, axis, 0, moment_idx, sample_block_size, n_moments)] =
                static_cast<cd>(data[row_begin + row]);
        }
    }
}

/*
 * Build the contribution of one stochastic sample block.
 *
 * For each partition section we regenerate the left/right Chebyshev sequences,
 * cache only the owned rows belonging to that section, and immediately turn
 * those rows into energy-space vectors. The per-row energy-space inner
 * products are accumulated into `kernel_accum`, so no explicit M x M moment
 * tensor is ever stored.
 */
template <typename T, typename VectorType>
void accumulate_fftcheb_block(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& Vx,
    BlockSpMat<T>& Vy,
    BlockSpMat<T>& Vz,
    const VectorType& samples,
    int n_moments,
    DCFormulation formulation,
    int resolved_sections,
    KpmKernel kernel,
    std::vector<cd>& kernel_accum
) {
    DistGraph* graph = samples.graph;
    const int rank = graph->rank;
    const int local_rows = graph->block_offsets[graph->owned_global_indices.size()];
    const int sample_block_size = vector_block_width(samples);
    const int n_sections = std::max(1, resolved_sections);
    // Summation weights: Jackson for a pointwise-clean sigma(eps) spectrum;
    // CosTaper for integrated sigma(mu, T) where the weak-form bias decay is
    // higher order (see kpm.hpp / benchmarks/fermi_dirac/METHOD.md).
    const std::vector<double> jackson = kpm_kernel_weights(kernel, n_moments);

    std::vector<double> phi3_weight(n_moments, 0.0);
    std::vector<double> eps_grid(n_moments, 0.0);
    std::vector<double> sqrt_one_minus_eps2(n_moments, 0.0);
    std::vector<cd> phase_weight(n_moments, cd(0.0, 0.0));
    std::vector<cd> phase_weight_m(n_moments, cd(0.0, 0.0));

    // We follow the supplement notation:
    // - phi3 uses the cosine/Chebyshev transform from Eq. S6.
    // - phi1 and phi2 use the complex phase factors appearing in Eq. S6,
    //   with phi2 carrying the extra factor of m.
    for (int m = 0; m < n_moments; ++m) {
        const double coeff = 2.0 * jackson[m] / (M_PI * (1.0 + (m == 0 ? 1.0 : 0.0)));
        phi3_weight[m] = jackson[m] / M_PI;

        const double theta_phase = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(n_moments));
        const cd cm = std::exp(cd(0.0, theta_phase));
        phase_weight[m] = coeff * cm;
        phase_weight_m[m] = phase_weight[m] * static_cast<double>(m);
    }

    for (int k = 0; k < n_moments; ++k) {
        const double theta = M_PI * (static_cast<double>(k) + 0.5) / static_cast<double>(n_moments);
        eps_grid[k] = std::cos(theta);
        sqrt_one_minus_eps2[k] = std::sin(theta);
    }

    const double velocity_scale = 1.0 / (2.0 * cheby.half_width);

    // Right sequence: v_b T_n(H)|r>
    // Left sequence:  <r|T_n(H) v_a  encoded via adjoint propagation and
    // `current_adjoint(...)` to respect the existing non-orthogonal setup.
    std::array<VectorType, 3> right_init = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> right_prev = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> right_curr = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> right_next = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> left_axes = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    current(cheby, Vx, Vy, Vz, samples, right_init.data(), velocity_scale);

    VectorType left_base_prev = make_transport_vector<VectorType, T>(graph, sample_block_size);
    VectorType left_base_curr = make_transport_vector<VectorType, T>(graph, sample_block_size);
    VectorType left_base_next = make_transport_vector<VectorType, T>(graph, sample_block_size);
    
    const int section_rows = std::max(1, (local_rows + n_sections - 1) / n_sections);

    for (int section = 0; section < n_sections; ++section) {
        const int row_begin = section * section_rows;
        const int row_end = std::min(local_rows, row_begin + section_rows);
        const int row_count = row_end - row_begin;
        if (row_count <= 0) {
            continue;
        }

        if (rank == 0 && n_sections > 1 &&
            ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress)) {
            std::cout << "fftcheb section " << (section + 1) << " / " << n_sections << std::endl;
        }

        // we can allocate it once and reuse.
        std::vector<cd> left_store(
            static_cast<size_t>(row_count) * 3 * static_cast<size_t>(sample_block_size) *
                static_cast<size_t>(n_moments),
            cd(0.0, 0.0)
        );
        std::vector<cd> right_store(
            static_cast<size_t>(row_count) * 3 * static_cast<size_t>(sample_block_size) *
                static_cast<size_t>(n_moments),
            cd(0.0, 0.0)
        );

        for (int axis = 0; axis < 3; ++axis) {
            right_prev[axis].copy_from(right_init[axis]);
        }
        left_base_prev.copy_from(samples);
        current_adjoint(cheby, Vx, Vy, Vz, left_base_prev, left_axes.data(), velocity_scale);
        for (int axis = 0; axis < 3; ++axis) {
            store_axis_rows(left_axes[axis], left_store, row_begin, row_count, axis, 0, sample_block_size, n_moments);
            store_axis_rows(right_prev[axis], right_store, row_begin, row_count, axis, 0, sample_block_size, n_moments);
        }

        if (n_moments > 1) {
            for (int axis = 0; axis < 3; ++axis) {
                cheby.apply_op(right_prev[axis], right_curr[axis], ChebyMode::NORMAL);
            }
            cheby.apply_op(left_base_prev, left_base_curr, ChebyMode::ADJOINT);
            current_adjoint(cheby, Vx, Vy, Vz, left_base_curr, left_axes.data(), velocity_scale);

            for (int axis = 0; axis < 3; ++axis) {
                store_axis_rows(left_axes[axis], left_store, row_begin, row_count, axis, 1, sample_block_size, n_moments);
                store_axis_rows(right_curr[axis], right_store, row_begin, row_count, axis, 1, sample_block_size, n_moments);
            }
        }

        for (int moment = 2; moment < n_moments; ++moment) {
            // Poll every 16 moments: one pass is a handful of matvecs, and a
            // production n_moments runs to the tens of thousands.
            if ((moment & 15) == 0) ::rsatb::util::throw_if_interrupted(graph->comm);
            for (int axis = 0; axis < 3; ++axis) {
                cheby.apply_op(right_curr[axis], right_next[axis], ChebyMode::NORMAL);
                right_next[axis].axpby(static_cast<T>(-1.0), right_prev[axis], static_cast<T>(2.0));
            }

            cheby.apply_op(left_base_curr, left_base_next, ChebyMode::ADJOINT);
            left_base_next.axpby(static_cast<T>(-1.0), left_base_prev, static_cast<T>(2.0));
            current_adjoint(cheby, Vx, Vy, Vz, left_base_next, left_axes.data(), velocity_scale);

            for (int axis = 0; axis < 3; ++axis) {
                store_axis_rows(left_axes[axis], left_store, row_begin, row_count, axis, moment, sample_block_size, n_moments);
                store_axis_rows(right_next[axis], right_store, row_begin, row_count, axis, moment, sample_block_size, n_moments);

                right_prev[axis].swap(right_curr[axis]);
                right_curr[axis].swap(right_next[axis]);
            }
            left_base_prev.swap(left_base_curr);
            left_base_curr.swap(left_base_next);
        }

        RowFFTWorkspace workspace(n_moments, sample_block_size);
        std::vector<cd> weighted(static_cast<size_t>(sample_block_size) * static_cast<size_t>(n_moments), cd(0.0, 0.0));

        std::array<std::vector<cd>, 3> left_phi1, left_phi2, left_phi3;
        std::array<std::vector<cd>, 3> right_phi1, right_phi2, right_phi3;
        for (int axis = 0; axis < 3; ++axis) {
            const size_t axis_size = static_cast<size_t>(sample_block_size) * static_cast<size_t>(n_moments);
            left_phi1[axis].assign(axis_size, cd(0.0, 0.0));
            left_phi2[axis].assign(axis_size, cd(0.0, 0.0));
            left_phi3[axis].assign(axis_size, cd(0.0, 0.0));
            right_phi1[axis].assign(axis_size, cd(0.0, 0.0));
            right_phi2[axis].assign(axis_size, cd(0.0, 0.0));
            right_phi3[axis].assign(axis_size, cd(0.0, 0.0));
        }

        auto build_phi3 = [&](const cd* seq, std::vector<cd>& out) {
            for (int sample_idx = 0; sample_idx < sample_block_size; ++sample_idx) {
                const size_t base = static_cast<size_t>(sample_idx) * static_cast<size_t>(n_moments);
                for (int m = 0; m < n_moments; ++m) {
                    weighted[base + static_cast<size_t>(m)] = phi3_weight[m] * seq[base + static_cast<size_t>(m)];
                }
            }
            workspace.dct_type3_batch(weighted.data(), out.data());
        };

        auto build_phi_complex = [&](const cd* seq, const std::vector<cd>& weights, std::vector<cd>& out) {
            for (int sample_idx = 0; sample_idx < sample_block_size; ++sample_idx) {
                const size_t base = static_cast<size_t>(sample_idx) * static_cast<size_t>(n_moments);
                for (int m = 0; m < n_moments; ++m) {
                    weighted[base + static_cast<size_t>(m)] = weights[m] * seq[base + static_cast<size_t>(m)];
                }
            }
            workspace.zero_padded_ifft_batch(weighted.data(), n_moments, out.data(), n_moments);
        };

        for (int row = 0; row < row_count; ++row) {
            for (int axis = 0; axis < 3; ++axis) {
                const cd* left_row = fftcheb_row_axis_ptr(left_store, row, axis, sample_block_size, n_moments);
                const cd* right_row = fftcheb_row_axis_ptr(right_store, row, axis, sample_block_size, n_moments);

                build_phi3(left_row, left_phi3[axis]);
                build_phi3(right_row, right_phi3[axis]);

                if (formulation == DCFormulation::Bastin) {
                    // The Bastin path needs the 6 energy-space vectors from
                    // Eqs. S5-S6: phi1/phi2/phi3 for both left and right.
                    build_phi_complex(left_row, phase_weight, left_phi1[axis]);
                    build_phi_complex(left_row, phase_weight_m, left_phi2[axis]);
                    build_phi_complex(right_row, phase_weight, right_phi1[axis]);
                    build_phi_complex(right_row, phase_weight_m, right_phi2[axis]);
                }
            }

            if (formulation == DCFormulation::Greenwood) {
                // Greenwood is the cheaper special case: one cosine transform
                // per side, then divide by (1 - eps^2) on the Chebyshev grid.
                for (int axis = 0; axis < 3; ++axis) {
                    const int comp = conductivity_component_index(axis, axis);
                    for (int k = 0; k < n_moments; ++k) {
                        const double denom = std::max(1e-14, 1.0 - eps_grid[k] * eps_grid[k]);
                        cd accum = cd(0.0, 0.0);
                        for (int sample_idx = 0; sample_idx < sample_block_size; ++sample_idx) {
                            const size_t idx = static_cast<size_t>(sample_idx) * static_cast<size_t>(n_moments) +
                                               static_cast<size_t>(k);
                            accum += std::conj(left_phi3[axis][idx]) * right_phi3[axis][idx];
                        }
                        kernel_accum[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] += accum / denom;
                    }
                }
                continue;
            }

            for (int a = 0; a < 3; ++a) {
                for (int b = a; b < 3; ++b) {
                    const int comp = conductivity_component_index(a, b);
                    for (int k = 0; k < n_moments; ++k) {
                        // Supplement Eq. S5 written in rowwise accumulated form,
                        // with sample averaging done by summing only same-sample
                        // pairings inside the current sample block.
                        cd value = cd(0.0, 0.0);
                        for (int sample_idx = 0; sample_idx < sample_block_size; ++sample_idx) {
                            const size_t idx = static_cast<size_t>(sample_idx) * static_cast<size_t>(n_moments) +
                                               static_cast<size_t>(k);
                            value +=
                                eps_grid[k] * (std::conj(left_phi3[a][idx]) * right_phi1[b][idx] +
                                               std::conj(left_phi1[a][idx]) * right_phi3[b][idx]) +
                                cd(0.0, sqrt_one_minus_eps2[k]) *
                                    (std::conj(left_phi2[a][idx]) * right_phi3[b][idx] -
                                     std::conj(left_phi3[a][idx]) * right_phi2[b][idx]);
                        }

                        kernel_accum[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] += value;
                    }
                }
            }
        }
    }
}

/*
 * Compute the full spectral transport response for all stochastic samples
 * assigned to the current k-point.
 *
 * Output layout:
 * - `energy_out[k]` is the physical energy corresponding to Chebyshev node k,
 *   sorted in ascending order.
 * - `sigma_out[k * 6 + comp]` stores one tensor slice in the order
 *   (xx, xy, xz, yy, yz, zz).
 *
 * Greenwood returns the zero-temperature conductivity spectrum directly.
 * Bastin first builds the energy-resolved zeta kernel and then integrates it
 * over the dimensionless energy variable, so the returned quantity is the
 * zero-temperature conductivity as a function of Fermi energy.
 */
template <typename T>
void dc_spectrum_fftcheb(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& Vx,
    BlockSpMat<T>& Vy,
    BlockSpMat<T>& Vz,
    DistMultiVector<T>& psi_samples, // we should not assert the input is a multivector
    int n_moments,
    DCFormulation formulation,
    double prefactor,
    int partition_sections,
    size_t max_workspace_mb,
    std::vector<double>& sigma_out,
    std::vector<double>& energy_out,
    KpmKernel kernel = KpmKernel::Jackson
) {
    if (n_moments <= 0) {
        throw std::runtime_error("n_moments must be positive");
    }

    DistGraph* graph = psi_samples.graph;
    const int n_samples = psi_samples.num_vectors;
    if (n_samples <= 0) {
        throw std::runtime_error("At least one stochastic sample is required");
    }

    // Local owned scalar rows set the memory pressure for the row buffers.
    const int local_rows = psi_samples.local_rows;
    const int sample_block_size = choose_sample_block_size(
        n_samples, local_rows, n_moments, partition_sections, max_workspace_mb
    );

    std::vector<cd> local_kernel(static_cast<size_t>(n_moments) * 6, cd(0.0, 0.0));
    for (int sample_begin = 0; sample_begin < n_samples; sample_begin += sample_block_size) {
        const int block_width = std::min(sample_block_size, n_samples - sample_begin);
        const int resolved_sections = infer_partition_sections(
            local_rows, n_moments, block_width, partition_sections, max_workspace_mb
        );

        if (block_width == 1) {
            DistVector<T> sample = psi_samples.get_col(sample_begin);
            accumulate_fftcheb_block(
                cheby, Vx, Vy, Vz, sample, n_moments, formulation, resolved_sections, kernel, local_kernel
            ); // there should be another switch when block_width == n_samples so we do not need to extract.
        } else {
            DistMultiVector<T> sample_block = extract_sample_block(psi_samples, sample_begin, block_width);
            accumulate_fftcheb_block(
                cheby, Vx, Vy, Vz, sample_block, n_moments, formulation, resolved_sections, kernel, local_kernel
            );
        }
    }

    for (cd& value : local_kernel) {
        value /= static_cast<double>(n_samples);
    }

    std::vector<cd> global_kernel(local_kernel.size(), cd(0.0, 0.0));
    if (graph->size == 1) {
        global_kernel = local_kernel;
    } else {
        MPI_Allreduce(
            local_kernel.data(),
            global_kernel.data(),
            static_cast<int>(local_kernel.size()),
            MPI_CXX_DOUBLE_COMPLEX,
            MPI_SUM,
            graph->comm
        );
    }

    std::vector<double> eps_sorted(n_moments, 0.0);
    energy_out.resize(n_moments);
    sigma_out.assign(static_cast<size_t>(n_moments) * 6, 0.0);

    for (int k = 0; k < n_moments; ++k) {
        const int src = n_moments - 1 - k;
        const double theta = M_PI * (static_cast<double>(src) + 0.5) / static_cast<double>(n_moments);
        const double eps = std::cos(theta);
        eps_sorted[k] = eps;
        energy_out[k] = cheby.center + cheby.half_width * eps;
    }

    if (formulation == DCFormulation::Greenwood) {
        for (int k = 0; k < n_moments; ++k) {
            const int src = n_moments - 1 - k;
            for (int comp = 0; comp < 6; ++comp) {
                sigma_out[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] =
                    prefactor * global_kernel[static_cast<size_t>(src) * 6 + static_cast<size_t>(comp)].real();
            }
        }
        return;
    }

    // Bastin returns sigma(mu, T=0), so after building zeta(epsilon) we perform
    // a cumulative integration over epsilon on the ascending Chebyshev grid.
    std::array<double, 6> running = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 6> prev_integrand = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int k = 0; k < n_moments; ++k) {
        const int src = n_moments - 1 - k;
        std::array<double, 6> integrand = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int comp = 0; comp < 6; ++comp) {
            integrand[comp] = (cd(0.0, 1.0) * global_kernel[static_cast<size_t>(src) * 6 + static_cast<size_t>(comp)]).real();
        }

        if (k > 0) {
            const double depsilon = eps_sorted[k] - eps_sorted[k - 1];
            for (int comp = 0; comp < 6; ++comp) {
                running[comp] += 0.5 * (prev_integrand[comp] + integrand[comp]) * depsilon;
            }
        }

        for (int comp = 0; comp < 6; ++comp) {
            sigma_out[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] = prefactor * running[comp];
            prev_integrand[comp] = integrand[comp];
        }
    }
}

/*
 * Quasi-eigenstate (windowed) evaluation of the Greenwood spectrum.
 *
 * Same bilinear form as the fftcheb Greenwood path -- sigma(eps_k) is
 * Sum_{m,m'} w_m(eps_k) w_m'(eps_k) <L_m|R_m'> with identical kernel
 * weights, so the two routes agree to rounding -- but with the TBPM memory
 * layout of arXiv:2510.17432 (their EC-Concurrent-M): the right Chebyshev
 * sequences accumulate directly into energy-filtered vectors
 * ("quasi-eigenstates") at the requested nodes, and the left chain
 * contracts against them with streaming local dots. No row store, no row
 * FFTs, no partition re-passes: memory is 3 * NE * rows against fftcheb's
 * 6 * n_moments * rows, where NE is the size of the node window.
 *
 * NE is the control knob. For sigma(mu, T) the thermal window is a few
 * dozen nodes, collapsing the workspace from GBs (or section re-passes) to
 * MBs in a single pass. For the FULL grid the accumulation is
 * O(n_moments * NE * rows) where fftcheb's row DCT is
 * O(n_moments * log * rows): use fftcheb there. Greenwood diagonal only,
 * like the fftcheb Greenwood path; Bastin needs the row transform.
 */
template <typename T, typename VectorType>
void accumulate_qe_block(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& Vx,
    BlockSpMat<T>& Vy,
    BlockSpMat<T>& Vz,
    const VectorType& samples,
    int n_moments,
    const std::vector<int>& src_nodes,
    KpmKernel kernel,
    std::vector<cd>& kernel_accum
) {
    DistGraph* graph = samples.graph;
    const int rank = graph->rank;
    const int local_rows = graph->block_offsets[graph->owned_global_indices.size()];
    const int sample_block_size = vector_block_width(samples);
    const int n_nodes = static_cast<int>(src_nodes.size());
    const size_t entries = static_cast<size_t>(local_rows) * static_cast<size_t>(sample_block_size);

    const std::vector<double> jackson = kpm_kernel_weights(kernel, n_moments);

    // Per-node theta and the cos(m theta) recurrence state.
    std::vector<double> cos_theta(n_nodes), denom(n_nodes);
    std::vector<double> c_prev(n_nodes), c_curr(n_nodes);
    for (int j = 0; j < n_nodes; ++j) {
        const double theta = M_PI * (static_cast<double>(src_nodes[j]) + 0.5) /
                             static_cast<double>(n_moments);
        cos_theta[j] = std::cos(theta);
        const double eps = cos_theta[j];
        denom[j] = std::max(1e-14, 1.0 - eps * eps);
    }

    auto value_at = [&](const VectorType& vec, int row, int sample_idx) -> cd {
        if constexpr (is_dist_multivector_v<VectorType>) {
            return static_cast<cd>(vec(row, sample_idx));
        } else {
            (void)sample_idx;
            return static_cast<cd>(vec.local_data()[row]);
        }
    };

    const double velocity_scale = 1.0 / (2.0 * cheby.half_width);

    // Quasi-eigenstate store: [axis][node][row * samples].
    std::vector<std::vector<cd>> q(static_cast<size_t>(3) * n_nodes);
    for (auto& slot : q) slot.assign(entries, cd(0.0, 0.0));
    auto q_slot = [&](int axis, int j) -> std::vector<cd>& {
        return q[static_cast<size_t>(axis) * n_nodes + static_cast<size_t>(j)];
    };

    // Right pass: R_m = T_m(Op) J_b psi accumulates into the node vectors.
    std::array<VectorType, 3> r_prev = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> r_curr = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    std::array<VectorType, 3> r_next = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };
    current(cheby, Vx, Vy, Vz, samples, r_prev.data(), velocity_scale);

    auto accumulate_right = [&](int m, const std::array<VectorType, 3>& r) {
        for (int j = 0; j < n_nodes; ++j) {
            double t_m;
            if (m == 0) {
                c_prev[j] = 1.0;
                t_m = 1.0;
            } else if (m == 1) {
                c_curr[j] = cos_theta[j];
                t_m = cos_theta[j];
            } else {
                const double t_next = 2.0 * cos_theta[j] * c_curr[j] - c_prev[j];
                c_prev[j] = c_curr[j];
                c_curr[j] = t_next;
                t_m = t_next;
            }
            const double w = (m == 0 ? 1.0 : 2.0) * jackson[m] / M_PI * t_m;
            for (int axis = 0; axis < 3; ++axis) {
                std::vector<cd>& slot = q_slot(axis, j);
                size_t idx = 0;
                for (int row = 0; row < local_rows; ++row) {
                    for (int s = 0; s < sample_block_size; ++s, ++idx) {
                        slot[idx] += w * value_at(r[axis], row, s);
                    }
                }
            }
        }
    };

    accumulate_right(0, r_prev);
    if (n_moments > 1) {
        for (int axis = 0; axis < 3; ++axis) {
            cheby.apply_op(r_prev[axis], r_curr[axis], ChebyMode::NORMAL);
        }
        accumulate_right(1, r_curr);
    }
    for (int m = 2; m < n_moments; ++m) {
        if ((m & 15) == 0) ::rsatb::util::throw_if_interrupted(graph->comm);
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
            ::rsatb::util::every_nth(m - 2, std::max(1, n_moments / 10))) {
            std::cout << "dc: qe right pass, moment " << m << " / " << n_moments << std::endl;
        }
        for (int axis = 0; axis < 3; ++axis) {
            cheby.apply_op(r_curr[axis], r_next[axis], ChebyMode::NORMAL);
            r_next[axis].axpby(static_cast<T>(-1.0), r_prev[axis], static_cast<T>(2.0));
            r_prev[axis].swap(r_curr[axis]);
            r_curr[axis].swap(r_next[axis]);
        }
        accumulate_right(m, r_curr);
    }

    // Left pass: L_m = J_a^adj T_m(Op_adj) psi contracts against the stored
    // node vectors with local partial dots; one MPI reduction happens later
    // over the whole kernel_accum.
    for (int j = 0; j < n_nodes; ++j) { c_prev[j] = 1.0; c_curr[j] = cos_theta[j]; }

    VectorType l_prev = make_transport_vector<VectorType, T>(graph, sample_block_size);
    VectorType l_curr = make_transport_vector<VectorType, T>(graph, sample_block_size);
    VectorType l_next = make_transport_vector<VectorType, T>(graph, sample_block_size);
    std::array<VectorType, 3> l_axes = {
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size),
        make_transport_vector<VectorType, T>(graph, sample_block_size)
    };

    auto accumulate_left = [&](int m) {
        current_adjoint(cheby, Vx, Vy, Vz, m == 0 ? l_prev : (m == 1 ? l_curr : l_next),
                        l_axes.data(), velocity_scale);
        for (int j = 0; j < n_nodes; ++j) {
            double t_m;
            if (m == 0) {
                t_m = 1.0;
            } else if (m == 1) {
                t_m = cos_theta[j];
            } else {
                const double t_next = 2.0 * cos_theta[j] * c_curr[j] - c_prev[j];
                t_m = t_next;
            }
            const double w = (m == 0 ? 1.0 : 2.0) * jackson[m] / M_PI * t_m;
            for (int axis = 0; axis < 3; ++axis) {
                const int comp = conductivity_component_index(axis, axis);
                const std::vector<cd>& slot = q_slot(axis, j);
                cd partial(0.0, 0.0);
                size_t idx = 0;
                for (int row = 0; row < local_rows; ++row) {
                    for (int s = 0; s < sample_block_size; ++s, ++idx) {
                        partial += std::conj(value_at(l_axes[axis], row, s)) * slot[idx];
                    }
                }
                kernel_accum[static_cast<size_t>(src_nodes[j]) * 6 + static_cast<size_t>(comp)] +=
                    w * partial / denom[j];
            }
        }
        // Advance the shared cos recurrence once per moment (not per node
        // read): only after every node used t_m at this m.
        if (m >= 2) {
            for (int j = 0; j < n_nodes; ++j) {
                const double t_next = 2.0 * cos_theta[j] * c_curr[j] - c_prev[j];
                c_prev[j] = c_curr[j];
                c_curr[j] = t_next;
            }
        }
    };

    l_prev.copy_from(samples);
    accumulate_left(0);
    if (n_moments > 1) {
        cheby.apply_op(l_prev, l_curr, ChebyMode::ADJOINT);
        accumulate_left(1);
    }
    for (int m = 2; m < n_moments; ++m) {
        if ((m & 15) == 0) ::rsatb::util::throw_if_interrupted(graph->comm);
        if (rank == 0 && ::rsatb::util::Log::enabled(::rsatb::util::LogLevel::kProgress) &&
            ::rsatb::util::every_nth(m - 2, std::max(1, n_moments / 10))) {
            std::cout << "dc: qe left pass, moment " << m << " / " << n_moments << std::endl;
        }
        cheby.apply_op(l_curr, l_next, ChebyMode::ADJOINT);
        l_next.axpby(static_cast<T>(-1.0), l_prev, static_cast<T>(2.0));
        accumulate_left(m);
        l_prev.swap(l_curr);
        l_curr.swap(l_next);
    }
}

// Wrapper matching dc_spectrum_fftcheb's contract: full-length ascending
// sigma/energy arrays, with sigma zero outside the requested node window.
// `nodes_ascending` indexes the ascending output grid; empty = all nodes.
template <typename T>
void dc_spectrum_qe(
    Chebyshev<T>& cheby,
    BlockSpMat<T>& Vx,
    BlockSpMat<T>& Vy,
    BlockSpMat<T>& Vz,
    DistMultiVector<T>& psi_samples,
    int n_moments,
    double prefactor,
    std::vector<double>& sigma_out,
    std::vector<double>& energy_out,
    KpmKernel kernel = KpmKernel::Jackson,
    std::vector<int> nodes_ascending = {}
) {
    if (n_moments <= 0) {
        throw std::runtime_error("n_moments must be positive");
    }
    DistGraph* graph = psi_samples.graph;
    const int n_samples = psi_samples.num_vectors;
    if (n_samples <= 0) {
        throw std::runtime_error("At least one stochastic sample is required");
    }

    if (nodes_ascending.empty()) {
        nodes_ascending.resize(n_moments);
        for (int k = 0; k < n_moments; ++k) nodes_ascending[k] = k;
    }
    std::sort(nodes_ascending.begin(), nodes_ascending.end());
    nodes_ascending.erase(std::unique(nodes_ascending.begin(), nodes_ascending.end()),
                          nodes_ascending.end());
    if (nodes_ascending.front() < 0 || nodes_ascending.back() >= n_moments) {
        throw std::runtime_error("dc_spectrum_qe: node index out of range");
    }
    std::vector<int> src_nodes(nodes_ascending.size());
    for (size_t j = 0; j < nodes_ascending.size(); ++j) {
        src_nodes[j] = n_moments - 1 - nodes_ascending[j];
    }

    std::vector<cd> local_kernel(static_cast<size_t>(n_moments) * 6, cd(0.0, 0.0));
    if (n_samples == 1) {
        DistVector<T> sample = psi_samples.get_col(0);
        accumulate_qe_block(cheby, Vx, Vy, Vz, sample, n_moments, src_nodes, kernel, local_kernel);
    } else {
        accumulate_qe_block(cheby, Vx, Vy, Vz, psi_samples, n_moments, src_nodes, kernel, local_kernel);
    }

    for (cd& value : local_kernel) {
        value /= static_cast<double>(n_samples);
    }

    std::vector<cd> global_kernel(local_kernel.size(), cd(0.0, 0.0));
    if (graph->size == 1) {
        global_kernel = local_kernel;
    } else {
        MPI_Allreduce(local_kernel.data(), global_kernel.data(),
                      static_cast<int>(local_kernel.size()), MPI_CXX_DOUBLE_COMPLEX,
                      MPI_SUM, graph->comm);
    }

    energy_out.resize(n_moments);
    sigma_out.assign(static_cast<size_t>(n_moments) * 6, 0.0);
    for (int k = 0; k < n_moments; ++k) {
        const int src = n_moments - 1 - k;
        const double theta = M_PI * (static_cast<double>(src) + 0.5) / static_cast<double>(n_moments);
        energy_out[k] = cheby.center + cheby.half_width * std::cos(theta);
    }
    for (int k : nodes_ascending) {
        const int src = n_moments - 1 - k;
        for (int comp = 0; comp < 6; ++comp) {
            sigma_out[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] =
                prefactor * global_kernel[static_cast<size_t>(src) * 6 + static_cast<size_t>(comp)].real();
        }
    }
}

inline double interpolate_spectrum_component(
    const std::vector<double>& energy,
    const std::vector<double>& sigma,
    int n_comp,
    int comp_idx,
    double mu
) {
    if (energy.empty()) {
        return 0.0;
    }
    if (mu <= energy.front()) {
        return sigma[comp_idx];
    }
    if (mu >= energy.back()) {
        return sigma[(energy.size() - 1) * static_cast<size_t>(n_comp) + static_cast<size_t>(comp_idx)];
    }

    auto upper = std::lower_bound(energy.begin(), energy.end(), mu);
    const size_t hi = static_cast<size_t>(upper - energy.begin());
    const size_t lo = hi - 1;

    const double x0 = energy[lo];
    const double x1 = energy[hi];
    const double y0 = sigma[lo * static_cast<size_t>(n_comp) + static_cast<size_t>(comp_idx)];
    const double y1 = sigma[hi * static_cast<size_t>(n_comp) + static_cast<size_t>(comp_idx)];
    const double t = (mu - x0) / (x1 - x0);
    return (1.0 - t) * y0 + t * y1;
}

// Integrate a precomputed zero-temperature spectrum against -df/dE to obtain
// sigma(mu, T). At T=0 this reduces to interpolation on the Chebyshev grid.
inline void integrate_dc_spectrum(
    const std::vector<double>& sigma_spectrum,
    const std::vector<double>& energy,
    double mu,
    double temp,
    std::vector<double>& sigma_out
) {
    const size_t n_energy = energy.size();
    sigma_out.assign(6, 0.0);
    if (n_energy == 0) {
        return;
    }

    if (temp <= 0.0) {
        for (int comp = 0; comp < 6; ++comp) {
            sigma_out[comp] = interpolate_spectrum_component(energy, sigma_spectrum, 6, comp, mu);
        }
        return;
    }

    for (size_t i = 1; i < n_energy; ++i) {
        const double e0 = energy[i - 1];
        const double e1 = energy[i];
        const double w0 = fermi_dirac_derivative_energy(e0, mu, temp);
        const double w1 = fermi_dirac_derivative_energy(e1, mu, temp);
        const double de = e1 - e0;
        for (int comp = 0; comp < 6; ++comp) {
            const double s0 = sigma_spectrum[(i - 1) * 6 + static_cast<size_t>(comp)];
            const double s1 = sigma_spectrum[i * 6 + static_cast<size_t>(comp)];
            sigma_out[comp] += 0.5 * (w0 * s0 + w1 * s1) * de;
        }
    }
}

} // namespace calc
} // namespace rsatb

#endif
