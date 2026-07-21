#ifndef RSATB_CALC_FFTCHEB_TRANSPORT_HPP
#define RSATB_CALC_FFTCHEB_TRANSPORT_HPP

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
    bool verbose,
    std::vector<cd>& kernel_accum
) {
    DistGraph* graph = samples.graph;
    const int rank = graph->rank;
    const int local_rows = graph->block_offsets[graph->owned_global_indices.size()];
    const int sample_block_size = vector_block_width(samples);
    const int n_sections = std::max(1, resolved_sections);
    const std::vector<double> jackson = jackson_kernel(n_moments);

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

        if (rank == 0 && verbose && n_sections > 1) {
            std::cout << "fftcheb section " << (section + 1) << " / " << n_sections << "\n";
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
    bool verbose = false
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
                cheby, Vx, Vy, Vz, sample, n_moments, formulation, resolved_sections, verbose, local_kernel
            ); // there should be another switch when block_width == n_samples so we do not need to extract.
        } else {
            DistMultiVector<T> sample_block = extract_sample_block(psi_samples, sample_begin, block_width);
            accumulate_fftcheb_block(
                cheby, Vx, Vy, Vz, sample_block, n_moments, formulation, resolved_sections, verbose, local_kernel
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
