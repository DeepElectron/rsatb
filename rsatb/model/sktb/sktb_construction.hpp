#ifndef RSATB_MODEL_SKTB_CONSTRUCTION_HPP
#define RSATB_MODEL_SKTB_CONSTRUCTION_HPP

#include "sk_table.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <array>
#include <complex>
#include <memory>
#include <vector>

using namespace vbcsr;
using namespace vbcsr::atomic;

namespace rsatb {
namespace model {
namespace sktb {

namespace detail {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using DBlock = std::vector<double>;

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline double norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

inline Vec3 normalized(const Vec3& value) {
    const double value_norm = norm(value);
    if (value_norm <= 1e-12) {
        return {0.0, 0.0, 0.0};
    }
    return {value[0] / value_norm, value[1] / value_norm, value[2] / value_norm};
}

inline Vec3 negated(const Vec3& value) {
    return {-value[0], -value[1], -value[2]};
}

inline Mat3 transpose(const Mat3& matrix) {
    Mat3 result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}

inline Mat3 matmul(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += left[i][k] * right[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

inline double frobenius_inner_product(const Mat3& left, const Mat3& right) {
    double result = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result += left[i][j] * right[i][j];
        }
    }
    return result;
}

inline const std::array<Mat3, 5>& canonical_d_matrices() {
    static const std::array<Mat3, 5> matrices = {{
        {{{0.0, 1.0 / std::sqrt(2.0), 0.0},
           {1.0 / std::sqrt(2.0), 0.0, 0.0},
           {0.0, 0.0, 0.0}}},
        {{{0.0, 0.0, 0.0},
           {0.0, 0.0, 1.0 / std::sqrt(2.0)},
           {0.0, 1.0 / std::sqrt(2.0), 0.0}}},
        {{{-1.0 / std::sqrt(6.0), 0.0, 0.0},
           {0.0, -1.0 / std::sqrt(6.0), 0.0},
           {0.0, 0.0, 2.0 / std::sqrt(6.0)}}},
        {{{0.0, 0.0, 1.0 / std::sqrt(2.0)},
           {0.0, 0.0, 0.0},
           {1.0 / std::sqrt(2.0), 0.0, 0.0}}},
        {{{1.0 / std::sqrt(2.0), 0.0, 0.0},
           {0.0, -1.0 / std::sqrt(2.0), 0.0},
           {0.0, 0.0, 0.0}}}
    }};
    return matrices;
}

inline Mat3 build_local_frame(const Vec3& direction) {
    const Vec3 ez = normalized(direction);
    Vec3 ex = {1.0, 0.0, 0.0};
    if (std::abs(ez[2]) < 1.0 - 1e-12) {
        ex = normalized(cross(Vec3{0.0, 0.0, 1.0}, ez));
    }
    const Vec3 ey = cross(ez, ex);

    Mat3 frame{};
    for (int i = 0; i < 3; ++i) {
        frame[i][0] = ex[i];
        frame[i][1] = ey[i];
        frame[i][2] = ez[i];
    }
    return frame;
}

inline int shell_dimension(int l) {
    if (l < 0 || l > 2) {
        throw std::runtime_error("SKTB shell rotation supports only s, p, and d orbitals.");
    }
    return 2 * l + 1;
}

inline int p_axis_for_m_index(int m_index) {
    static const std::array<int, 3> axes = {1, 2, 0}; // py, pz, px
    return axes[static_cast<size_t>(m_index)];
}

inline DBlock build_p_rotation(const Mat3& frame) {
    DBlock rotation(9, 0.0);
    for (int local = 0; local < 3; ++local) {
        const int local_axis = p_axis_for_m_index(local);
        for (int global = 0; global < 3; ++global) {
            const int global_axis = p_axis_for_m_index(global);
            rotation[static_cast<size_t>(local) * 3 + static_cast<size_t>(global)] =
                frame[global_axis][local_axis];
        }
    }
    return rotation;
}

inline DBlock build_d_rotation(const Mat3& frame) {
    DBlock rotation(25, 0.0);
    const auto frame_t = transpose(frame);
    const auto& canonical = canonical_d_matrices();
    for (int local = 0; local < 5; ++local) {
        const Mat3 rotated = matmul(frame, matmul(canonical[local], frame_t));
        for (int global = 0; global < 5; ++global) {
            rotation[static_cast<size_t>(local) * 5 + static_cast<size_t>(global)] =
                frobenius_inner_product(canonical[global], rotated);
        }
    }
    return rotation;
}

inline DBlock shell_rotation(int l, const Mat3& frame) {
    if (l == 0) {
        return {1.0};
    }
    if (l == 1) {
        return build_p_rotation(frame);
    }
    if (l == 2) {
        return build_d_rotation(frame);
    }
    throw std::runtime_error("SKTB shell rotation supports only s, p, and d orbitals.");
}

inline double shell_pair_direction_sign(int row_l, int col_l) {
    if (row_l > col_l && ((row_l + col_l) % 2 != 0)) {
        return -1.0;
    }
    return 1.0;
}

inline DBlock diagonal_block_row_major(const std::vector<double>& diagonal) {
    DBlock block(diagonal.size() * diagonal.size(), 0.0);
    for (size_t i = 0; i < diagonal.size(); ++i) {
        block[i * diagonal.size() + i] = diagonal[i];
    }
    return block;
}

inline bool has_nonzero_entries(const DBlock& block, double tol = 1e-14) {
    for (double value : block) {
        if (std::abs(value) > tol) {
            return true;
        }
    }
    return false;
}

/// Builds the bond-aligned contribution for one sigma/pi/delta channel.
///
/// Args:
///   row_l: Angular momentum of the row shell.
///   col_l: Angular momentum of the column shell.
///   channel: SK channel index, with 0/1/2 meaning sigma/pi/delta.
///   value: Radial channel value at the current distance.
/// Returns:
///   Row-major `(2 * row_l + 1) x (2 * col_l + 1)` aligned-z block.
inline DBlock build_aligned_shell_block(
    int row_l,
    int col_l,
    int channel,
    double value
) {
    const int nrow = shell_dimension(row_l);
    const int ncol = shell_dimension(col_l);
    DBlock block(static_cast<size_t>(nrow) * static_cast<size_t>(ncol), 0.0);
    const int m_abs = channel;
    const double signed_value = shell_pair_direction_sign(row_l, col_l) * value;

    if (m_abs == 0) {
        block[static_cast<size_t>(row_l) * static_cast<size_t>(ncol) + static_cast<size_t>(col_l)] = signed_value;
    } else {
        block[static_cast<size_t>(row_l - m_abs) * static_cast<size_t>(ncol) +
              static_cast<size_t>(col_l - m_abs)] = signed_value;
        block[static_cast<size_t>(row_l + m_abs) * static_cast<size_t>(ncol) +
              static_cast<size_t>(col_l + m_abs)] = signed_value;
    }

    return block;
}

inline DBlock rotate_shell_block(
    const DBlock& aligned_block,
    int row_l,
    int col_l,
    const DBlock& row_rotation,
    const DBlock& col_rotation
) {
    const int nrow = shell_dimension(row_l);
    const int ncol = shell_dimension(col_l);
    DBlock rotated(static_cast<size_t>(nrow) * static_cast<size_t>(ncol), 0.0);

    for (int global_row = 0; global_row < nrow; ++global_row) {
        for (int global_col = 0; global_col < ncol; ++global_col) {
            double value = 0.0;
            for (int local_row = 0; local_row < nrow; ++local_row) {
                for (int local_col = 0; local_col < ncol; ++local_col) {
                    value += row_rotation[static_cast<size_t>(local_row) * static_cast<size_t>(nrow) +
                                          static_cast<size_t>(global_row)] *
                             aligned_block[static_cast<size_t>(local_row) * static_cast<size_t>(ncol) +
                                           static_cast<size_t>(local_col)] *
                             col_rotation[static_cast<size_t>(local_col) * static_cast<size_t>(ncol) +
                                          static_cast<size_t>(global_col)];
                }
            }
            rotated[static_cast<size_t>(global_row) * static_cast<size_t>(ncol) +
                    static_cast<size_t>(global_col)] = value;
        }
    }

    return rotated;
}

inline void copy_shell_block_to_pair_block(
    const DBlock& shell_block,
    int row_l,
    int col_l,
    const SkAtomType& row_atom,
    const SkAtomType& col_atom,
    DBlock& pair_block
) {
    const size_t pair_ncol = col_atom.orbitals.size();
    const int shell_ncol = shell_dimension(col_l);

    for (size_t row = 0; row < row_atom.orbitals.size(); ++row) {
        const auto& row_orbital = row_atom.orbitals[row];
        if (row_orbital.l != row_l) {
            continue;
        }
        for (size_t col = 0; col < col_atom.orbitals.size(); ++col) {
            const auto& col_orbital = col_atom.orbitals[col];
            if (col_orbital.l != col_l) {
                continue;
            }
            pair_block[row * pair_ncol + col] +=
                shell_block[static_cast<size_t>(row_orbital.m_index) * static_cast<size_t>(shell_ncol) +
                            static_cast<size_t>(col_orbital.m_index)];
        }
    }
}

inline DBlock transpose_row_major(const DBlock& block, int nrow, int ncol) {
    DBlock result(static_cast<size_t>(nrow) * static_cast<size_t>(ncol), 0.0);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            result[static_cast<size_t>(j) * static_cast<size_t>(nrow) + static_cast<size_t>(i)] =
                block[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)];
        }
    }
    return result;
}

}  // namespace detail

class SkTableConstruction {
public:
    explicit SkTableConstruction(SkTable resolved)
        : resolved_(std::move(resolved)) {}

    std::unique_ptr<ImageContainer<double>> build_hamiltonian_images(AtomicData& atomic_data) const {
        return build_images(atomic_data, false);
    }

    std::unique_ptr<ImageContainer<double>> build_overlap_images(AtomicData& atomic_data) const {
        if (!resolved_.has_overlap) {
            throw std::runtime_error("SKTB overlap construction requires an overlap model.");
        }
        return build_images(atomic_data, true);
    }

private:
    std::unique_ptr<ImageContainer<double>> build_images(
        AtomicData& atomic_data,
        bool use_overlap
    ) const {
        validate_atomic_data(atomic_data);

        auto images = std::make_unique<ImageContainer<double>>(&atomic_data);
        int start = 0;
        int end = 0;
        atomic_data.get_vertex_range_local(&start, &end);

        for (int i = start; i < end; ++i) {
            if (atomic_data.atom_is_ghost(i)) {
                continue;
            }

            int type = 0;
            atomic_data.get_atom_type(i, &type);
            const auto& onsite = use_overlap
                ? resolved_.atom_types[type].s_onsite
                : resolved_.atom_types[type].h_onsite;
            const auto block = detail::diagonal_block_row_major(onsite);
            const int gid = atomic_data.get_global_index(i);
            const std::vector<int> zero_shift = {0, 0, 0};
            images->add_block(
                zero_shift,
                gid,
                gid,
                block.data(),
                static_cast<int>(onsite.size()),
                static_cast<int>(onsite.size()),
                AssemblyMode::ADD,
                MatrixLayout::RowMajor);
        }

        int estart = 0;
        int eend = 0;
        atomic_data.get_edge_range_local(&estart, &eend);

        // Each AtomicData edge maps to one unique `(row, col, R)` image block.
        // With that graph invariant, SK block evaluation and insertion are independent per edge.
        #pragma omp parallel for schedule(dynamic)
        for (int edge = estart; edge < eend; ++edge) {
            int i = 0;
            int j = 0;
            atomic_data.get_connected_atoms(edge, &i, &j);

            double dx = 0.0;
            double dy = 0.0;
            double dz = 0.0;
            atomic_data.get_edge_vec(edge, &dx, &dy, &dz);

            int row_type = 0;
            int col_type = 0;
            atomic_data.get_atom_type(i, &row_type);
            atomic_data.get_atom_type(j, &col_type);

            const auto block = evaluate_pair_block(use_overlap, row_type, col_type, {dx, dy, dz});
            if (block.empty() || !detail::has_nonzero_entries(block)) {
                continue;
            }

            int rx = 0;
            int ry = 0;
            int rz = 0;
            atomic_data.get_edge_shift_vec(edge, &rx, &ry, &rz);

            int nrow = 0;
            int ncol = 0;
            atomic_data.get_atom_norb(i, &nrow);
            atomic_data.get_atom_norb(j, &ncol);

            const std::vector<int> shift = {rx, ry, rz};
            images->add_block(
                shift,
                atomic_data.get_global_index(i),
                atomic_data.get_global_index(j),
                block.data(),
                nrow,
                ncol,
                AssemblyMode::ADD,
                MatrixLayout::RowMajor);
        }

        images->assemble();
        return images;
    }

    void validate_atomic_data(AtomicData& atomic_data) const {
        if (atomic_data.n_type() != static_cast<int>(resolved_.atom_types.size())) {
            throw std::runtime_error(
                "SKTB type count mismatch: AtomicData has " + std::to_string(atomic_data.n_type()) +
                " types, but the resolved SKTB model has " + std::to_string(resolved_.atom_types.size()) + ".");
        }
        for (int type = 0; type < atomic_data.n_type(); ++type) {
            if (atomic_data.type_norb[type] != static_cast<int>(resolved_.atom_types[type].orbitals.size())) {
                throw std::runtime_error(
                    "SKTB orbital count mismatch for species " + resolved_.atom_types[type].symbol + ".");
            }
        }
    }

    detail::DBlock evaluate_direct_pair_block(
        bool use_overlap,
        int row_type,
        int col_type,
        const detail::Vec3& displacement
    ) const {
        const auto* pair = resolved_.FindPair(row_type, col_type);
        if (pair == nullptr) {
            return {};
        }

        const double distance = detail::norm(displacement);
        const double pair_cutoff = use_overlap ? pair->s_max_cutoff : pair->h_max_cutoff;
        if (distance <= 1e-12 || distance > pair_cutoff + 1e-12) {
            return {};
        }

        const auto frame = detail::build_local_frame(displacement);
        const std::array<detail::DBlock, 3> rotations = {
            detail::shell_rotation(0, frame),
            detail::shell_rotation(1, frame),
            detail::shell_rotation(2, frame)
        };

        const auto& row_atom = resolved_.atom_types[row_type];
        const auto& col_atom = resolved_.atom_types[col_type];
        detail::DBlock block(row_atom.orbitals.size() * col_atom.orbitals.size(), 0.0);

        for (const auto& channel : pair->channels) {
            const bool has_value = use_overlap ? channel.has_s : channel.has_h;
            if (!has_value) {
                continue;
            }
            const auto& radial = use_overlap ? channel.s : channel.h;
            if (distance > radial.Cutoff() + 1e-12) {
                continue;
            }
            const int row_l = channel.row_l;
            const int col_l = channel.col_l;

            const auto aligned = detail::build_aligned_shell_block(
                row_l, col_l, channel.channel, radial.Evaluate(distance));
            const auto rotated = detail::rotate_shell_block(
                aligned, row_l, col_l, rotations[static_cast<size_t>(row_l)], rotations[static_cast<size_t>(col_l)]);
            detail::copy_shell_block_to_pair_block(
                rotated, row_l, col_l, row_atom, col_atom, block);
        }

        return block;
    }

    /// Evaluates a pair block, falling back to the real Hermitian transpose.
    ///
    /// Args:
    ///   use_overlap: Whether to evaluate overlap instead of Hamiltonian channels.
    ///   row_type: Atom type of the row atom.
    ///   col_type: Atom type of the column atom.
    ///   displacement: Vector from row atom to column atom.
    /// Returns:
    ///   Row-major block for `(row_type, col_type)`, or empty when no pair exists.
    detail::DBlock evaluate_pair_block(
        bool use_overlap,
        int row_type,
        int col_type,
        const detail::Vec3& displacement
    ) const {
        auto direct = evaluate_direct_pair_block(use_overlap, row_type, col_type, displacement);
        if (!direct.empty()) {
            return direct;
        }

        auto reverse = evaluate_direct_pair_block(use_overlap, col_type, row_type, detail::negated(displacement));
        if (reverse.empty()) {
            return {};
        }

        const int nrow = static_cast<int>(resolved_.atom_types[col_type].orbitals.size());
        const int ncol = static_cast<int>(resolved_.atom_types[row_type].orbitals.size());
        return detail::transpose_row_major(reverse, nrow, ncol);
    }

    SkTable resolved_;
};

}  // namespace sktb
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_CONSTRUCTION_HPP
