#include "../msd.hpp"
#include "../msd_spectral.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <array>
#include <vector>

using namespace rsatb::calc;

namespace {

using Scalar = std::complex<double>;
using Matrix = BlockSpMat<Scalar>;
using Vector = DistVector<Scalar>;
using MultiVector = DistMultiVector<Scalar>;
using Dense2 = std::array<Scalar, 4>;
using Dense4 = std::array<Scalar, 16>;
using DenseVec2 = std::array<Scalar, 2>;
using DenseVec4 = std::array<Scalar, 4>;

DistGraph make_graph(MPI_Comm comm, int n, bool dense) {
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        if (dense) {
            adj[i].reserve(n);
            for (int j = 0; j < n; ++j) {
                adj[i].push_back(j);
            }
        } else {
            adj[i] = {i};
        }
    }

    DistGraph graph(comm);
    graph.construct_serial(n, block_sizes, adj);
    return graph;
}

void fill_matrix(Matrix& mat, const std::vector<std::vector<Scalar>>& values) {
    for (int local_row = 0; local_row < static_cast<int>(mat.graph->owned_global_indices.size()); ++local_row) {
        const int global_row = mat.graph->owned_global_indices[local_row];
        for (int j = 0; j < static_cast<int>(values[global_row].size()); ++j) {
            if (std::abs(values[global_row][j]) > 1e-14) {
                Scalar val = values[global_row][j];
                mat.add_block(global_row, j, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
    }
    mat.assemble();
}

void fill_vector(Vector& vec, const std::vector<Scalar>& values) {
    vec.set_constant(Scalar(0.0, 0.0));
    for (int local_row = 0; local_row < vec.local_size; ++local_row) {
        const int global_row = vec.graph->owned_global_indices[local_row];
        vec[local_row] = values[static_cast<size_t>(global_row)];
    }
}

Dense2 dense2_identity() {
    return {Scalar(1.0, 0.0), Scalar(0.0, 0.0),
            Scalar(0.0, 0.0), Scalar(1.0, 0.0)};
}

Dense4 dense4_identity() {
    return {
        Scalar(1.0, 0.0), Scalar(0.0, 0.0), Scalar(0.0, 0.0), Scalar(0.0, 0.0),
        Scalar(0.0, 0.0), Scalar(1.0, 0.0), Scalar(0.0, 0.0), Scalar(0.0, 0.0),
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), Scalar(1.0, 0.0), Scalar(0.0, 0.0),
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), Scalar(0.0, 0.0), Scalar(1.0, 0.0)
    };
}

Dense2 dense2_add(const Dense2& a, const Dense2& b) {
    Dense2 out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Dense4 dense4_add(const Dense4& a, const Dense4& b) {
    Dense4 out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Dense2 dense2_scale(const Dense2& a, Scalar alpha) {
    Dense2 out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = alpha * a[i];
    }
    return out;
}

Dense4 dense4_scale(const Dense4& a, Scalar alpha) {
    Dense4 out{};
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = alpha * a[i];
    }
    return out;
}

Dense2 dense2_mul(const Dense2& a, const Dense2& b) {
    Dense2 out{};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Scalar sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += a[static_cast<size_t>(i) * 2 + static_cast<size_t>(k)] *
                       b[static_cast<size_t>(k) * 2 + static_cast<size_t>(j)];
            }
            out[static_cast<size_t>(i) * 2 + static_cast<size_t>(j)] = sum;
        }
    }
    return out;
}

Dense4 dense4_mul(const Dense4& a, const Dense4& b) {
    Dense4 out{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Scalar sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += a[static_cast<size_t>(i) * 4 + static_cast<size_t>(k)] *
                       b[static_cast<size_t>(k) * 4 + static_cast<size_t>(j)];
            }
            out[static_cast<size_t>(i) * 4 + static_cast<size_t>(j)] = sum;
        }
    }
    return out;
}

DenseVec2 dense2_mul_vec(const Dense2& a, const DenseVec2& x) {
    DenseVec2 out{};
    for (int i = 0; i < 2; ++i) {
        Scalar sum = 0.0;
        for (int j = 0; j < 2; ++j) {
            sum += a[static_cast<size_t>(i) * 2 + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
        }
        out[static_cast<size_t>(i)] = sum;
    }
    return out;
}

DenseVec4 dense4_mul_vec(const Dense4& a, const DenseVec4& x) {
    DenseVec4 out{};
    for (int i = 0; i < 4; ++i) {
        Scalar sum = 0.0;
        for (int j = 0; j < 4; ++j) {
            sum += a[static_cast<size_t>(i) * 4 + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
        }
        out[static_cast<size_t>(i)] = sum;
    }
    return out;
}

Dense4 dense4_expm(const Dense4& a) {
    const double norm_bound = [&]() {
        double max_row = 0.0;
        for (int i = 0; i < 4; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < 4; ++j) {
                row_sum += std::abs(a[static_cast<size_t>(i) * 4 + static_cast<size_t>(j)]);
            }
            max_row = std::max(max_row, row_sum);
        }
        return max_row;
    }();

    int squarings = 0;
    if (norm_bound > 0.5) {
        squarings = static_cast<int>(std::ceil(std::log2(norm_bound / 0.5)));
    }
    const double scale = std::ldexp(1.0, squarings);
    Dense4 scaled = dense4_scale(a, Scalar(1.0 / scale, 0.0));

    Dense4 result = dense4_identity();
    Dense4 term = dense4_identity();
    for (int n = 1; n <= 40; ++n) {
        term = dense4_mul(term, scaled);
        term = dense4_scale(term, Scalar(1.0 / static_cast<double>(n), 0.0));
        result = dense4_add(result, term);
    }

    for (int i = 0; i < squarings; ++i) {
        result = dense4_mul(result, result);
    }
    return result;
}

Dense2 dense2_expm(const Dense2& a) {
    const double norm_bound = [&]() {
        double max_row = 0.0;
        for (int i = 0; i < 2; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < 2; ++j) {
                row_sum += std::abs(a[static_cast<size_t>(i) * 2 + static_cast<size_t>(j)]);
            }
            max_row = std::max(max_row, row_sum);
        }
        return max_row;
    }();

    int squarings = 0;
    if (norm_bound > 0.5) {
        squarings = static_cast<int>(std::ceil(std::log2(norm_bound / 0.5)));
    }
    const double scale = std::ldexp(1.0, squarings);
    Dense2 scaled = dense2_scale(a, Scalar(1.0 / scale, 0.0));

    Dense2 result = dense2_identity();
    Dense2 term = dense2_identity();
    for (int n = 1; n <= 40; ++n) {
        term = dense2_mul(term, scaled);
        term = dense2_scale(term, Scalar(1.0 / static_cast<double>(n), 0.0));
        result = dense2_add(result, term);
    }
    for (int i = 0; i < squarings; ++i) {
        result = dense2_mul(result, result);
    }
    return result;
}

Scalar dense2_trace(const Dense2& a) { return a[0] + a[3]; }

// Chebyshev matrix polynomials T_p(a) for p = 0..n-1, dense.
std::vector<Dense2> dense2_chebyshev_family(const Dense2& a, int n) {
    std::vector<Dense2> t;
    t.reserve(static_cast<size_t>(n));
    t.push_back(dense2_identity());
    if (n > 1) t.push_back(a);
    for (int p = 2; p < n; ++p) {
        Dense2 next = dense2_scale(dense2_mul(a, t[static_cast<size_t>(p - 1)]), Scalar(2.0, 0.0));
        next = dense2_add(next, dense2_scale(t[static_cast<size_t>(p - 2)], Scalar(-1.0, 0.0)));
        t.push_back(next);
    }
    return t;
}

std::vector<Scalar> dense_chebyshev_moments(const Dense2& h_scaled, const DenseVec2& v, int n_moments) {
    std::vector<Scalar> moments(static_cast<size_t>(n_moments), Scalar(0.0, 0.0));
    DenseVec2 t0 = v;
    auto dot = [](const DenseVec2& left, const DenseVec2& right) {
        Scalar sum = 0.0;
        for (int i = 0; i < 2; ++i) {
            sum += std::conj(left[static_cast<size_t>(i)]) * right[static_cast<size_t>(i)];
        }
        return sum;
    };

    moments[0] = dot(v, t0);
    if (n_moments > 1) {
        DenseVec2 t1 = dense2_mul_vec(h_scaled, t0);
        moments[1] = dot(v, t1);
        for (int n = 2; n < n_moments; ++n) {
            DenseVec2 t2 = dense2_mul_vec(h_scaled, t1);
            for (int i = 0; i < 2; ++i) {
                t2[static_cast<size_t>(i)] = Scalar(2.0, 0.0) * t2[static_cast<size_t>(i)] - t0[static_cast<size_t>(i)];
            }
            moments[static_cast<size_t>(n)] = dot(v, t2);
            t0 = t1;
            t1 = t2;
        }
    }
    return moments;
}

std::vector<cd> to_complex_moments(const std::vector<Scalar>& moments) {
    std::vector<cd> lifted(moments.size(), cd(0.0, 0.0));
    for (size_t i = 0; i < moments.size(); ++i) {
        lifted[i] = static_cast<cd>(moments[i]);
    }
    return lifted;
}

} // namespace

TEST(MSDTest, CommutingDiagonalOperatorsGiveZeroSpectrumWithOverlap) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, false);
    Matrix H(&graph);
    Matrix S(&graph);
    Matrix X(&graph);

    fill_matrix(H, {
        {Scalar(-0.4, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.35, 0.0)}
    });
    fill_matrix(S, {
        {Scalar(1.2, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.9, 0.0)}
    });
    fill_matrix(X, {
        {Scalar(-0.7, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(1.1, 0.0)}
    });

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.3, -0.4)});

    Chebyshev<Scalar> cheby(H, S, -1.0, 1.0);
    std::vector<double> msd;
    std::vector<double> energy;
    corr_msd(cheby, X, sample, 3, 0.5, 64, msd, energy);

    ASSERT_EQ(energy.size(), 64u);
    ASSERT_EQ(msd.size(), static_cast<size_t>(4 * 64));
    for (double value : msd) {
        EXPECT_NEAR(value, 0.0, 1e-10);
    }
}

TEST(MSDTest, RepeatedSamplesInMultiVectorMatchSingleVectorRun) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph);
    Matrix X(&graph);

    fill_matrix(H, {
        {Scalar(0.0, 0.0), Scalar(0.35, 0.0)},
        {Scalar(0.35, 0.0), Scalar(0.1, 0.0)}
    });
    fill_matrix(X, {
        {Scalar(-0.5, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(0.5, 0.0)}
    });

    Vector sample_a(&graph);
    fill_vector(sample_a, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.0, 1.0 / std::sqrt(2.0))});

    std::vector<double> msd_single;
    std::vector<double> energy_single;
    std::vector<double> energy_mv;

    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, X, sample_a, 2, 0.4, 48, msd_single, energy_single);
    }

    MultiVector samples(&graph, 2);
    samples.set_col(0, sample_a);
    samples.set_col(1, sample_a);

    std::vector<double> msd_mv;
    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, X, samples, 2, 0.4, 48, msd_mv, energy_mv);
    }

    ASSERT_EQ(msd_mv.size(), msd_single.size());
    ASSERT_EQ(energy_single.size(), energy_mv.size());
    for (size_t i = 0; i < energy_single.size(); ++i) {
        EXPECT_NEAR(energy_single[i], energy_mv[i], 1e-12);
    }
    for (size_t i = 0; i < msd_single.size(); ++i) {
        EXPECT_NEAR(msd_mv[i], msd_single[i], 1e-10);
    }
}

TEST(MSDTest, DiagonalExplicitPositionMultiVectorMatchesSingleVectorRun) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph);

    fill_matrix(H, {
        {Scalar(0.0, 0.0), Scalar(0.35, 0.0)},
        {Scalar(0.35, 0.0), Scalar(0.1, 0.0)}
    });

    Vector diagonal_x(&graph);
    fill_vector(diagonal_x, {Scalar(-0.5, 0.0), Scalar(0.5, 0.0)});

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.0, 1.0 / std::sqrt(2.0))});

    std::vector<double> msd_single;
    std::vector<double> energy_single;
    std::vector<double> msd_multi;
    std::vector<double> energy_multi;

    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, diagonal_x, sample, 2, 0.4, 48, msd_single, energy_single);
    }

    MultiVector samples(&graph, 2);
    samples.set_col(0, sample);
    samples.set_col(1, sample);

    {
        Chebyshev<Scalar> cheby(H, -1.0, 1.0);
        corr_msd(cheby, diagonal_x, samples, 2, 0.4, 48, msd_multi, energy_multi);
    }

    ASSERT_EQ(energy_single.size(), energy_multi.size());
    ASSERT_EQ(msd_single.size(), msd_multi.size());
    for (size_t i = 0; i < energy_single.size(); ++i) {
        EXPECT_NEAR(energy_single[i], energy_multi[i], 1e-12);
    }
    for (size_t i = 0; i < msd_single.size(); ++i) {
        EXPECT_NEAR(msd_single[i], msd_multi[i], 1e-10);
    }
}

TEST(MSDTest, GammaAffineRouteMatchesDenseReference) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H0(&graph);
    Matrix H1(&graph);

    fill_matrix(H0, {
        {Scalar(-0.2, 0.0), Scalar(0.35, 0.0)},
        {Scalar(0.35, 0.0), Scalar(0.1, 0.0)}
    });
    fill_matrix(H1, {
        {Scalar(0.0, 0.0), Scalar(0.18, 0.0)},
        {Scalar(0.18, 0.0), Scalar(0.0, 0.0)}
    });

    Vector diagonal_x(&graph);
    fill_vector(diagonal_x, {Scalar(0.0, 0.0), Scalar(0.4, 0.0)});

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.0, 1.0 / std::sqrt(2.0))});

    Chebyshev<Scalar> cheby(H0, -1.0, 1.0);
    std::vector<double> msd;
    std::vector<double> energy;
    const int n_steps = 2;
    const double dt_fs = 0.4;
    const int n_kpm = 48;
    corr_msd(cheby, H1, diagonal_x, sample, n_steps, dt_fs, n_kpm, msd, energy);

    const Dense2 h0_dense = {
        Scalar(-0.2, 0.0), Scalar(0.35, 0.0),
        Scalar(0.35, 0.0), Scalar(0.1, 0.0)
    };
    const Dense2 h1_dense = {
        Scalar(0.0, 0.0), Scalar(0.18, 0.0),
        Scalar(0.18, 0.0), Scalar(0.0, 0.0)
    };
    const Dense2 x0_dense = {
        Scalar(0.0, 0.0), Scalar(0.0, 0.0),
        Scalar(0.0, 0.0), Scalar(0.4, 0.0)
    };
    const Dense2 h_scaled = dense2_scale(dense2_add(h0_dense, dense2_scale(dense2_identity(), Scalar(-cheby.center, 0.0))), Scalar(1.0 / cheby.half_width, 0.0));

    const double hbar_ev_fs = 0.6582119569;
    const double time = dt_fs / hbar_ev_fs;
    Dense4 affine_generator = {
        h0_dense[0], h0_dense[1], h1_dense[0], h1_dense[1],
        h0_dense[2], h0_dense[3], h1_dense[2], h1_dense[3],
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), h0_dense[0], h0_dense[1],
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), h0_dense[2], h0_dense[3]
    };
    for (auto& value : affine_generator) {
        value *= Scalar(0.0, -1.0);
    }

    const DenseVec2 psi_dense = {
        Scalar(1.0 / std::sqrt(2.0), 0.0),
        Scalar(0.0, 1.0 / std::sqrt(2.0))
    };
    const DenseVec2 xpsi_dense = dense2_mul_vec(x0_dense, psi_dense);
    const DenseVec4 initial_affine = {
        xpsi_dense[0], xpsi_dense[1],
        psi_dense[0], psi_dense[1]
    };

    std::vector<cd> density_moments = to_complex_moments(dense_chebyshev_moments(h_scaled, psi_dense, n_kpm));
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);

    ASSERT_EQ(msd.size(), static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm));
    ASSERT_EQ(energy.size(), static_cast<size_t>(n_kpm));

    for (int step = 0; step <= n_steps; ++step) {
        const Dense4 propagator = dense4_expm(dense4_scale(affine_generator, Scalar(time * static_cast<double>(step), 0.0)));
        const DenseVec4 propagated = dense4_mul_vec(propagator, initial_affine);
        const DenseVec2 a = {propagated[0], propagated[1]};
        const DenseVec2 b = {propagated[2], propagated[3]};
        DenseVec2 comm = dense2_mul_vec(x0_dense, b);
        comm[0] -= a[0];
        comm[1] -= a[1];

        std::vector<cd> msd_moments = to_complex_moments(dense_chebyshev_moments(h_scaled, comm, n_kpm));
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);

        for (int k = 0; k < n_kpm; ++k) {
            const double rho = density_spectrum[static_cast<size_t>(k)].real();
            double expected = 0.0;
            if (std::abs(rho) > 1e-12) {
                expected = msd_spectrum[static_cast<size_t>(k)].real() / rho;
            }
            EXPECT_NEAR(
                msd[static_cast<size_t>(step) * static_cast<size_t>(n_kpm) + static_cast<size_t>(k)],
                expected,
                1e-8);
        }
    }
}

// The commutator (non-orthogonal periodic) route against a dense reference.
//
// A 2x2 non-orthogonal fixture: S non-trivial, H Hermitian, and arbitrary
// sparse pieces {bond, xbar, delta} standing in for the section-4 assembly --
// the route only requires that the driver it is handed IS the operator the
// dense reference exponentiates, so the pieces need no geometric meaning here.
// The dense side builds V = i(bond - xbar S^-1 H + H S^-1 delta), the
// coefficient driver C = i S^-1 V, propagates the pair generator
// [[A, C], [0, A]] exactly (A = S^-1 H), reads out phi(t) = [X, U(t)] psi, and
// takes S-WEIGHTED Chebyshev moments of the scaled A -- everything the
// production route must reproduce through CG solves and apply_dual.
TEST(MSDTest, CommutatorRouteMatchesDenseReferenceWithOverlap) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H0(&graph), S0(&graph), Bond(&graph), Xbar(&graph), Delta(&graph);

    fill_matrix(H0, {
        {Scalar(-0.25, 0.0), Scalar(0.3, 0.05)},
        {Scalar(0.3, -0.05), Scalar(0.15, 0.0)}
    });
    fill_matrix(S0, {
        {Scalar(1.0, 0.0), Scalar(0.2, 0.0)},
        {Scalar(0.2, 0.0), Scalar(1.0, 0.0)}
    });
    fill_matrix(Bond, {
        {Scalar(0.0, 0.0), Scalar(0.0, 0.22)},
        {Scalar(0.0, 0.22), Scalar(0.0, 0.0)}
    });
    fill_matrix(Xbar, {
        {Scalar(0.05, 0.0), Scalar(0.12, 0.0)},
        {Scalar(0.12, 0.0), Scalar(0.4, 0.0)}
    });
    fill_matrix(Delta, {
        {Scalar(0.03, 0.0), Scalar(0.07, 0.02)},
        {Scalar(0.07, -0.02), Scalar(-0.05, 0.0)}
    });

    Vector sample(&graph);
    fill_vector(sample, {Scalar(0.8, 0.0), Scalar(0.0, 0.6)});

    // Solve form: the route must metric-weight its moments and dress the
    // driver with S^-1 through CG. Window generous around spec(S^-1 H).
    Chebyshev<Scalar> cheby(H0, S0, -1.2, 1.2);

    std::vector<double> msd, energy;
    const int n_steps = 2;
    const double dt_fs = 0.4;
    const int n_kpm = 48;
    corr_msd(cheby, VelocityOp<Scalar>(Bond, Xbar, Delta), sample,
             n_steps, dt_fs, n_kpm, msd, energy);

    // ---- dense reference ----
    const Dense2 h0 = {Scalar(-0.25, 0.0), Scalar(0.3, 0.05),
                       Scalar(0.3, -0.05), Scalar(0.15, 0.0)};
    const Dense2 s0 = {Scalar(1.0, 0.0), Scalar(0.2, 0.0),
                       Scalar(0.2, 0.0), Scalar(1.0, 0.0)};
    const Dense2 bond = {Scalar(0.0, 0.0), Scalar(0.0, 0.22),
                         Scalar(0.0, 0.22), Scalar(0.0, 0.0)};
    const Dense2 xbar = {Scalar(0.05, 0.0), Scalar(0.12, 0.0),
                         Scalar(0.12, 0.0), Scalar(0.4, 0.0)};
    const Dense2 delta = {Scalar(0.03, 0.0), Scalar(0.07, 0.02),
                          Scalar(0.07, -0.02), Scalar(-0.05, 0.0)};

    // NOTE: fill_matrix fills ROW-major rows above; Dense2 helpers here index
    // {a00, a01, a10, a11}, consistent with the listings.
    auto inv2 = [](const Dense2& m) {
        const Scalar det = m[0] * m[3] - m[1] * m[2];
        return Dense2{m[3] / det, -m[1] / det, -m[2] / det, m[0] / det};
    };
    const Dense2 s_inv = inv2(s0);
    const Dense2 a0 = dense2_mul(s_inv, h0);                     // A = S^-1 H
    // V = i (bond - xbar A + H S^-1 delta)
    Dense2 v{};
    {
        const Dense2 t1 = dense2_mul(xbar, a0);
        const Dense2 t2 = dense2_mul(h0, dense2_mul(s_inv, delta));
        for (int i = 0; i < 4; ++i) {
            v[static_cast<size_t>(i)] = Scalar(0.0, 1.0)
                * (bond[static_cast<size_t>(i)] - t1[static_cast<size_t>(i)]
                   + t2[static_cast<size_t>(i)]);
        }
    }
    const Dense2 c_driver_raw = dense2_mul(s_inv, v);
    Dense2 c_driver{};
    for (int i = 0; i < 4; ++i) {
        c_driver[static_cast<size_t>(i)] = Scalar(0.0, 1.0) * c_driver_raw[static_cast<size_t>(i)];
    }

    // Pair generator exp(-i t [[A, C],[0, A]]) acting on (phi, psi), phi(0)=0.
    const double hbar_ev_fs = 0.6582119569;
    const double time = dt_fs / hbar_ev_fs;
    Dense4 gen = {
        a0[0], a0[1], c_driver[0], c_driver[1],
        a0[2], a0[3], c_driver[2], c_driver[3],
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), a0[0], a0[1],
        Scalar(0.0, 0.0), Scalar(0.0, 0.0), a0[2], a0[3]
    };
    for (auto& value : gen) value *= Scalar(0.0, -1.0);

    const DenseVec2 psi = {Scalar(0.8, 0.0), Scalar(0.0, 0.6)};
    const DenseVec4 initial = {Scalar(0.0, 0.0), Scalar(0.0, 0.0), psi[0], psi[1]};

    // S-weighted dense moments of the scaled A.
    const Dense2 a_scaled = dense2_scale(
        dense2_add(a0, dense2_scale(dense2_identity(), Scalar(-cheby.center, 0.0))),
        Scalar(1.0 / cheby.half_width, 0.0));
    auto weighted_moments = [&](const DenseVec2& phi) {
        const DenseVec2 left = dense2_mul_vec(s0, phi);
        std::vector<Scalar> moments(static_cast<size_t>(n_kpm), Scalar(0.0, 0.0));
        auto dot = [](const DenseVec2& l, const DenseVec2& r) {
            Scalar sum = 0.0;
            for (int i = 0; i < 2; ++i) sum += std::conj(l[static_cast<size_t>(i)]) * r[static_cast<size_t>(i)];
            return sum;
        };
        DenseVec2 t0 = phi;
        moments[0] = dot(left, t0);
        if (n_kpm > 1) {
            DenseVec2 t1 = dense2_mul_vec(a_scaled, t0);
            moments[1] = dot(left, t1);
            for (int n = 2; n < n_kpm; ++n) {
                DenseVec2 t2 = dense2_mul_vec(a_scaled, t1);
                for (int i = 0; i < 2; ++i)
                    t2[static_cast<size_t>(i)] = Scalar(2.0, 0.0) * t2[static_cast<size_t>(i)]
                        - t0[static_cast<size_t>(i)];
                moments[static_cast<size_t>(n)] = dot(left, t2);
                t0 = t1;
                t1 = t2;
            }
        }
        return moments;
    };

    std::vector<cd> density_moments = to_complex_moments(weighted_moments(psi));
    std::vector<cd> density_spectrum;
    reconstruct_spectrum_from_moments(density_moments, density_spectrum, nullptr, true);

    ASSERT_EQ(msd.size(), static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm));

    for (int step = 0; step <= n_steps; ++step) {
        const Dense4 propagator =
            dense4_expm(dense4_scale(gen, Scalar(time * static_cast<double>(step), 0.0)));
        const DenseVec4 propagated = dense4_mul_vec(propagator, initial);
        const DenseVec2 phi = {propagated[0], propagated[1]};

        std::vector<cd> msd_moments = to_complex_moments(weighted_moments(phi));
        std::vector<cd> msd_spectrum;
        reconstruct_spectrum_from_moments(msd_moments, msd_spectrum, nullptr, true);

        for (int k = 0; k < n_kpm; ++k) {
            const double rho = density_spectrum[static_cast<size_t>(k)].real();
            double expected = 0.0;
            if (std::abs(rho) > 1e-12) {
                expected = msd_spectrum[static_cast<size_t>(k)].real() / rho;
            }
            EXPECT_NEAR(
                msd[static_cast<size_t>(step) * static_cast<size_t>(n_kpm) + static_cast<size_t>(k)],
                expected,
                1e-6);
        }
    }
}

TEST(MSDTest, GammaAffineRepeatedSamplesInMultiVectorMatchSingleVectorRun) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H0(&graph);
    Matrix H1(&graph);

    fill_matrix(H0, {
        {Scalar(-0.2, 0.0), Scalar(0.35, 0.0)},
        {Scalar(0.35, 0.0), Scalar(0.1, 0.0)}
    });
    fill_matrix(H1, {
        {Scalar(0.0, 0.0), Scalar(0.18, 0.0)},
        {Scalar(0.18, 0.0), Scalar(0.0, 0.0)}
    });

    Vector diagonal_x(&graph);
    fill_vector(diagonal_x, {Scalar(0.0, 0.0), Scalar(0.4, 0.0)});

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.0, 1.0 / std::sqrt(2.0))});

    std::vector<double> msd_single;
    std::vector<double> energy_single;
    std::vector<double> msd_multi;
    std::vector<double> energy_multi;
    const int n_steps = 2;
    const double dt_fs = 0.4;
    const int n_kpm = 48;

    {
        Chebyshev<Scalar> cheby(H0, -1.0, 1.0);
        corr_msd(
            cheby,
            H1,
            diagonal_x,
            sample,
            n_steps,
            dt_fs,
            n_kpm,
            msd_single,
            energy_single);
    }

    MultiVector samples(&graph, 2);
    samples.set_col(0, sample);
    samples.set_col(1, sample);

    {
        Chebyshev<Scalar> cheby(H0, -1.0, 1.0);
        corr_msd(
            cheby,
            H1,
            diagonal_x,
            samples,
            n_steps,
            dt_fs,
            n_kpm,
            msd_multi,
            energy_multi);
    }

    ASSERT_EQ(energy_single.size(), energy_multi.size());
    ASSERT_EQ(msd_single.size(), msd_multi.size());
    for (size_t i = 0; i < energy_single.size(); ++i) {
        EXPECT_NEAR(energy_single[i], energy_multi[i], 1e-12);
    }
    for (size_t i = 0; i < msd_single.size(); ++i) {
        EXPECT_NEAR(msd_single[i], msd_multi[i], 1e-10);
    }
}

// The diagonal-position overload applies X as a plain elementwise product,
// which is the position operator only in an orthonormal basis. Handed a
// Solve- or Sandwich-form Chebyshev it would iterate a dressed operator
// against an undressed X. Production never did, but only because
// observables.hpp routes overlap-bearing models to the matrix overload; the
// guard makes it a stated precondition rather than a routing accident.
TEST(MSDTest, DiagonalPositionOverloadRejectsNonOrthogonalForms) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph);
    Matrix S(&graph);
    fill_matrix(H, {
        {Scalar(-0.4, 0.0), Scalar(0.15, 0.2)},
        {Scalar(0.15, -0.2), Scalar(0.35, 0.0)}
    });
    fill_matrix(S, {
        {Scalar(1.2, 0.0), Scalar(0.1, 0.0)},
        {Scalar(0.1, 0.0), Scalar(0.9, 0.0)}
    });

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.3, -0.4)});
    Vector diagonal_position(&graph);
    fill_vector(diagonal_position, {Scalar(-0.7, 0.0), Scalar(1.1, 0.0)});

    std::vector<double> msd;
    std::vector<double> energy;

    Chebyshev<Scalar> cheby_solve(H, S, -1.0, 1.0);
    EXPECT_THROW(corr_msd(cheby_solve, diagonal_position, sample, 2, 0.5, 16, msd, energy),
                 std::runtime_error);

    // The guard is targeted: the orthogonal form this overload exists for
    // still runs.
    Chebyshev<Scalar> cheby_orth(H, -1.0, 1.0);
    EXPECT_NO_THROW(corr_msd(cheby_orth, diagonal_position, sample, 2, 0.5, 16, msd, energy));
}

// The overlap-form MSD, on a fixture that can actually see the dressing.
//
// CommutingDiagonalOperatorsGiveZeroSpectrumWithOverlap above cannot: it is
// built on a graph with no off-diagonal blocks, so everything commutes, the
// expected spectrum is identically zero, and S^-1 X, X S^-1 and a bare
// undressed X all reproduce it. Here H and X are complex Hermitian and do not
// commute, and the overlap is a multiple of the identity so that the solve
// form (S^-1 H, dressing X to S^-1 X) and an orthogonal run on H/s with X/s
// are the SAME operator and the same dressed X -- so the two must agree while
// reaching that answer by different code. The conj(X) run is the margin a
// conjugation slip would have to clear.
TEST(MSDTest, OverlapFormMatchesEquivalentOrthogonalRun) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    const double s_ovl = 2.0;

    Matrix H(&graph), X(&graph), S(&graph), Hs(&graph), Xs(&graph), Xconj(&graph);
    fill_matrix(H, {
        {Scalar(-0.4, 0.0), Scalar(0.15, 0.2)},
        {Scalar(0.15, -0.2), Scalar(0.35, 0.0)}
    });
    fill_matrix(X, {
        {Scalar(-0.7, 0.0), Scalar(0.3, 0.1)},
        {Scalar(0.3, -0.1), Scalar(1.1, 0.0)}
    });
    fill_matrix(S, {
        {Scalar(s_ovl, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(s_ovl, 0.0)}
    });
    fill_matrix(Hs, {
        {Scalar(-0.4 / s_ovl, 0.0), Scalar(0.15 / s_ovl, 0.2 / s_ovl)},
        {Scalar(0.15 / s_ovl, -0.2 / s_ovl), Scalar(0.35 / s_ovl, 0.0)}
    });
    fill_matrix(Xs, {
        {Scalar(-0.7 / s_ovl, 0.0), Scalar(0.3 / s_ovl, 0.1 / s_ovl)},
        {Scalar(0.3 / s_ovl, -0.1 / s_ovl), Scalar(1.1 / s_ovl, 0.0)}
    });
    fill_matrix(Xconj, {
        {Scalar(-0.7 / s_ovl, 0.0), Scalar(0.3 / s_ovl, -0.1 / s_ovl)},
        {Scalar(0.3 / s_ovl, 0.1 / s_ovl), Scalar(1.1 / s_ovl, 0.0)}
    });

    Vector sample(&graph);
    fill_vector(sample, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.3, -0.4)});

    const int n_steps = 3, n_kpm = 32;
    const double dt = 0.5;
    std::vector<double> msd_solve, msd_orth, msd_conj, e_solve, e_orth, e_conj;

    Chebyshev<Scalar> cheby_solve(H, S, -1.0, 1.0);
    corr_msd(cheby_solve, X, sample, n_steps, dt, n_kpm, msd_solve, e_solve);

    Chebyshev<Scalar> cheby_orth(Hs, -1.0, 1.0);
    corr_msd(cheby_orth, Xs, sample, n_steps, dt, n_kpm, msd_orth, e_orth);

    Chebyshev<Scalar> cheby_conj(Hs, -1.0, 1.0);
    corr_msd(cheby_conj, Xconj, sample, n_steps, dt, n_kpm, msd_conj, e_conj);

    ASSERT_EQ(msd_solve.size(), msd_orth.size());
    double scale = 0.0, gap_conj = 0.0;
    for (size_t i = 0; i < msd_orth.size(); ++i) {
        scale = std::max(scale, std::abs(msd_orth[i]));
        gap_conj = std::max(gap_conj, std::abs(msd_orth[i] - msd_conj[i]));
    }
    // Not the zero spectrum: this fixture actually exercises the dressing.
    EXPECT_GT(scale, 1e-6);
    // ... and it can tell X from conj(X).
    EXPECT_GT(gap_conj, 1e-3 * scale);

    for (size_t i = 0; i < msd_solve.size(); ++i) {
        EXPECT_NEAR(msd_solve[i], msd_orth[i], 1e-8 * std::max(1.0, scale));
    }
}

// The doubled metric-moment recurrence against the plain one-moment-per-apply
// recurrence it replaced, on operators whose forms exercise both metrics: the
// Solve form (S-weighted dots, one S product per moment pair) and the
// orthogonal form (plain dots). The product identity behind the doubling is
// an operator identity, not an approximation, so the two paths agree to
// rounding; the tolerance here is the "bit-comparable" bar, orders below the
// 1e-6..1e-8 the physics tests above compare spectra at.
TEST(MSDTest, DoubledMetricMomentsMatchPlainRecurrence) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph);
    Matrix S(&graph);
    fill_matrix(H, {
        {Scalar(-0.4, 0.0), Scalar(0.15, 0.2)},
        {Scalar(0.15, -0.2), Scalar(0.35, 0.0)}
    });
    fill_matrix(S, {
        {Scalar(1.2, 0.0), Scalar(0.1, 0.0)},
        {Scalar(0.1, 0.0), Scalar(0.9, 0.0)}
    });

    Vector sample_a(&graph);
    fill_vector(sample_a, {Scalar(1.0 / std::sqrt(2.0), 0.0), Scalar(0.3, -0.4)});
    Vector sample_b(&graph);
    fill_vector(sample_b, {Scalar(-0.2, 0.5), Scalar(0.6, 0.1)});

    // The recurrence the doubled path replaced, S-weighted left vector when
    // the form asks for one.
    auto plain_moments = [&](Chebyshev<Scalar>& cheby, const Vector& v, int n_kpm) {
        Vector left(&graph);
        if (cheby.form == ChebyOpForm::Solve) {
            Vector v_copy(&graph);
            v_copy.copy_from(v);
            cheby.S->mult(v_copy, left);
        } else {
            left.copy_from(v);
        }
        Vector t0(&graph), t1(&graph), t2(&graph);
        t0.copy_from(v);
        std::vector<cd> mu;
        mu.push_back(static_cast<cd>(left.dot(t0)));
        if (n_kpm > 1) {
            cheby.apply_op(t0, t1, ChebyMode::NORMAL);
            mu.push_back(static_cast<cd>(left.dot(t1)));
        }
        for (int n = 2; n < n_kpm; ++n) {
            cheby.apply_op(t1, t2, ChebyMode::NORMAL);
            t2.axpby(Scalar(-1.0), t0, Scalar(2.0));
            mu.push_back(static_cast<cd>(left.dot(t2)));
            t0.swap(t1);
            t1.swap(t2);
        }
        return mu;
    };

    // Odd and even moment counts, so both tails of the doubled loop run.
    for (int n_kpm : {32, 33}) {
        Chebyshev<Scalar> cheby_solve(H, S, -1.0, 1.0);
        std::vector<cd> doubled =
            rsatb::calc::detail::metric_density_moments(cheby_solve, sample_a, n_kpm);
        std::vector<cd> plain = plain_moments(cheby_solve, sample_a, n_kpm);
        ASSERT_EQ(doubled.size(), static_cast<size_t>(n_kpm));
        for (int n = 0; n < n_kpm; ++n) {
            EXPECT_NEAR(std::abs(doubled[static_cast<size_t>(n)] - plain[static_cast<size_t>(n)]),
                        0.0, 1e-12) << "solve form, moment " << n;
        }

        Chebyshev<Scalar> cheby_orth(H, -1.0, 1.0);
        doubled = rsatb::calc::detail::metric_density_moments(cheby_orth, sample_a, n_kpm);
        plain = plain_moments(cheby_orth, sample_a, n_kpm);
        for (int n = 0; n < n_kpm; ++n) {
            EXPECT_NEAR(std::abs(doubled[static_cast<size_t>(n)] - plain[static_cast<size_t>(n)]),
                        0.0, 1e-12) << "orthogonal form, moment " << n;
        }
    }

    // Multivector with DISTINCT columns: the block moments are the average of
    // the per-column moments, so a column mixup would not cancel here.
    {
        const int n_kpm = 33;
        MultiVector samples(&graph, 2);
        samples.set_col(0, sample_a);
        samples.set_col(1, sample_b);

        Chebyshev<Scalar> cheby_solve(H, S, -1.0, 1.0);
        std::vector<cd> block =
            rsatb::calc::detail::metric_density_moments(cheby_solve, samples, n_kpm);
        std::vector<cd> ref_a = plain_moments(cheby_solve, sample_a, n_kpm);
        std::vector<cd> ref_b = plain_moments(cheby_solve, sample_b, n_kpm);
        for (int n = 0; n < n_kpm; ++n) {
            const cd expected = 0.5 * (ref_a[static_cast<size_t>(n)] + ref_b[static_cast<size_t>(n)]);
            EXPECT_NEAR(std::abs(block[static_cast<size_t>(n)] - expected), 0.0, 1e-12)
                << "multivector solve form, moment " << n;
        }
    }
}

// The spectral (single-chain VAC) route against a dense reference and against
// the stepwise route, orthogonal form. Summed over the complete basis both
// stochastic estimators ARE the exact trace, so the three must agree; the
// spectral route's residual is its cardinal time quadrature.
TEST(MSDTest, SpectralRouteMatchesDenseAndStepwiseOrthogonal) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph), X(&graph), V(&graph);

    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 x = {Scalar(-0.7, 0.0), Scalar(0.3, 0.1),
                      Scalar(0.3, -0.1), Scalar(1.1, 0.0)};
    // V = i[H, X], Hermitian; the commutator convention the engines consume.
    Dense2 v{};
    {
        const Dense2 hx = dense2_mul(h, x);
        const Dense2 xh = dense2_mul(x, h);
        for (int i = 0; i < 4; ++i) {
            v[static_cast<size_t>(i)] = Scalar(0.0, 1.0) *
                (hx[static_cast<size_t>(i)] - xh[static_cast<size_t>(i)]);
        }
    }
    fill_matrix(H, {{h[0], h[1]}, {h[2], h[3]}});
    fill_matrix(X, {{x[0], x[1]}, {x[2], x[3]}});
    fill_matrix(V, {{v[0], v[1]}, {v[2], v[3]}});

    const int n_steps = 3, n_kpm = 32;
    const double dt_fs = 1.5;
    const double emin = -1.2, emax = 1.2;

    // Complete basis as a sample block: the averaged moments are Tr[...]/2.
    Vector e0(&graph), e1(&graph);
    fill_vector(e0, {Scalar(1.0, 0.0), Scalar(0.0, 0.0)});
    fill_vector(e1, {Scalar(0.0, 0.0), Scalar(1.0, 0.0)});
    MultiVector samples(&graph, 2);
    samples.set_col(0, e0);
    samples.set_col(1, e1);

    std::vector<double> msd_spec, e_spec, msd_step, e_step;
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                          msd_spec, e_spec);
    }
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd(cheby, X, samples, n_steps, dt_fs, n_kpm, msd_step, e_step);
    }

    // Dense reference: trace moments of (X(t) - X) T_p (X(t) - X), averaged
    // over the 2 basis columns, through the shared reconstruction.
    const double hw = 0.5 * (emax - emin);
    const double center = 0.5 * (emax + emin);
    const double hbar = 0.6582119569;
    const Dense2 a_s = dense2_scale(
        dense2_add(h, dense2_scale(dense2_identity(), Scalar(-center, 0.0))),
        Scalar(1.0 / hw, 0.0));
    const std::vector<Dense2> tfam = dense2_chebyshev_family(a_s, n_kpm);

    std::vector<cd> rho_ref(static_cast<size_t>(n_kpm));
    for (int p = 0; p < n_kpm; ++p) {
        rho_ref[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(tfam[static_cast<size_t>(p)])) * 0.5;
    }
    std::vector<cd> rho_spectrum;
    reconstruct_spectrum_from_moments(rho_ref, rho_spectrum, nullptr, true);

    ASSERT_EQ(msd_spec.size(), static_cast<size_t>(n_steps + 1) * n_kpm);
    ASSERT_EQ(msd_step.size(), msd_spec.size());

    double max_err_spec = 0.0, max_err_step = 0.0, scale_ref = 0.0;
    for (int k = 0; k <= n_steps; ++k) {
        const double tau = k * dt_fs * hw / hbar;
        const Dense2 up = dense2_expm(dense2_scale(a_s, Scalar(0.0, tau)));
        const Dense2 um = dense2_expm(dense2_scale(a_s, Scalar(0.0, -tau)));
        const Dense2 xt = dense2_mul(up, dense2_mul(x, um));
        const Dense2 dx = dense2_add(xt, dense2_scale(x, Scalar(-1.0, 0.0)));

        std::vector<cd> num_ref(static_cast<size_t>(n_kpm));
        for (int p = 0; p < n_kpm; ++p) {
            const Dense2 mid = dense2_mul(dx, dense2_mul(tfam[static_cast<size_t>(p)], dx));
            num_ref[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(mid)) * 0.5;
        }
        std::vector<cd> num_spectrum;
        reconstruct_spectrum_from_moments(num_ref, num_spectrum, nullptr, true);

        for (int i = 0; i < n_kpm; ++i) {
            const double rho = rho_spectrum[static_cast<size_t>(i)].real();
            double ref = 0.0;
            if (std::abs(rho) > 1e-12) ref = num_spectrum[static_cast<size_t>(i)].real() / rho;
            scale_ref = std::max(scale_ref, std::abs(ref));
            const size_t idx = static_cast<size_t>(k) * n_kpm + static_cast<size_t>(i);
            max_err_spec = std::max(max_err_spec, std::abs(msd_spec[idx] - ref));
            max_err_step = std::max(max_err_step, std::abs(msd_step[idx] - ref));
        }
    }
    std::cout << "spectral vs dense max err " << max_err_spec
              << ", stepwise vs dense max err " << max_err_step
              << ", scale " << scale_ref << std::endl;
    EXPECT_GT(scale_ref, 1e-3);
    EXPECT_LT(max_err_step, 1e-6 * std::max(1.0, scale_ref));
    EXPECT_LT(max_err_spec, 1e-6 * std::max(1.0, scale_ref));
}

// Long-window regression of the spectral route: the benchmark regime is
// hundreds of scaled-time units, far past the short windows of the tests
// above, and the cardinal quadrature, coefficient truncations, and block
// reseeding all accumulate there if they are going to.
TEST(MSDTest, SpectralRouteLongWindowOrthogonal) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph), X(&graph), V(&graph);

    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 x = {Scalar(-0.7, 0.0), Scalar(0.3, 0.1),
                      Scalar(0.3, -0.1), Scalar(1.1, 0.0)};
    Dense2 v{};
    {
        const Dense2 hx = dense2_mul(h, x);
        const Dense2 xh = dense2_mul(x, h);
        for (int i = 0; i < 4; ++i) {
            v[static_cast<size_t>(i)] = Scalar(0.0, 1.0) *
                (hx[static_cast<size_t>(i)] - xh[static_cast<size_t>(i)]);
        }
    }
    fill_matrix(H, {{h[0], h[1]}, {h[2], h[3]}});
    fill_matrix(X, {{x[0], x[1]}, {x[2], x[3]}});
    fill_matrix(V, {{v[0], v[1]}, {v[2], v[3]}});

    const int n_steps = 32, n_kpm = 24;
    const double dt_fs = 5.0;   // tau_step ~ 9.1, full window ~ 291 scaled
    const double emin = -1.2, emax = 1.2;

    Vector e0(&graph), e1(&graph);
    fill_vector(e0, {Scalar(1.0, 0.0), Scalar(0.0, 0.0)});
    fill_vector(e1, {Scalar(0.0, 0.0), Scalar(1.0, 0.0)});
    MultiVector samples(&graph, 2);
    samples.set_col(0, e0);
    samples.set_col(1, e1);

    std::vector<double> msd_spec, e_spec;
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                          msd_spec, e_spec);
    }

    const double hw = 0.5 * (emax - emin);
    const double center = 0.5 * (emax + emin);
    const double hbar = 0.6582119569;
    const Dense2 a_s = dense2_scale(
        dense2_add(h, dense2_scale(dense2_identity(), Scalar(-center, 0.0))),
        Scalar(1.0 / hw, 0.0));
    const std::vector<Dense2> tfam = dense2_chebyshev_family(a_s, n_kpm);

    std::vector<cd> rho_ref(static_cast<size_t>(n_kpm));
    for (int p = 0; p < n_kpm; ++p) {
        rho_ref[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(tfam[static_cast<size_t>(p)])) * 0.5;
    }
    std::vector<cd> rho_spectrum;
    reconstruct_spectrum_from_moments(rho_ref, rho_spectrum, nullptr, true);

    double max_err_spec = 0.0, scale_ref = 0.0;
    for (int k = 0; k <= n_steps; ++k) {
        const double tau = k * dt_fs * hw / hbar;
        const Dense2 up = dense2_expm(dense2_scale(a_s, Scalar(0.0, tau)));
        const Dense2 um = dense2_expm(dense2_scale(a_s, Scalar(0.0, -tau)));
        const Dense2 xt = dense2_mul(up, dense2_mul(x, um));
        const Dense2 dx = dense2_add(xt, dense2_scale(x, Scalar(-1.0, 0.0)));

        std::vector<cd> num_ref(static_cast<size_t>(n_kpm));
        for (int p = 0; p < n_kpm; ++p) {
            const Dense2 mid = dense2_mul(dx, dense2_mul(tfam[static_cast<size_t>(p)], dx));
            num_ref[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(mid)) * 0.5;
        }
        std::vector<cd> num_spectrum;
        reconstruct_spectrum_from_moments(num_ref, num_spectrum, nullptr, true);

        for (int i = 0; i < n_kpm; ++i) {
            const double rho = rho_spectrum[static_cast<size_t>(i)].real();
            double ref = 0.0;
            if (std::abs(rho) > 1e-12) ref = num_spectrum[static_cast<size_t>(i)].real() / rho;
            scale_ref = std::max(scale_ref, std::abs(ref));
            const size_t idx = static_cast<size_t>(k) * n_kpm + static_cast<size_t>(i);
            max_err_spec = std::max(max_err_spec, std::abs(msd_spec[idx] - ref));
        }
    }
    std::cout << "long-window spectral vs dense max err " << max_err_spec
              << ", scale " << scale_ref << std::endl;
    EXPECT_GT(scale_ref, 1e-3);
    EXPECT_LT(max_err_spec, 1e-5 * std::max(1.0, scale_ref));
}

// The spectral route on the Solve form: complete METRIC basis (columns of L
// with L L^H = S^-1), dense VAC reference built directly from the dressed
// velocity, and the stepwise commutator route on the same inputs.
TEST(MSDTest, SpectralRouteMatchesDenseWithOverlap) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph), S(&graph), V(&graph);

    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 s = {Scalar(1.2, 0.0), Scalar(0.1, 0.0),
                      Scalar(0.1, 0.0), Scalar(0.9, 0.0)};
    const Dense2 v = {Scalar(0.1, 0.0), Scalar(0.2, 0.1),
                      Scalar(0.2, -0.1), Scalar(-0.3, 0.0)};
    fill_matrix(H, {{h[0], h[1]}, {h[2], h[3]}});
    fill_matrix(S, {{s[0], s[1]}, {s[2], s[3]}});
    fill_matrix(V, {{v[0], v[1]}, {v[2], v[3]}});

    const int n_steps = 3, n_kpm = 32;
    const double dt_fs = 1.5;
    const double emin = -1.2, emax = 1.2;

    auto inv2 = [](const Dense2& m) {
        const Scalar det = m[0] * m[3] - m[1] * m[2];
        return Dense2{m[3] / det, -m[1] / det, -m[2] / det, m[0] / det};
    };
    const Dense2 s_inv = inv2(s);
    // Cholesky of S^-1 = L L^H; its columns are the complete metric basis.
    Dense2 lchol{};
    {
        const Scalar a = s_inv[0], b = s_inv[1], c = s_inv[3];
        const Scalar l00 = std::sqrt(a.real());
        lchol[0] = l00;
        lchol[2] = std::conj(b) / l00;
        lchol[3] = std::sqrt((c - std::norm(b) / a).real());
    }

    Vector r0(&graph), r1(&graph);
    fill_vector(r0, {lchol[0], lchol[2]});
    fill_vector(r1, {lchol[1], lchol[3]});
    MultiVector samples(&graph, 2);
    samples.set_col(0, r0);
    samples.set_col(1, r1);

    std::vector<double> msd_spec, e_spec, msd_step, e_step;
    {
        Chebyshev<Scalar> cheby(H, S, emin, emax);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                          msd_spec, e_spec);
    }
    {
        Chebyshev<Scalar> cheby(H, S, emin, emax);
        corr_msd(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                 msd_step, e_step);
    }

    ASSERT_EQ(msd_step.size(), msd_spec.size());

    // Dense reference straight from the VAC definition: coefficient traces
    // are basis-invariant, so num_p(t) = (2 / hw^2) int_0^t (t - u)
    // Re Tr[T_p(A_s) e^{+iA_s u} D e^{-iA_s u} D] du with D = S^-1 V.
    const double hw = 0.5 * (emax - emin);
    const double center = 0.5 * (emax + emin);
    const double hbar = 0.6582119569;
    const Dense2 a0 = dense2_mul(s_inv, h);
    const Dense2 a_s = dense2_scale(
        dense2_add(a0, dense2_scale(dense2_identity(), Scalar(-center, 0.0))),
        Scalar(1.0 / hw, 0.0));
    const Dense2 d_op = dense2_mul(s_inv, v);
    const std::vector<Dense2> tfam = dense2_chebyshev_family(a_s, n_kpm);

    std::vector<cd> rho_ref(static_cast<size_t>(n_kpm));
    for (int p = 0; p < n_kpm; ++p) {
        rho_ref[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(tfam[static_cast<size_t>(p)])) * 0.5;
    }
    std::vector<cd> rho_spectrum;
    reconstruct_spectrum_from_moments(rho_ref, rho_spectrum, nullptr, true);

    // Fine-grid VAC in scaled time, integrated per output step.
    const double tau_max = n_steps * dt_fs * hw / hbar;
    const double du = 1e-3;
    const int n_fine = static_cast<int>(std::ceil(tau_max / du)) + 1;
    const Dense2 estep = dense2_expm(dense2_scale(a_s, Scalar(0.0, du)));
    const Dense2 estep_m = dense2_expm(dense2_scale(a_s, Scalar(0.0, -du)));

    std::vector<std::vector<double>> vac(static_cast<size_t>(n_kpm),
                                         std::vector<double>(static_cast<size_t>(n_fine)));
    Dense2 up = dense2_identity(), um = dense2_identity();
    for (int f = 0; f < n_fine; ++f) {
        const Dense2 d_u = dense2_mul(up, dense2_mul(d_op, um));
        const Dense2 dd = dense2_mul(d_u, d_op);
        for (int p = 0; p < n_kpm; ++p) {
            vac[static_cast<size_t>(p)][static_cast<size_t>(f)] =
                static_cast<cd>(dense2_trace(dense2_mul(tfam[static_cast<size_t>(p)], dd))).real() * 0.5;
        }
        up = dense2_mul(estep, up);
        um = dense2_mul(um, estep_m);
    }

    double max_err_spec = 0.0, max_err_step = 0.0, scale_ref = 0.0;
    for (int k = 0; k <= n_steps; ++k) {
        const double tk = k * dt_fs * hw / hbar;
        const int nf = std::min(n_fine - 1, static_cast<int>(std::floor(tk / du)));
        std::vector<cd> num_ref(static_cast<size_t>(n_kpm));
        for (int p = 0; p < n_kpm; ++p) {
            double acc = 0.0;
            for (int f = 0; f <= nf; ++f) {
                const double u = f * du;
                const double wgt = (f == 0 || f == nf) ? 0.5 : 1.0;
                acc += wgt * (tk - u) * vac[static_cast<size_t>(p)][static_cast<size_t>(f)];
            }
            num_ref[static_cast<size_t>(p)] = cd(2.0 / (hw * hw) * acc * du, 0.0);
        }
        std::vector<cd> num_spectrum;
        reconstruct_spectrum_from_moments(num_ref, num_spectrum, nullptr, true);

        for (int i = 0; i < n_kpm; ++i) {
            const double rho = rho_spectrum[static_cast<size_t>(i)].real();
            double ref = 0.0;
            if (std::abs(rho) > 1e-12) ref = num_spectrum[static_cast<size_t>(i)].real() / rho;
            scale_ref = std::max(scale_ref, std::abs(ref));
            const size_t idx = static_cast<size_t>(k) * n_kpm + static_cast<size_t>(i);
            max_err_spec = std::max(max_err_spec, std::abs(msd_spec[idx] - ref));
            max_err_step = std::max(max_err_step, std::abs(msd_step[idx] - ref));
        }
    }
    std::cout << "solve spectral vs dense max err " << max_err_spec
              << ", stepwise vs dense max err " << max_err_step
              << ", scale " << scale_ref << std::endl;
    EXPECT_GT(scale_ref, 1e-3);
    EXPECT_LT(max_err_step, 1e-6 * std::max(1.0, scale_ref));
    EXPECT_LT(max_err_spec, 1e-6 * std::max(1.0, scale_ref));
}

// Burg linear prediction on a dense moment sequence: a 2x2 spectrum is two
// cosines in p, exactly the AR model class, so extrapolating 5x from 40
// computed moments must reproduce the directly computed sequence.
TEST(MSDTest, LinearPredictionRecoversDenseMoments) {
    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 a_s = dense2_scale(h, Scalar(1.0 / 1.2, 0.0));
    const int n_full = 200, n_short = 40;
    const std::vector<Dense2> tfam = dense2_chebyshev_family(a_s, n_full);

    std::vector<cd> full(static_cast<size_t>(n_full));
    for (int p = 0; p < n_full; ++p) {
        full[static_cast<size_t>(p)] = static_cast<cd>(dense2_trace(tfam[static_cast<size_t>(p)]));
    }
    std::vector<cd> shortm(full.begin(), full.begin() + n_short);
    const std::vector<cd> predicted = extrapolate_moments_lp(shortm, n_full);

    ASSERT_EQ(predicted.size(), static_cast<size_t>(n_full));
    double max_err = 0.0;
    for (int p = 0; p < n_full; ++p) {
        max_err = std::max(max_err, std::abs(predicted[static_cast<size_t>(p)] -
                                             full[static_cast<size_t>(p)]));
    }
    std::cout << "lp extrapolation max err " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-8);
}

// The n_kpm_compute knob end to end: both routes at quarter order plus
// linear prediction against their own full-order runs. On the 2x2 fixture
// the moment sequences are exact AR signals, so the knob must cost nothing.
TEST(MSDTest, MomentExtrapolationMatchesFullOrderRuns) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph), X(&graph), V(&graph);

    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 x = {Scalar(-0.7, 0.0), Scalar(0.3, 0.1),
                      Scalar(0.3, -0.1), Scalar(1.1, 0.0)};
    Dense2 v{};
    {
        const Dense2 hx = dense2_mul(h, x);
        const Dense2 xh = dense2_mul(x, h);
        for (int i = 0; i < 4; ++i) {
            v[static_cast<size_t>(i)] = Scalar(0.0, 1.0) *
                (hx[static_cast<size_t>(i)] - xh[static_cast<size_t>(i)]);
        }
    }
    fill_matrix(H, {{h[0], h[1]}, {h[2], h[3]}});
    fill_matrix(X, {{x[0], x[1]}, {x[2], x[3]}});
    fill_matrix(V, {{v[0], v[1]}, {v[2], v[3]}});

    const int n_steps = 3, n_kpm = 64, n_compute = 16;
    const double dt_fs = 1.5;
    const double emin = -1.2, emax = 1.2;

    Vector e0(&graph), e1(&graph);
    fill_vector(e0, {Scalar(1.0, 0.0), Scalar(0.0, 0.0)});
    fill_vector(e1, {Scalar(0.0, 0.0), Scalar(1.0, 0.0)});
    MultiVector samples(&graph, 2);
    samples.set_col(0, e0);
    samples.set_col(1, e1);

    std::vector<double> full_step, lp_step, full_spec, lp_spec, energy;
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd(cheby, X, samples, n_steps, dt_fs, n_kpm, full_step, energy);
    }
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd(cheby, X, samples, n_steps, dt_fs, n_kpm, lp_step, energy, n_compute);
    }
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                          full_spec, energy);
    }
    {
        Chebyshev<Scalar> cheby(H, emin, emax);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), samples, n_steps, dt_fs, n_kpm,
                          lp_spec, energy, 2, 0, n_compute);
    }

    ASSERT_EQ(full_step.size(), lp_step.size());
    ASSERT_EQ(full_spec.size(), lp_spec.size());
    double err_step = 0.0, err_spec = 0.0, scale = 0.0;
    for (size_t i = 0; i < full_step.size(); ++i) {
        scale = std::max(scale, std::abs(full_step[i]));
        err_step = std::max(err_step, std::abs(full_step[i] - lp_step[i]));
        err_spec = std::max(err_spec, std::abs(full_spec[i] - lp_spec[i]));
    }
    std::cout << "lp stepwise err " << err_step << ", lp spectral err " << err_spec
              << ", scale " << scale << std::endl;
    EXPECT_GT(scale, 1e-3);
    EXPECT_LT(err_step, 1e-6 * std::max(1.0, scale));
    EXPECT_LT(err_spec, 1e-6 * std::max(1.0, scale));
}

// Segmented anchoring is an operator-identity rearrangement of the same
// quadratic form, so small-segment and single-segment runs must agree per
// sample, to rounding.
TEST(MSDTest, SpectralRouteSegmentsMatchSingleSegment) {
    DistGraph graph = make_graph(MPI_COMM_WORLD, 2, true);
    Matrix H(&graph), V(&graph), S(&graph);
    const Dense2 h = {Scalar(-0.4, 0.0), Scalar(0.15, 0.2),
                      Scalar(0.15, -0.2), Scalar(0.35, 0.0)};
    const Dense2 s = {Scalar(1.2, 0.0), Scalar(0.1, 0.0),
                      Scalar(0.1, 0.0), Scalar(0.9, 0.0)};
    const Dense2 v = {Scalar(0.1, 0.0), Scalar(0.2, 0.1),
                      Scalar(0.2, -0.1), Scalar(-0.3, 0.0)};
    fill_matrix(H, {{h[0], h[1]}, {h[2], h[3]}});
    fill_matrix(S, {{s[0], s[1]}, {s[2], s[3]}});
    fill_matrix(V, {{v[0], v[1]}, {v[2], v[3]}});

    Vector sample(&graph);
    fill_vector(sample, {Scalar(0.8, 0.0), Scalar(0.0, 0.6)});

    // Long enough for several blocks; small block_substeps forces many.
    const int n_steps = 16, n_kpm = 24;
    const double dt_fs = 4.0;

    std::vector<double> ref, tiled, energy;
    {
        Chebyshev<Scalar> cheby(H, S, -1.2, 1.2);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), sample, n_steps, dt_fs, n_kpm,
                          ref, energy, 2, 20, 0, 1.0e9);
    }
    {
        Chebyshev<Scalar> cheby(H, S, -1.2, 1.2);
        corr_msd_spectral(cheby, VelocityOp<Scalar>(V), sample, n_steps, dt_fs, n_kpm,
                          tiled, energy, 2, 20, 0, 40.0);
    }
    ASSERT_EQ(ref.size(), tiled.size());
    double max_diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        scale = std::max(scale, std::abs(ref[i]));
        max_diff = std::max(max_diff, std::abs(ref[i] - tiled[i]));
    }
    std::cout << "segments max diff " << max_diff << ", scale " << scale << std::endl;
    EXPECT_GT(scale, 1e-6);
    EXPECT_LT(max_diff, 1e-10 * std::max(1.0, scale));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
