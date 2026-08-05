#include "../msd.hpp"

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
