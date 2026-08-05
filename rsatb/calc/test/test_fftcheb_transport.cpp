#include "../fftcheb_transport.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <complex>
#include <vector>

using namespace rsatb::calc;

namespace {

using Scalar = std::complex<double>;
using Matrix = BlockSpMat<Scalar>;
using Vector = DistVector<Scalar>;
using MultiVector = DistMultiVector<Scalar>;

DistGraph make_dense_graph(MPI_Comm comm, int n) {
    std::vector<int> block_sizes(n, 1);
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        adj[i].reserve(n);
        for (int j = 0; j < n; ++j) {
            adj[i].push_back(j);
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
        vec[local_row] = values[global_row];
    }
}

struct ToyProblem {
    DistGraph graph;
    Matrix H;
    Matrix Vx;
    Matrix Vy;
    Matrix Vz;
    std::unique_ptr<Matrix> S;
    Vector sample_a;
    Vector sample_b;

    ToyProblem(MPI_Comm comm, bool with_overlap)
        : graph(make_dense_graph(comm, 2)),
          H(&graph),
          Vx(&graph),
          Vy(&graph),
          Vz(&graph),
          S(with_overlap ? std::make_unique<Matrix>(&graph) : nullptr),
          sample_a(&graph),
          sample_b(&graph) {}
};

ToyProblem make_greenwood_problem(MPI_Comm comm, bool with_overlap) {
    ToyProblem problem(comm, with_overlap);

    fill_matrix(problem.H, {
        {Scalar(-0.35, 0.0), Scalar(0.18, -0.07)},
        {Scalar(0.18, 0.07), Scalar(0.42, 0.0)}
    });
    fill_matrix(problem.Vx, {
        {Scalar(0.0, 0.0), Scalar(1.0, 0.0)},
        {Scalar(1.0, 0.0), Scalar(0.0, 0.0)}
    });
    fill_matrix(problem.Vy, {
        {Scalar(0.2, 0.0), Scalar(0.0, 0.1)},
        {Scalar(0.0, -0.1), Scalar(-0.15, 0.0)}
    });
    fill_matrix(problem.Vz, {
        {Scalar(0.1, 0.0), Scalar(0.0, 0.0)},
        {Scalar(0.0, 0.0), Scalar(-0.1, 0.0)}
    });

    if (with_overlap) {
        fill_matrix(*problem.S, {
            {Scalar(1.15, 0.0), Scalar(0.05, 0.0)},
            {Scalar(0.05, 0.0), Scalar(1.05, 0.0)}
        });
    }

    fill_vector(problem.sample_a, {
        Scalar(1.0 / std::sqrt(2.0), 0.0),
        Scalar(0.0, 1.0 / std::sqrt(2.0))
    });
    fill_vector(problem.sample_b, {
        Scalar(0.3, -0.4),
        Scalar(-0.2, 0.5)
    });

    return problem;
}

ToyProblem make_bastin_problem(MPI_Comm comm, bool with_overlap) {
    ToyProblem problem(comm, with_overlap);

    fill_matrix(problem.H, {
        {Scalar(0.05, 0.0), Scalar(0.0, 0.8)},
        {Scalar(0.0, -0.8), Scalar(-0.1, 0.0)}
    });
    fill_matrix(problem.Vx, {
        {Scalar(0.0, 0.0), Scalar(0.9, 0.2)},
        {Scalar(0.9, -0.2), Scalar(0.0, 0.0)}
    });
    fill_matrix(problem.Vy, {
        {Scalar(0.4, 0.0), Scalar(0.0, -0.7)},
        {Scalar(0.0, 0.7), Scalar(-0.3, 0.0)}
    });
    fill_matrix(problem.Vz, {
        {Scalar(0.2, 0.0), Scalar(0.5, 0.0)},
        {Scalar(0.5, 0.0), Scalar(-0.25, 0.0)}
    });

    if (with_overlap) {
        fill_matrix(*problem.S, {
            {Scalar(1.0, 0.0), Scalar(0.08, 0.0)},
            {Scalar(0.08, 0.0), Scalar(1.2, 0.0)}
        });
    }

    fill_vector(problem.sample_a, {
        Scalar(0.6, 0.0),
        Scalar(0.0, 0.8)
    });
    fill_vector(problem.sample_b, {
        Scalar(-0.5, 0.1),
        Scalar(0.2, 0.7)
    });

    return problem;
}

template <typename ChebyT>
void build_reference_sequences(
    ChebyT& cheby,
    Matrix& Vx,
    Matrix& Vy,
    Matrix& Vz,
    const Vector& sample,
    int n_moments,
    std::array<std::vector<Vector>, 3>& left_seq,
    std::array<std::vector<Vector>, 3>& right_seq
) {
    for (int axis = 0; axis < 3; ++axis) {
        left_seq[axis].clear();
        right_seq[axis].clear();
        for (int m = 0; m < n_moments; ++m) {
            left_seq[axis].emplace_back(sample.graph);
            right_seq[axis].emplace_back(sample.graph);
        }
    }

    std::array<Vector, 3> right_init = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
    std::array<Vector, 3> left_axes = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
    const double velocity_scale = 1.0 / (2.0 * cheby.half_width);

    current(cheby, Vx, Vy, Vz, sample, right_init.data(), velocity_scale);
    for (int axis = 0; axis < 3; ++axis) {
        right_seq[axis][0].copy_from(right_init[axis]);
    }

    Vector left_base_prev(sample.graph);
    Vector left_base_curr(sample.graph);
    Vector left_base_next(sample.graph);
    left_base_prev.copy_from(sample);
    current_adjoint(cheby, Vx, Vy, Vz, left_base_prev, left_axes.data(), velocity_scale);
    for (int axis = 0; axis < 3; ++axis) {
        left_seq[axis][0].copy_from(left_axes[axis]);
    }

    MultiVector right_prev(sample.graph, 3);
    MultiVector right_curr(sample.graph, 3);
    MultiVector right_next(sample.graph, 3);
    set_multivector_cols(right_prev, right_init);

    if (n_moments > 1) {
        cheby.apply_op(right_prev, right_curr, ChebyMode::NORMAL);
        cheby.apply_op(left_base_prev, left_base_curr, ChebyMode::ADJOINT);
        current_adjoint(cheby, Vx, Vy, Vz, left_base_curr, left_axes.data(), velocity_scale);

        for (int axis = 0; axis < 3; ++axis) {
            right_seq[axis][1].copy_from(right_curr.get_col(axis));
            left_seq[axis][1].copy_from(left_axes[axis]);
        }
    }

    for (int moment = 2; moment < n_moments; ++moment) {
        cheby.apply_op(right_curr, right_next, ChebyMode::NORMAL);
        right_next.axpby(Scalar(-1.0), right_prev, Scalar(2.0));

        cheby.apply_op(left_base_curr, left_base_next, ChebyMode::ADJOINT);
        left_base_next.axpby(Scalar(-1.0), left_base_prev, Scalar(2.0));
        current_adjoint(cheby, Vx, Vy, Vz, left_base_next, left_axes.data(), velocity_scale);

        for (int axis = 0; axis < 3; ++axis) {
            right_seq[axis][moment].copy_from(right_next.get_col(axis));
            left_seq[axis][moment].copy_from(left_axes[axis]);
        }

        right_prev.swap(right_curr);
        right_curr.swap(right_next);
        left_base_prev.swap(left_base_curr);
        left_base_curr.swap(left_base_next);
    }
}

template <typename ChebyT>
std::vector<double> reference_spectrum(
    ChebyT& cheby,
    Matrix& Vx,
    Matrix& Vy,
    Matrix& Vz,
    const Vector& sample,
    int n_moments,
    DCFormulation formulation
) {
    std::array<std::vector<Vector>, 3> left_seq;
    std::array<std::vector<Vector>, 3> right_seq;
    build_reference_sequences(cheby, Vx, Vy, Vz, sample, n_moments, left_seq, right_seq);

    std::vector<double> jackson = jackson_kernel(n_moments);
    std::vector<double> eps_native(n_moments, 0.0);
    std::vector<double> gm(n_moments, 0.0);
    for (int m = 0; m < n_moments; ++m) {
        gm[m] = 2.0 * jackson[m] / (M_PI * (1.0 + (m == 0 ? 1.0 : 0.0)));
    }
    for (int k = 0; k < n_moments; ++k) {
        const double theta = M_PI * (static_cast<double>(k) + 0.5) / static_cast<double>(n_moments);
        eps_native[k] = std::cos(theta);
    }

    std::vector<double> result(static_cast<size_t>(n_moments) * 6, 0.0);
    std::vector<std::complex<double>> kernel_native(static_cast<size_t>(n_moments) * 6, std::complex<double>(0.0, 0.0));

    if (formulation == DCFormulation::Greenwood) {
        for (int axis = 0; axis < 3; ++axis) {
            const int comp = conductivity_component_index(axis, axis);
            for (int k = 0; k < n_moments; ++k) {
                const double eps = eps_native[k];
                const double denom = std::max(1e-14, 1.0 - eps * eps);
                double accum = 0.0;
                for (int m = 0; m < n_moments; ++m) {
                    const double Tm = std::cos(static_cast<double>(m) * std::acos(eps));
                    for (int n = 0; n < n_moments; ++n) {
                        const double Tn = std::cos(static_cast<double>(n) * std::acos(eps));
                        accum += (gm[m] * Tm * gm[n] * Tn / denom) *
                                 std::real(left_seq[axis][m].dot(right_seq[axis][n]));
                    }
                }
                kernel_native[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] = accum;
            }
        }

        for (int k = 0; k < n_moments; ++k) {
            const int src = n_moments - 1 - k;
            for (int comp = 0; comp < 6; ++comp) {
                result[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] =
                    kernel_native[static_cast<size_t>(src) * 6 + static_cast<size_t>(comp)].real();
            }
        }
        return result;
    }

    for (int k = 0; k < n_moments; ++k) {
        const double theta = M_PI * (static_cast<double>(k) + 0.5) / static_cast<double>(n_moments);
        const double eps = std::cos(theta);
        const double root = std::sin(theta);

        std::array<Vector, 3> left_phi1 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
        std::array<Vector, 3> left_phi2 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
        std::array<Vector, 3> left_phi3 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
        std::array<Vector, 3> right_phi1 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
        std::array<Vector, 3> right_phi2 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};
        std::array<Vector, 3> right_phi3 = {Vector(sample.graph), Vector(sample.graph), Vector(sample.graph)};

        for (int axis = 0; axis < 3; ++axis) {
            left_phi1[axis].set_constant(Scalar(0.0, 0.0));
            left_phi2[axis].set_constant(Scalar(0.0, 0.0));
            left_phi3[axis].set_constant(Scalar(0.0, 0.0));
            right_phi1[axis].set_constant(Scalar(0.0, 0.0));
            right_phi2[axis].set_constant(Scalar(0.0, 0.0));
            right_phi3[axis].set_constant(Scalar(0.0, 0.0));
        }

        for (int m = 0; m < n_moments; ++m) {
            const double coeff = gm[m];
            const double cosine = std::cos(static_cast<double>(m) * theta);
            const Scalar phase = std::exp(Scalar(0.0, static_cast<double>(m) * theta));
            const Scalar weighted = coeff * phase;
            const Scalar weighted_m = static_cast<double>(m) * weighted;
            const Scalar weighted_cos = coeff * cosine;

            for (int axis = 0; axis < 3; ++axis) {
                left_phi1[axis].axpy(weighted, left_seq[axis][m]);
                left_phi2[axis].axpy(weighted_m, left_seq[axis][m]);
                left_phi3[axis].axpy(weighted_cos, left_seq[axis][m]);

                right_phi1[axis].axpy(weighted, right_seq[axis][m]);
                right_phi2[axis].axpy(weighted_m, right_seq[axis][m]);
                right_phi3[axis].axpy(weighted_cos, right_seq[axis][m]);
            }
        }

        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                const int comp = conductivity_component_index(a, b);
                const Scalar value =
                    eps * (left_phi3[a].dot(right_phi1[b]) + left_phi1[a].dot(right_phi3[b])) +
                    Scalar(0.0, root) * (left_phi2[a].dot(right_phi3[b]) - left_phi3[a].dot(right_phi2[b]));
                kernel_native[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] = value;
            }
        }
    }

    std::vector<double> eps_sorted(n_moments, 0.0);
    for (int k = 0; k < n_moments; ++k) {
        eps_sorted[k] = eps_native[n_moments - 1 - k];
    }

    std::array<double, 6> running = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 6> prev = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < n_moments; ++k) {
        const int src = n_moments - 1 - k;
        std::array<double, 6> integrand = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int comp = 0; comp < 6; ++comp) {
            integrand[comp] = (std::complex<double>(0.0, 1.0) *
                               kernel_native[static_cast<size_t>(src) * 6 + static_cast<size_t>(comp)]).real();
        }
        if (k > 0) {
            const double de = eps_sorted[k] - eps_sorted[k - 1];
            for (int comp = 0; comp < 6; ++comp) {
                running[comp] += 0.5 * (prev[comp] + integrand[comp]) * de;
            }
        }
        for (int comp = 0; comp < 6; ++comp) {
            result[static_cast<size_t>(k) * 6 + static_cast<size_t>(comp)] = running[comp];
            prev[comp] = integrand[comp];
        }
    }

    return result;
}

std::vector<double> run_fftcheb(
    ToyProblem& problem,
    int n_moments,
    DCFormulation formulation,
    const std::vector<Vector>& samples,
    int partition_sections,
    size_t max_workspace_mb = 0
) {
    std::vector<double> sigma;
    std::vector<double> energy;

    MultiVector sample_mv(&problem.graph, static_cast<int>(samples.size()));
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        sample_mv.set_col(i, samples[i]);
    }

    if (problem.S) {
        Chebyshev<Scalar> cheby(problem.H, *problem.S, -1.0, 1.0);
        dc_spectrum_fftcheb(
            cheby, problem.Vx, problem.Vy, problem.Vz, sample_mv, n_moments,
            formulation, 1.0, partition_sections, max_workspace_mb, sigma, energy
        );
    } else {
        Chebyshev<Scalar> cheby(problem.H, -1.0, 1.0);
        dc_spectrum_fftcheb(
            cheby, problem.Vx, problem.Vy, problem.Vz, sample_mv, n_moments,
            formulation, 1.0, partition_sections, max_workspace_mb, sigma, energy
        );
    }

    return sigma;
}

std::vector<double> run_reference(
    ToyProblem& problem,
    int n_moments,
    DCFormulation formulation,
    const Vector& sample
) {
    if (problem.S) {
        Chebyshev<Scalar> cheby(problem.H, *problem.S, -1.0, 1.0);
        return reference_spectrum(cheby, problem.Vx, problem.Vy, problem.Vz, sample, n_moments, formulation);
    }

    Chebyshev<Scalar> cheby(problem.H, -1.0, 1.0);
    return reference_spectrum(cheby, problem.Vx, problem.Vy, problem.Vz, sample, n_moments, formulation);
}

void expect_close(const std::vector<double>& lhs, const std::vector<double>& rhs, double tol) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        EXPECT_NEAR(lhs[i], rhs[i], tol) << "mismatch at index " << i;
    }
}

std::vector<Scalar> explicit_dct_type3(const std::vector<Scalar>& input, int n, int batch) {
    std::vector<Scalar> out(static_cast<size_t>(n) * static_cast<size_t>(batch), Scalar(0.0, 0.0));
    for (int b = 0; b < batch; ++b) {
        const size_t base = static_cast<size_t>(b) * static_cast<size_t>(n);
        for (int k = 0; k < n; ++k) {
            Scalar value = input[base];
            for (int m = 1; m < n; ++m) {
                const double theta =
                    M_PI * static_cast<double>(m) * (static_cast<double>(k) + 0.5) / static_cast<double>(n);
                value += 2.0 * input[base + static_cast<size_t>(m)] * std::cos(theta);
            }
            out[base + static_cast<size_t>(k)] = value;
        }
    }
    return out;
}

void require_single_rank() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        GTEST_SKIP() << "FFTCheb transport reference tests run in serial mode only.";
    }
}

void require_two_ranks() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "Distributed FFTCheb transport regression currently runs with exactly 2 MPI ranks.";
    }
}

} // namespace

TEST(FFTChebTransportTest, GreenwoodMatchesMomentReferenceOrthogonal) {
    require_single_rank();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, false);
    const int n_moments = 12;
    const std::vector<Vector> samples = {problem.sample_a};

    const auto fftcheb = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, samples, 1);
    const auto reference = run_reference(problem, n_moments, DCFormulation::Greenwood, problem.sample_a);

    expect_close(fftcheb, reference, 1e-8);
}

TEST(FFTChebTransportTest, RowDctType3MatchesExplicitCosineSum) {
    require_single_rank();
    const int n = 8;
    const int batch = 2;
    std::vector<Scalar> input(static_cast<size_t>(n) * static_cast<size_t>(batch), Scalar(0.0, 0.0));
    for (int b = 0; b < batch; ++b) {
        const size_t base = static_cast<size_t>(b) * static_cast<size_t>(n);
        for (int m = 0; m < n; ++m) {
            input[base + static_cast<size_t>(m)] =
                Scalar(0.1 * (1 + b) * (m + 1), -0.05 * (2 + b) * (m + 1));
        }
    }

    RowFFTWorkspace workspace(n, batch);
    std::vector<Scalar> actual(static_cast<size_t>(n) * static_cast<size_t>(batch), Scalar(0.0, 0.0));
    workspace.dct_type3_batch(input.data(), actual.data());
    const auto expected = explicit_dct_type3(input, n, batch);

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i].real(), expected[i].real(), 1e-12) << "real mismatch at index " << i;
        EXPECT_NEAR(actual[i].imag(), expected[i].imag(), 1e-12) << "imag mismatch at index " << i;
    }
}

TEST(FFTChebTransportTest, BastinMatchesMomentReferenceOrthogonal) {
    require_single_rank();
    ToyProblem problem = make_bastin_problem(MPI_COMM_WORLD, false);
    const int n_moments = 12;
    const std::vector<Vector> samples = {problem.sample_a};

    const auto fftcheb = run_fftcheb(problem, n_moments, DCFormulation::Bastin, samples, 1);
    const auto reference = run_reference(problem, n_moments, DCFormulation::Bastin, problem.sample_a);

    expect_close(fftcheb, reference, 1e-8);
}

TEST(FFTChebTransportTest, GreenwoodMatchesMomentReferenceNonOrthogonal) {
    require_single_rank();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, true);
    const int n_moments = 10;
    const std::vector<Vector> samples = {problem.sample_a};

    const auto fftcheb = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, samples, 1);
    const auto reference = run_reference(problem, n_moments, DCFormulation::Greenwood, problem.sample_a);

    expect_close(fftcheb, reference, 1e-8);
}

TEST(FFTChebTransportTest, PartitionedMatchesUnpartitionedAndMultivectorAverage) {
    require_single_rank();
    ToyProblem problem = make_bastin_problem(MPI_COMM_WORLD, true);
    const int n_moments = 10;
    const std::vector<Vector> sample_a_only = {problem.sample_a};
    const std::vector<Vector> sample_b_only = {problem.sample_b};
    const std::vector<Vector> sample_batch = {problem.sample_a, problem.sample_b};

    const auto unpartitioned = run_fftcheb(problem, n_moments, DCFormulation::Bastin, sample_a_only, 1);
    const auto partitioned = run_fftcheb(problem, n_moments, DCFormulation::Bastin, sample_a_only, 2);
    expect_close(unpartitioned, partitioned, 1e-8);

    const auto sample_a = run_fftcheb(problem, n_moments, DCFormulation::Bastin, sample_a_only, 1);
    const auto sample_b = run_fftcheb(problem, n_moments, DCFormulation::Bastin, sample_b_only, 1);
    const auto batched = run_fftcheb(problem, n_moments, DCFormulation::Bastin, sample_batch, 1);

    std::vector<double> averaged(sample_a.size(), 0.0);
    for (size_t i = 0; i < averaged.size(); ++i) {
        averaged[i] = 0.5 * (sample_a[i] + sample_b[i]);
    }
    expect_close(averaged, batched, 1e-8);
}

TEST(FFTChebTransportTest, GreenwoodMultisampleMatchesAverageNonOrthogonal) {
    require_single_rank();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, true);
    const int n_moments = 10;
    const std::vector<Vector> sample_a_only = {problem.sample_a};
    const std::vector<Vector> sample_b_only = {problem.sample_b};
    const std::vector<Vector> sample_batch = {problem.sample_a, problem.sample_b};

    const auto sample_a = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_a_only, 1);
    const auto sample_b = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_b_only, 1);
    const auto batched = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_batch, 1);

    std::vector<double> averaged(sample_a.size(), 0.0);
    for (size_t i = 0; i < averaged.size(); ++i) {
        averaged[i] = 0.5 * (sample_a[i] + sample_b[i]);
    }
    expect_close(averaged, batched, 1e-8);
}

TEST(FFTChebTransportTest, SampleBlockingMatchesFullBatchUnderWorkspaceCap) {
    require_single_rank();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, false);
    const int n_moments = 1024;
    std::vector<Vector> many_samples;
    many_samples.reserve(8);
    for (int i = 0; i < 8; ++i) {
        many_samples.push_back((i % 2 == 0) ? problem.sample_a : problem.sample_b);
    }

    const auto full_batch = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, many_samples, 1, 0);
    const auto blocked = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, many_samples, 1, 1);

    expect_close(full_batch, blocked, 1e-8);
}

TEST(FFTChebTransportTest, DistributedGreenwoodMatchesMomentReferenceTwoRanks) {
    require_two_ranks();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, true);
    const int n_moments = 10;
    const std::vector<Vector> samples = {problem.sample_a};

    const auto fftcheb = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, samples, 1);
    const auto reference = run_reference(problem, n_moments, DCFormulation::Greenwood, problem.sample_a);

    expect_close(fftcheb, reference, 1e-8);
}

TEST(FFTChebTransportTest, DistributedBastinMatchesReferenceAndPartitioningTwoRanks) {
    require_two_ranks();
    ToyProblem problem = make_bastin_problem(MPI_COMM_WORLD, false);
    const int n_moments = 10;
    const std::vector<Vector> samples = {problem.sample_a};

    const auto unpartitioned = run_fftcheb(problem, n_moments, DCFormulation::Bastin, samples, 1);
    const auto partitioned = run_fftcheb(problem, n_moments, DCFormulation::Bastin, samples, 2);
    const auto reference = run_reference(problem, n_moments, DCFormulation::Bastin, problem.sample_a);

    expect_close(unpartitioned, reference, 1e-8);
    expect_close(partitioned, reference, 1e-8);
}

TEST(FFTChebTransportTest, DistributedGreenwoodMultisampleMatchesAverageTwoRanks) {
    require_two_ranks();
    ToyProblem problem = make_greenwood_problem(MPI_COMM_WORLD, true);
    const int n_moments = 10;
    const std::vector<Vector> sample_a_only = {problem.sample_a};
    const std::vector<Vector> sample_b_only = {problem.sample_b};
    const std::vector<Vector> sample_batch = {problem.sample_a, problem.sample_b};

    const auto sample_a = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_a_only, 1);
    const auto sample_b = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_b_only, 1);
    const auto batched = run_fftcheb(problem, n_moments, DCFormulation::Greenwood, sample_batch, 1);

    std::vector<double> averaged(sample_a.size(), 0.0);
    for (size_t i = 0; i < averaged.size(); ++i) {
        averaged[i] = 0.5 * (sample_a[i] + sample_b[i]);
    }
    expect_close(averaged, batched, 1e-8);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
