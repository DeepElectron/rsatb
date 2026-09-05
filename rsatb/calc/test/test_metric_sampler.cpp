// Correctness and benchmark for the metric-correct sampler
// (doc/design_stochastic_ldos.md section 4.3).
//
// The certificate hierarchy, cheapest first:
//   1. Deterministic bias: ||W q(W)^2 xi - xi|| / ||xi|| <= O(eps) per vector,
//      no sampling noise. This is what the covariance identity reduces to.
//   2. Statistical covariance: E|u^H c0|^2 -> u^H S^{-1} u at rate 1/sqrt(S),
//      with the CG solve of S x = u as the exact reference. The naive sampler
//      c0 = xi converges to u^H u instead -- the bias the correction removes.
//   3. Timing / SpMV counts across the three Zt regimes of the design table.

#include "../metric_sampler.hpp"
#include "../isqrtm.hpp"
#include "../solver.hpp"
#include "vbcsr/core/atomic/io.hpp"
#include "../../model/lcao/op2c_engine.hpp"
#include "../../test_data_path.hpp"
#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/atomic/neighbourlist.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace vbcsr;
using rsatb::calc::MetricSampler;
using rsatb::calc::MetricSamplerOptions;

namespace {

// 1D chain S = tridiag(-1, 2 + shift, -1): SPD with lambda in
// [shift, 4 + shift]; kappa ~ (4 + shift) / shift is set by the shift.
struct ChainOverlap {
    DistGraph graph;
    std::unique_ptr<BlockSpMat<double>> S_ptr;

    ChainOverlap(int n, double shift) : graph(MPI_COMM_WORLD) {
        std::vector<int> block_sizes(n, 1);
        std::vector<std::vector<int>> adj(n);
        for (int i = 0; i < n; ++i) {
            adj[i].push_back(i);
            if (i > 0) adj[i].push_back(i - 1);
            if (i + 1 < n) adj[i].push_back(i + 1);
        }
        graph.construct_serial(n, block_sizes, adj);
        S_ptr.reset(new BlockSpMat<double>(&graph));
        BlockSpMat<double>& S = *S_ptr;
        const int n_owned = static_cast<int>(graph.owned_global_indices.size());
        for (int i = 0; i < n_owned; ++i) {
            const int gid = graph.owned_global_indices[i];
            double val = 2.0 + shift;
            S.add_block(gid, gid, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            val = -1.0;
            if (gid > 0) S.add_block(gid, gid - 1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            if (gid + 1 < n) S.add_block(gid, gid + 1, &val, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        S.assemble();
    }

    BlockSpMat<double>& S() { return *S_ptr; }
};

enum class ZtRegime { kIdentity, kPurified, kLoose, kGraphSeed };

const char* RegimeName(ZtRegime r) {
    switch (r) {
        case ZtRegime::kIdentity: return "identity  ";
        case ZtRegime::kPurified: return "purified ";
        case ZtRegime::kLoose: return "loose NS ";
        case ZtRegime::kGraphSeed: return "graph-seed";
    }
    return "?";
}

BlockSpMat<double> MakeZt(BlockSpMat<double>& S, ZtRegime regime) {
    if (regime == ZtRegime::kIdentity) {
        // No metric factor at all: W = S. Baseline showing what any Zt buys.
        // Built explicitly -- duplicate(true) copies S's values, so shifting a
        // duplicate would give S + I, not I.
        BlockSpMat<double> Zt(S.graph);
        const int n_owned = static_cast<int>(S.graph->owned_global_indices.size());
        for (int i = 0; i < n_owned; ++i) {
            const int gid = S.graph->owned_global_indices[i];
            const int bs = S.graph->block_sizes[i];
            std::vector<double> eye(static_cast<size_t>(bs) * bs, 0.0);
            for (int d = 0; d < bs; ++d) eye[static_cast<size_t>(d) * bs + d] = 1.0;
            Zt.add_block(gid, gid, eye.data(), bs, bs, AssemblyMode::INSERT,
                         MatrixLayout::RowMajor);
        }
        Zt.assemble();
        return Zt;
    }
    BlockSpMat<double> Zt = S.duplicate(true);
    switch (regime) {
        case ZtRegime::kIdentity:
            break;  // handled above
        case ZtRegime::kPurified:
            rsatb::calc::inverse_sqrt(S, &Zt, {1e-10, 1e-11, 50});
            break;
        case ZtRegime::kLoose:
            rsatb::calc::inverse_sqrt(S, &Zt, {1e-5, 1e-6, 50});
            break;
        case ZtRegime::kGraphSeed:
            // The subgraph dense seed alone, exactly the "we are not sure how
            // good it is" object: maxit = 0 skips every Newton-Schulz sweep.
            rsatb::calc::inverse_sqrt(S, &Zt, {1e-10, 1e-10, 0});
            break;
    }
    return Zt;
}

// The real diamond overlap at Gamma, built by the op2c two-center integral
// engine -- the same path production uses (LcaoModel::BuildOverlapImages).
// A Gram matrix of the actual basis: SPD by construction, unlike the
// R-truncated S(R) stored in the DeePTB files.
struct DiamondIntegralsOverlap {
    std::unique_ptr<vbcsr::atomic::AtomicData> ad;
    std::unique_ptr<vbcsr::atomic::ImageContainer<double>> img;
    std::unique_ptr<BlockSpMat<std::complex<double>>> S_gamma;
    std::unique_ptr<BlockSpMat<double>> S_real;

    explicit DiamondIntegralsOverlap(int repeat = 4) {
        const std::string data = rsatb::test::data_dir() + "diamond/";
        vbcsr::atomic::Structure st = vbcsr::atomic::ReadPoscar(data + "conf.vasp");
        st.supercell = {{repeat, repeat, repeat}};
        st = st.BuildSupercell();

        // 2s2p1d orbital, r_cut = 7 au; the overlap pattern reaches 2 r_cut.
        const double rcut_ang = 7.0 * 0.529177210903;
        vbcsr::atomic::NeighborList nl;
        nl.build(st.pos, st.cell, {true, true, true}, 2.0 * rcut_ang + 1e-6);

        const int n_atom = st.n_atom;
        std::vector<int> atom_index(n_atom), atom_type(n_atom, 0);
        for (int i = 0; i < n_atom; ++i) atom_index[i] = i;
        const int type_norb[1] = {13};

        // No explicit (i, i, 0) self-edges: build_overlap computes the
        // on-site block in its own vertex loop, and a duplicate edge would
        // double the diagonal (found the hard way: diag(S) came out exactly 2).
        std::vector<int> edge_index, edge_shift;
        for (int i = 0; i < n_atom; ++i) {
            for (const auto& nb : nl.neighbors[i]) {
                edge_index.push_back(i); edge_index.push_back(nb.index);
                edge_shift.push_back(nb.rx); edge_shift.push_back(nb.ry);
                edge_shift.push_back(nb.rz);
            }
        }
        const size_t n_edge = edge_index.size() / 2;

        ad.reset(new vbcsr::atomic::AtomicData(
            n_atom, n_atom, /*atom_offset=*/0, n_edge, n_edge,
            atom_index.data(), atom_type.data(), edge_index.data(), type_norb,
            edge_shift.data(), st.cell.data(), st.pos.data(), MPI_COMM_WORLD,
            {true, true, true}));

        rsatb::model::lcao::Op2cEngine op(MPI_COMM_WORLD);
        op.init(1, 1, false, data, {"C_gga_7au_100Ry_2s2p1d.orb"}, "", {},
                "metric_sampler_opconst.log");
        img = op.build_overlap(*ad);
        S_gamma.reset(img->sample_k({0.0, 0.0, 0.0},
                                    vbcsr::atomic::PhaseConvention::R_AND_POSITION));

        // At Gamma the overlap is real; peel the (numerically zero) imaginary
        // part into a real matrix so the double-typed benchmark applies.
        S_real.reset(new BlockSpMat<double>(S_gamma->graph));
        const auto& row_ptr = S_gamma->row_ptr();
        const auto& col_ind = S_gamma->col_ind();
        const int n_rows = static_cast<int>(row_ptr.size()) - 1;
        double max_imag = 0.0;
        std::vector<double> buf;
        for (int i = 0; i < n_rows; ++i) {
            const int gr = S_gamma->graph->get_global_index(i);
            const int r = S_gamma->graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int lc = col_ind[k];
                const int gc = S_gamma->graph->get_global_index(lc);
                const int c = S_gamma->graph->block_sizes[lc];
                const std::complex<double>* blk = S_gamma->block_data(k);
                buf.resize(static_cast<size_t>(r) * c);
                for (size_t t = 0; t < buf.size(); ++t) {
                    buf[t] = blk[t].real();
                    max_imag = std::max(max_imag, std::abs(blk[t].imag()));
                }
                S_real->add_block(gr, gc, buf.data(), r, c,
                                  AssemblyMode::INSERT, MatrixLayout::RowMajor);
            }
        }
        S_real->assemble();
        // Record the diagonal and trace: settles orbital normalization and
        // anchors the measured spectrum against trace(S)/N = mean eigenvalue.
        double dmin = 1e300, dmax = -1e300, trace = 0.0;
        for (int i = 0; i < n_rows; ++i) {
            const int gr = S_gamma->graph->get_global_index(i);
            const int r = S_gamma->graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (S_gamma->graph->get_global_index(col_ind[k]) != gr) continue;
                const std::complex<double>* blk = S_gamma->block_data(k);
                for (int d = 0; d < r; ++d) {
                    const double v = blk[static_cast<size_t>(d) * r + d].real();
                    dmin = std::min(dmin, v); dmax = std::max(dmax, v);
                    trace += v;
                }
            }
        }
        printf("  diag(S) in [%.6f, %.6f], trace/N = %.6f\n", dmin, dmax,
               trace / S_gamma->graph->global_scalar_rows());
        fflush(stdout);
        if (max_imag > 1e-10) {
            throw std::runtime_error("Gamma overlap has imaginary parts: " +
                                     std::to_string(max_imag));
        }
    }

    BlockSpMat<double>& S() { return *S_real; }
    DistGraph& graph() { return *S_real->graph; }
};

void FillRademacher(DistVector<double>& v, std::mt19937& gen) {
    std::bernoulli_distribution coin(0.5);
    for (int i = 0; i < v.local_size; ++i) v[i] = coin(gen) ? 1.0 : -1.0;
}

// Reference u^H S^{-1} u via tightly converged CG.
double QuadFormSinv(BlockSpMat<double>& S, DistVector<double>& u) {
    rsatb::calc::LinearSolver<double> solver(1e-12, 10000, false);
    DistVector<double> x(u.graph);
    x.scale(0.0);
    solver.solve(S, u, x);
    return u.dot(x);
}

struct RegimeReport {
    double w_min, w_max, e_norm;
    bool binomial;
    int degree;
    double residual;
    double cov_rel_err;
    double naive_rel_err;
    double us_per_sample;
    double spmv_per_sample;
};

RegimeReport RunRegime(BlockSpMat<double>& S, DistGraph& graph, ZtRegime regime,
                       int n_samples) {
    BlockSpMat<double> Zt = MakeZt(S, regime);
    MetricSamplerOptions opt;
    opt.eps = 1e-8;
    opt.interval_pad = 0.10;
    MetricSampler<double> sampler(S, Zt, opt);

    RegimeReport rep{};
    rep.w_min = sampler.w_min();
    rep.w_max = sampler.w_max();
    rep.e_norm = sampler.residual_norm();
    rep.binomial = (sampler.degree() <= 2);
    rep.degree = sampler.degree();

    std::mt19937 gen(4242 + graph.rank);

    // 1. Deterministic bias certificate, averaged over a few vectors.
    double res = 0.0;
    const int n_res = 5;
    for (int t = 0; t < n_res; ++t) {
        DistVector<double> xi(&graph);
        FillRademacher(xi, gen);
        res += sampler.Residual(xi);
    }
    rep.residual = res / n_res;

    // 2. Covariance against the CG reference, one fixed probe u.
    DistVector<double> u(&graph);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < u.local_size; ++i) u[i] = nd(gen);
    const double exact = QuadFormSinv(S, u);
    const double naive = u.dot(u);  // what E|u^H c0|^2 becomes for c0 = xi

    const auto t0 = std::chrono::steady_clock::now();
    double acc = 0.0;
    DistVector<double> xi(&graph), c0(&graph);
    for (int s = 0; s < n_samples; ++s) {
        FillRademacher(xi, gen);
        sampler.Sample(xi, c0);
        const double proj = u.dot(c0);
        acc += proj * proj;
    }
    const auto t1 = std::chrono::steady_clock::now();
    acc /= n_samples;

    rep.cov_rel_err = std::abs(acc - exact) / exact;
    rep.naive_rel_err = std::abs(naive - exact) / exact;
    rep.us_per_sample =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / n_samples;
    rep.spmv_per_sample = 3.0 * (sampler.degree() + 1) + 1.0;  // W applies + Zt
    return rep;
}

void PrintReport(const char* label, ZtRegime regime, const RegimeReport& r,
                 int n_samples) {
    printf("  %-10s | W in [%.3e, %.3e]  ||E||=%.2e | %s deg=%-4d | "
           "bias=%.2e | cov err=%.2e (naive %.2e, %d smp) | "
           "%.0f us/smp, %.0f SpMV/smp\n",
           label, r.w_min, r.w_max, r.e_norm,
           r.binomial ? "binomial " : "chebyshev", r.degree, r.residual,
           r.cov_rel_err, r.naive_rel_err, n_samples, r.us_per_sample,
           r.spmv_per_sample);
    fflush(stdout);
}

class MetricSamplerTest : public ::testing::Test {};

}  // namespace

// Well-conditioned S (kappa ~ 5): every regime must hit the certificate.
TEST_F(MetricSamplerTest, WellConditionedAllRegimes) {
    ChainOverlap chain(2000, 1.0);
    const int n_samples = 20000;
    printf("S: 1D chain n=2000, shift=1.0 (kappa~5)\n");
    for (ZtRegime regime :
         {ZtRegime::kPurified, ZtRegime::kLoose, ZtRegime::kGraphSeed}) {
        RegimeReport r = RunRegime(chain.S(), chain.graph, regime, n_samples);
        PrintReport(RegimeName(regime), regime, r, n_samples);

        // Deterministic: applicator honest to ~eps (allow slack for the
        // covariance being q^2, i.e. 2x the single-apply error).
        EXPECT_LT(r.residual, 1e-6) << RegimeName(regime);
        // Statistical: |u^H c0|^2 has relative std ~ sqrt(2/S) ~ 1%; 5 sigma.
        EXPECT_LT(r.cov_rel_err, 5.0 * std::sqrt(2.0 / n_samples))
            << RegimeName(regime);
        // And the naive sampler must be visibly wrong, or this test proves
        // nothing about the metric.
        EXPECT_GT(r.naive_rel_err, 0.2);
    }
}

// Ill-conditioned S (kappa ~ 200, the DZP-like regime): the graph seed must
// switch to the measured-interval Chebyshev applicator and still certify.
TEST_F(MetricSamplerTest, IllConditionedGraphSeedUsesChebyshev) {
    ChainOverlap chain(2000, 0.02);
    const int n_samples = 20000;
    printf("S: 1D chain n=2000, shift=0.02 (kappa~200)\n");

    RegimeReport seed =
        RunRegime(chain.S(), chain.graph, ZtRegime::kGraphSeed, n_samples);
    PrintReport(RegimeName(ZtRegime::kGraphSeed), ZtRegime::kGraphSeed, seed,
                n_samples);
    EXPECT_FALSE(seed.binomial) << "O(1) ||E|| must reject the binomial series";
    EXPECT_LT(seed.residual, 1e-6);
    EXPECT_LT(seed.cov_rel_err, 5.0 * std::sqrt(2.0 / n_samples));

    RegimeReport pure =
        RunRegime(chain.S(), chain.graph, ZtRegime::kPurified, n_samples);
    PrintReport(RegimeName(ZtRegime::kPurified), ZtRegime::kPurified, pure,
                n_samples);
    EXPECT_LT(pure.residual, 1e-6);
    EXPECT_LT(pure.cov_rel_err, 5.0 * std::sqrt(2.0 / n_samples));
}

// A starved applicator must be exposed, not silently wrong: with the degree
// capped far below what kappa(W) needs, the certificate has to fail loudly.
// This is the guard against trusting q(W) on a spectrum it does not cover.
TEST_F(MetricSamplerTest, UnderResolvedApplicatorFailsTheCertificate) {
    ChainOverlap chain(500, 0.02);
    BlockSpMat<double> Zt = MakeZt(chain.S(), ZtRegime::kGraphSeed);
    MetricSamplerOptions opt;
    opt.eps = 1e-8;
    opt.max_degree = 1;  // nowhere near enough for this kappa(W)
    MetricSampler<double> sampler(chain.S(), Zt, opt);

    std::mt19937 gen(7);
    DistVector<double> xi(&chain.graph);
    FillRademacher(xi, gen);
    const double res = sampler.Residual(xi);
    printf("  degree-capped applicator (kappa(W) = %.1f): certificate = %.3e\n",
           sampler.w_max() / sampler.w_min(), res);
    EXPECT_GT(res, 1e-3) << "certificate failed to expose a starved applicator";
}

// The sampler guard must reject an overlap with no S^{-1/2}. Synthetic:
// tridiag(-1, 1.5, -1) has spectrum [-0.5, 3.5] -- indefinite.
TEST_F(MetricSamplerTest, IndefiniteOverlapIsRejected) {
    ChainOverlap chain(500, -0.5);
    BlockSpMat<double> Zt = MakeZt(chain.S(), ZtRegime::kGraphSeed);
    EXPECT_THROW({ MetricSampler<double> bad(chain.S(), Zt); },
                 std::runtime_error)
        << "indefinite S must be rejected, not sampled";
}

// The real overlap: diamond sc444 (128 atoms x 13 orbitals, N = 1664), built
// by the op2c two-center integral engine -- the matrix the LDOS path will
// actually use. Every regime of the design table, including the no-factor
// identity baseline whose W is S itself (measuring the true kappa(S) of the
// basis).
TEST_F(MetricSamplerTest, DiamondSc444IntegralsAllRegimes) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    DiamondIntegralsOverlap diamond;
    printf("S: diamond sc444, op2c two-center integrals, N = %d\n",
           diamond.graph().global_scalar_rows());
    fflush(stdout);

    const int kMaxDegree = MetricSamplerOptions{}.max_degree;
    struct Plan { ZtRegime regime; int n_samples; };
    const Plan plans[] = {{ZtRegime::kIdentity, 1000},
                          {ZtRegime::kGraphSeed, 2000},
                          {ZtRegime::kLoose, 5000},
                          {ZtRegime::kPurified, 5000}};
    for (const Plan& plan : plans) {
        RegimeReport r =
            RunRegime(diamond.S(), diamond.graph(), plan.regime, plan.n_samples);
        PrintReport(RegimeName(plan.regime), plan.regime, r, plan.n_samples);
        const bool capped = (!r.binomial && r.degree >= kMaxDegree);
        if (plan.regime == ZtRegime::kIdentity && (capped || r.residual > 1e-6)) {
            // The no-factor baseline is allowed to be unusable -- that is a
            // measurement (raw S too spread for a direct x^{-1/2} fit within
            // the Lanczos/degree budget), and the certificate reporting it is
            // the designed behavior. The Zt regimes below must still pass.
            printf("    (identity baseline unusable at this kappa(S): %s -- "
                   "certificate %.2e; the Zt regimes below are the point)\n",
                   capped ? "degree capped" : "fit not converged", r.residual);
            fflush(stdout);
            continue;
        }
        EXPECT_LT(r.residual, 1e-6) << RegimeName(plan.regime);
        EXPECT_LT(r.cov_rel_err, 5.0 * std::sqrt(2.0 / plan.n_samples))
            << RegimeName(plan.regime);
        EXPECT_GT(r.naive_rel_err, 0.1) << RegimeName(plan.regime);
    }
}

// The graph seed with GENUINE approximation error: diamond sc888 (1024
// atoms, N = 13312, 28.6 A cell). Each atom's pattern-subgraph covers only
// ~300 atoms, so the subgraph dense S^{-1/2} is a real approximation -- the
// regime of production use, where its quality was previously unquantified.
// Heavy: 1024 dense ~3900-dim subgraph problems in the seed; run with many
// OpenMP threads (one problem per thread, ~0.45 GB each).
TEST_F(MetricSamplerTest, DiamondSc888GraphSeedHeavy) {
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1) GTEST_SKIP() << "serial fixture";

    DiamondIntegralsOverlap diamond(8);
    printf("S: diamond sc888, op2c two-center integrals, N = %d\n",
           diamond.graph().global_scalar_rows());
    fflush(stdout);

    const int n_samples = 500;
    RegimeReport r = RunRegime(diamond.S(), diamond.graph(),
                               ZtRegime::kGraphSeed, n_samples);
    PrintReport(RegimeName(ZtRegime::kGraphSeed), ZtRegime::kGraphSeed, r,
                n_samples);
    // The point of this test: the seed must NOT be exact here...
    EXPECT_GT(r.e_norm, 1e-6)
        << "subgraph == system? then this test lost its purpose";
    // ...and the sampler must still certify on top of it.
    EXPECT_LT(r.residual, 1e-6);
    EXPECT_LT(r.cov_rel_err, 5.0 * std::sqrt(2.0 / n_samples));
    EXPECT_GT(r.naive_rel_err, 0.1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
