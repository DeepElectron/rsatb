#ifndef RSATB_CORE_RSATB_HPP
#define RSATB_CORE_RSATB_HPP

#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "setup/operator_assembly.hpp"
#include "model/lcao/lcao_model.hpp"
#include "model/sktb/sktb_model.hpp"
#include "model/tbfile/tb_file_model.hpp"
#include "vbcsr/core/atomic/io.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "calc/eigen.hpp"
#include "calc/dos.hpp"
#include "calc/ac.hpp"
#include "calc/fftcheb_transport.hpp"
#include "calc/msd.hpp"
#include "calc/fermi_dirac.hpp"
#include "calc/isqrtm.hpp"
#include "calc/preconditioner.hpp"
#include "calc/solver.hpp" 
#include "calc/cheby.hpp"
#include "calc/kpm.hpp"
#include "calc/fft.hpp"
#include "calc/observables.hpp"

#include <memory>
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <complex>
#include <limits>
#include <iostream>
#include <map>
#include <type_traits>
#include <mpi.h>

using namespace vbcsr;
using namespace vbcsr::atomic;

namespace rsatb {

using namespace rsatb::setup;
using namespace rsatb::calc;


class RSATB {
public:
    using Scalar = std::complex<double>;
    using Real = double;
    using Matrix = BlockSpMat<Scalar>;
    using Vector = DistVector<Scalar>;
    using MultiVector = DistMultiVector<Scalar>;

    /// A pre-partitioned atomic graph supplied directly by the caller.
    ///
    /// Use this when the neighbour list already exists -- a hand-built model
    /// cell, or a graph produced by another tool. `type_norb` is deliberately
    /// absent: it comes from the model's basis, so there is one source of
    /// truth for the block structure.
    ///
    /// `pos` and `edge_shift_vec` must be mutually consistent: every bond is
    /// read as r_j + R - r_i, so a shift has to be expressed relative to the
    /// positions given here. That is the same convention the (Structure, Model)
    /// constructor follows when it derives the graph itself, so a graph built
    /// either way means the same thing.
    struct AtomicGraphArrays {
        size_t n_atom = 0;
        size_t N_atom = 0;
        size_t atom_offset = 0;
        size_t n_edge = 0;
        size_t N_edge = 0;
        const int* atom_index = nullptr;
        const int* atom_type = nullptr;
        const int* edge_index = nullptr;
        const int* edge_shift_vec = nullptr;
        const double* cell = nullptr;
        const double* pos = nullptr;
    };

    // vbcsr no longer selects a kernel at compile time (BlockSpMat::KernelType is
    // gone); the vendor backend is chosen per matrix at runtime, so report it from
    // a live operator when one exists.
    std::string get_kernel_info() const {
        if (operators_) {
            for (const auto& kp : operators_->local_kpoints()) {
                if (const auto* h = operators_->hamiltonian_k(kp.first)) {
                    return h->vendor_backend_name();
                }
            }
        }
        return "uninitialized";
    }

    /// Builds a ready-to-compute RSATB from a pre-partitioned graph and a model.
    ///
    /// `model` must outlive this RSATB.
    RSATB(const AtomicGraphArrays& graph,
          const model::Model& model,
          const model::SetupOptions& options = model::SetupOptions{},
          int npar = 1,
          MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar) {
        init_comms();
        begin_profile();

        const model::BasisSpec& basis = model.basis();
        RequireConsistentGraphPartition(graph);
        {
            RSATB_PHASE("setup/atomic_graph");
            atomic_data_ = std::make_unique<AtomicData>(
                graph.n_atom, graph.N_atom, graph.atom_offset, graph.n_edge, graph.N_edge,
                graph.atom_index, graph.atom_type, graph.edge_index, basis.type_norb.data(),
                graph.edge_shift_vec, graph.cell, graph.pos, sub_comm_);
        }
        RequireEveryRankOwnsAtoms();
        init_components();

        setup(model, options);
    }

    /// Builds a ready-to-compute RSATB from a structure and a model.
    ///
    /// The model reports the basis it works in, so the atomic graph is derived
    /// rather than described: no orbital counts and no cutoff radii are passed
    /// in. `model` must outlive this RSATB, because velocity and position
    /// operators are built lazily through it.
    ///
    /// Args:
    ///   structure: Atomic geometry.
    ///   model: One of the model families; supplies the basis and operators.
    ///   options: k-points, sampling, bounds, and preconditioner settings.
    ///   npar: Number of k-point-parallel sub-communicators.
    RSATB(const vbcsr::atomic::Structure& structure,
          const model::Model& model,
          const model::SetupOptions& options = model::SetupOptions{},
          int npar = 1,
          MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar) {
        init_comms();
        // Before the graph is built, not after: the neighbour list and the
        // distributed partition are setup work, and starting the clock in
        // setup() below would leave them permanently outside the table.
        begin_profile();

        const model::BasisSpec& basis = model.basis();
        const int primitive_n_atom = structure.n_atom;
        const auto structure_supercell = structure.supercell;
        const auto expanded_structure = structure.BuildSupercell();
        basis.RequireCovers(expanded_structure.TypeSymbols(), model.name() + " setup");

        {
            RSATB_PHASE("setup/atomic_graph");
            atomic_data_ = std::unique_ptr<AtomicData>(AtomicData::from_points(
                expanded_structure.pos, expanded_structure.z, expanded_structure.cell,
                expanded_structure.pbc,
                basis.cutoff_radius, basis.type_norb, sub_comm_));
            atomic_data_->primitive_n_atom = primitive_n_atom;
            atomic_data_->structure_supercell = structure_supercell;
        }
        RequireEveryRankOwnsAtoms();
        init_components();

        setup(model, options);
    }

    /// Sets up the operators for a model against the existing atomic data.
    void setup(const model::Model& model, const model::SetupOptions& options = model::SetupOptions{}) {
        // A standalone setup() starts its own clock; one reached through the
        // constructor keeps the clock the constructor started, so the graph
        // build stays in the same table as the operators built from it.
        if (!profile_open_) begin_profile();
        // Ctrl-C during setup lands at the next Newton-Schulz step; the seed
        // itself finishes first, since its batch loop lives in vbcsr.
        util::InterruptGuard interrupt_guard;

        const auto normalized_options = normalize_setup_options(options);
        operators_ = setup::BuildOperatorSet(
            *atomic_data_,
            sub_comm_,
            util::Log::enabled(util::LogLevel::kDebug),
            model,
            normalized_options,
            make_operator_runtime_context());
        {
            RSATB_PHASE("setup/sample_vectors");
            sampling_.initialize(*atomic_data_, normalized_options.nsamples);
        }
        // Retained rather than re-derived: the real-space observables need the
        // radials themselves, and the model is not kept past setup.
        orbital_files_ = model.RealSpaceOrbitalFiles();

        // Reported against wall time since the clock started, so the table shows
        // what the named phases did not account for. An unattributed remainder
        // is the thing worth chasing next, and hiding it would defeat the point.
        util::Profile::report(0.0, world_comm_);
        profile_open_ = false;
    }

    /// Rank of this process, or 0 when MPI is not initialised.
    static int world_rank_of(MPI_Comm comm) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized || comm == MPI_COMM_NULL) return 0;
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        return rank;
    }

    ~RSATB() {
        release_resources();
    }

    void test_solve() {
        LinearSolver<Scalar> solver(1e-8, 1000, util::Log::enabled(util::LogLevel::kDebug));
        auto& ops = require_operators();
        for (const auto& kp : ops.local_kpoints()) {
            int k_idx = kp.first;
            auto& state = ops.kpoint_state(k_idx);
            auto* P = state.preconditioner.get();
            auto* Sk = state.overlap.get();
            if (!P || !Sk || !sampling_.sample_vec) {
                continue;
            }
            if (sampling_.nsamples == 1) {
                Vector y = sampling_.sample_vec->duplicate();
                solver.solve(
                    *Sk, *sampling_.sample_vec, y, *P, P->get_type()
                );
            } else {
                continue;
            }
        }
    }

    void band_pardiso(double mu, int nband, std::vector<double>& band_out) {
        // use shift invert method to compute the band structure  around some preset energy mu.
        
    }

    // The observables live in calc/observables.hpp as free functions over an
    // ObservableContext. These forwarders keep the public surface unchanged.

    /// Density of states at energy resolution `de_ev` (eV); the Chebyshev
    /// step and moment counts are derived from the spectral span.
    void compute_dos(double de_ev, std::vector<double>& dos_out, std::vector<double>& omega_out,
                     const std::string& method = "kpm") {
        calc::ComputeDos(observable_context(), de_ev, dos_out, omega_out, method);
    }

    /// Returns the unit cell as nine row-major values, Angstrom.
    std::vector<double> cell() const {
        if (!atomic_data_) {
            throw std::runtime_error("RSATB has no atomic data yet.");
        }
        return atomic_data_->cell;
    }

    /// Returns the spectral window [emin, emax] the expansions are scaled to.
    ///
    /// Callers need it to place energies: every Chebyshev observable is
    /// defined on the open interval, and the reconstruction returns zero at
    /// and beyond the endpoints.
    std::pair<double, double> energy_bounds() {
        auto& ops = require_operators();
        return std::make_pair(ops.emin(), ops.emax());
    }

    void compute_ac(int nt, int n_kpm, double temp, double mu, std::vector<double>& ac_out,
                    std::vector<double>& omega_out, std::string axis = "ab",
                    const std::string& method = "concurrent", int block = 0) {
        calc::ComputeAc(observable_context(), nt, n_kpm, temp, mu, ac_out, omega_out, axis,
                        method, block);
    }

    void compute_dc_spectrum(int n_moments, std::vector<double>& sigma_out,
                             std::vector<double>& energy_out,
                             std::string formulation = "greenwood", std::string axis = "abc",
                             int partition_sections = 0, size_t max_workspace_mb = 0,
                             std::string kernel = "jackson",
                             const std::string& method = "fftcheb") {
        calc::ComputeDcSpectrum(observable_context(), n_moments, sigma_out, energy_out,
                                formulation, axis, partition_sections, max_workspace_mb, kernel,
                                method);
    }

    void compute_dc(int n_moments, double mu, double temp, std::vector<double>& dc_out,
                    std::string formulation = "greenwood", std::string axis = "abc",
                    int partition_sections = 0, size_t max_workspace_mb = 0,
                    std::string kernel = "jackson", const std::string& method = "fftcheb") {
        calc::ComputeDc(observable_context(), n_moments, mu, temp, dc_out, formulation, axis,
                        partition_sections, max_workspace_mb, kernel, method);
    }

    calc::OccupationResult compute_occupation(double mu, double temp,
                                              double indicator_threshold = 1e-4, int n_max = 1 << 16) {
        return calc::ComputeOccupation(observable_context(), mu, temp, indicator_threshold, n_max);
    }

    calc::OccupationResult find_fermi_level(
        double n_electrons, double temp, double indicator_threshold = 1e-4, int n_max = 1 << 16,
        double mu_guess = std::numeric_limits<double>::quiet_NaN()) {
        return calc::ComputeFermiLevel(observable_context(), n_electrons, temp, indicator_threshold,
                                       n_max, mu_guess);
    }

    void compute_msd_spectrum(int n_steps, double dt_fs, int n_kpm, std::vector<double>& msd_out,
                              std::vector<double>& energy_out, std::vector<double>& time_out,
                              std::string axis = "x") {
        calc::ComputeMsdSpectrum(observable_context(), n_steps, dt_fs, n_kpm, msd_out, energy_out,
                                 time_out, axis);
    }

    /// Full-cell LDOS map on the real-space grid (calc/rs_ldos.hpp).
    ///
    /// Gamma-only, and needs a model with real-space orbitals (LCAO, or DFTB
    /// with .orb files). Bounds come from the operator set; the options'
    /// emin/emax are overwritten. Each rank returns its own compact share of
    /// the grid -- see RealSpaceLdosResult.
    calc::RealSpaceLdosResult compute_rs_ldos(calc::RealSpaceLdosOptions options) {
        return calc::ComputeRealSpaceLdos(observable_context(), orbital_files_,
                                          std::move(options));
    }

    void compute_quantum_metric(int nt, int n_kpm, double temp, double mu,
                                std::vector<double>& gxx_out, std::vector<double>& sigma_xx_out,
                                std::vector<double>& omega_out, std::string axis = "abc",
                                double omega_min_ev = 0.0) {
        calc::ComputeQuantumMetric(observable_context(), nt, n_kpm, temp, mu, gxx_out,
                                   sigma_xx_out, omega_out, axis, omega_min_ev);
    }

    // Test Helpers
    Matrix* get_H_k(int k_idx) {
        return operators_ ? operators_->hamiltonian_k(k_idx) : nullptr;
    }

    setup::OperatorSet* operators() {
        return operators_.get();
    }

    const setup::OperatorSet* operators() const {
        return operators_.get();
    }
    
    AtomicData* atomic_data() {
        return atomic_data_.get();
    }

private:
    /// Records which rank may print and starts the profile clock.
    ///
    /// The level itself is process-wide state, set once by set_log_level()
    /// before a calculation is built -- it is not a constructor argument.
    /// It used to be both, and the two spellings disagreed: a `verbose` bool
    /// here silently meant kDebug and could out-shout an explicit level.
    /// Components that want "is this a debug run" ask util::Log, which is
    /// also rank-aware, so no component has to re-derive the printing rank.
    void begin_profile() {
        util::Log::rank() = world_rank_of(world_comm_);
        util::Profile::enabled() = true;
        util::Profile::reset();
        profile_open_ = true;
    }

    void init_comms() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized || world_comm_ == MPI_COMM_NULL) {
            world_rank_ = 0;
            world_size_ = 1;
            sub_rank_ = 0;
            sub_size_ = 1;
            sub_comm_ = MPI_COMM_NULL;
            master_comm_ = MPI_COMM_NULL;
            return;
        }

        MPI_Comm_rank(world_comm_, &world_rank_);
        MPI_Comm_size(world_comm_, &world_size_);

        if (npar_ > world_size_) npar_ = world_size_;

        int color = (world_rank_ * npar_) / world_size_;
        MPI_Comm_split(world_comm_, color, world_rank_, &sub_comm_);
        
        MPI_Comm_rank(sub_comm_, &sub_rank_);
        MPI_Comm_size(sub_comm_, &sub_size_);
        
        int master_color = (sub_rank_ == 0) ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(world_comm_, master_color, world_rank_, &master_comm_);
    }

    void init_components() {
        operators_.reset();
        
        // int color = (world_rank_ * npar_) / world_size_;
        // int io_unit = (sub_rank_ == 0 && color == 0) ? 6 : 0; 
        // c_dbcsr_init_lib(sub_comm_, &io_unit);
    }

    void release_resources() {
        sampling_.reset();
        operators_.reset();
        atomic_data_.reset();

        if (mpi_runtime_active()) {
            if (sub_comm_ != MPI_COMM_NULL) MPI_Comm_free(&sub_comm_);
            if (master_comm_ != MPI_COMM_NULL) MPI_Comm_free(&master_comm_);
        }
        sub_comm_ = MPI_COMM_NULL;
        master_comm_ = MPI_COMM_NULL;
    }

    static bool mpi_runtime_active() {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Finalized(&finalized);
        }
        return initialized && !finalized;
    }

    /// Rejects a graph whose local pieces do not add up to the declared totals.
    ///
    /// `AtomicGraphArrays` is a *distributed* description: each rank supplies
    /// only its own atoms and edges. Handing every rank the full graph instead
    /// is the natural mistake, and it silently builds a system duplicated once
    /// per rank -- a doubled spectrum rather than an error. Summing the local
    /// counts across the sub-communicator catches exactly that.
    void RequireConsistentGraphPartition(const AtomicGraphArrays& graph) const {
        if (sub_comm_ == MPI_COMM_NULL) {
            return;
        }

        long long local[2] = {static_cast<long long>(graph.n_atom),
                              static_cast<long long>(graph.n_edge)};
        long long total[2] = {0, 0};
        MPI_Allreduce(local, total, 2, MPI_LONG_LONG, MPI_SUM, sub_comm_);

        if (graph.N_atom != 0 && total[0] != static_cast<long long>(graph.N_atom)) {
            throw std::runtime_error(
                "AtomicGraphArrays is a distributed description: the local n_atom values sum to " +
                std::to_string(total[0]) + " across this sub-communicator, but N_atom is " +
                std::to_string(graph.N_atom) +
                ". Each rank must supply only its own atoms, or run with one rank per sub-communicator.");
        }
        if (graph.N_edge != 0 && total[1] != static_cast<long long>(graph.N_edge)) {
            throw std::runtime_error(
                "AtomicGraphArrays is a distributed description: the local n_edge values sum to " +
                std::to_string(total[1]) + " across this sub-communicator, but N_edge is " +
                std::to_string(graph.N_edge) +
                ". Each rank must supply only its own edges.");
        }
    }

    /// Rejects a layout that leaves a rank with no atoms.
    ///
    /// The operator stack assumes every rank in a sub-communicator owns part of
    /// the system; an empty rank deadlocks it rather than idling. That is only
    /// reachable by giving a sub-communicator more ranks than the cell has
    /// atoms, which is never what anyone wants, so say so instead of hanging.
    /// Raising `npar` puts the extra ranks on k-points, where they are useful.
    void RequireEveryRankOwnsAtoms() const {
        if (sub_comm_ == MPI_COMM_NULL || atomic_data_ == nullptr) {
            return;
        }
        int sub_size = 1;
        MPI_Comm_size(sub_comm_, &sub_size);
        if (sub_size <= 1) {
            return;
        }

        int local = atomic_data_->n_atom;
        int fewest = local;
        MPI_Allreduce(&local, &fewest, 1, MPI_INT, MPI_MIN, sub_comm_);
        if (fewest > 0) {
            return;
        }

        int total = 0;
        MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, sub_comm_);
        throw std::runtime_error(
            "This sub-communicator has " + std::to_string(sub_size) + " ranks but the cell has " +
            std::to_string(total) + " atoms, so at least one rank owns none. Use fewer ranks, or "
            "raise npar so the extra ranks work on k-points instead.");
    }

    setup::OperatorRuntimeContext make_operator_runtime_context() const {
        setup::OperatorRuntimeContext runtime;
        runtime.npar = npar_;
        runtime.world_rank = world_rank_;
        runtime.world_size = world_size_;
        runtime.master_comm = master_comm_;
        return runtime;
    }

    model::SetupOptions normalize_setup_options(model::SetupOptions options) const {
        if (options.kpoints.empty()) {
            options.kpoints = {{0.0, 0.0, 0.0}};
        }
        if (options.nsamples < 1) {
            options.nsamples = 1;
        }
        return options;
    }

    // The old validate_pairwise_model_setup() cross-checked user-supplied
    // orbital counts and cutoffs against a second derivation from the model.
    // Both now come from the model's BasisSpec, so there is nothing left to
    // reconcile; the block structure is checked once, where it is used, by
    // SkTableConstruction::validate_atomic_data.

    /// Bundles what the observables need out of this object.
    calc::ObservableContext observable_context() {
        calc::ObservableContext context;
        context.operators = operators_.get();
        context.atomic_data = atomic_data_.get();
        context.samples = &sampling_;
        context.master_comm = master_comm_;
        context.sub_comm = sub_comm_;
        context.verbose = util::Log::enabled(util::LogLevel::kDebug);
        return context;
    }

    setup::OperatorSet& require_operators() {
        if (!operators_) {
            throw std::runtime_error("RSATB is not set up yet. Call setup(...) before computing properties.");
        }
        return *operators_;
    }

    const setup::OperatorSet& require_operators() const {
        if (!operators_) {
            throw std::runtime_error("RSATB is not set up yet. Call setup(...) before computing properties.");
        }
        return *operators_;
    }

    MPI_Comm world_comm_;
    MPI_Comm sub_comm_;
    MPI_Comm master_comm_;
    int world_rank_, world_size_;
    int sub_rank_, sub_size_;
    int npar_;
    /// True between begin_profile() and the report that closes it, so a setup()
    /// reached through a constructor does not restart a clock already running.
    bool profile_open_ = false;

    std::unique_ptr<AtomicData> atomic_data_;
    std::unique_ptr<setup::OperatorSet> operators_;
    /// Radial orbital files of the model last set up, empty if it has none.
    std::vector<std::string> orbital_files_;
    calc::SampleVectors sampling_;
};

} // namespace rsatb

#endif
