#ifndef RSATB_CALC_OBSERVABLES_HPP
#define RSATB_CALC_OBSERVABLES_HPP

// The physical quantities RSATB computes.
//
// These were methods on RSATB, which meant they could only be exercised by
// constructing a full calculation: communicators, atomic data, a model, and a
// setup pass. As free functions over an ObservableContext they depend on the
// operators and sample vectors alone.
//
// Every one of them has the same shape: loop over the k-points this rank owns,
// write into the slice of a [global_nk][per_k] buffer, and reduce. The context
// owns that reduction so the loop bodies stay about physics.

#include "../setup/operator_assembly.hpp"
#include "ac.hpp"
#include "cheby.hpp"
#include "collocation.hpp"
#include "dos.hpp"
#include "fft.hpp"
#include "fftcheb_transport.hpp"
#include "msd.hpp"
#include "occupation.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "vbcsr/core/dist_vector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace rsatb {
namespace calc {

using ObservableScalar = std::complex<double>;

/// The random vectors the stochastic trace estimators sample with.
struct SampleVectors {
    using Vector = vbcsr::DistVector<ObservableScalar>;
    using MultiVector = vbcsr::DistMultiVector<ObservableScalar>;

    int nsamples = 1;
    std::unique_ptr<Vector> sample_vec;
    std::unique_ptr<MultiVector> sample_multi_vec;

    void reset() {
        sample_vec.reset();
        sample_multi_vec.reset();
        nsamples = 1;
    }

    /// Draws a fresh set of normally distributed sample vectors.
    ///
    /// A single sample uses the vector form rather than a one-column
    /// multivector because the kernels have a cheaper path for it.
    void initialize(vbcsr::atomic::AtomicData& atomic_data, int requested_nsamples) {
        reset();
        nsamples = requested_nsamples;
        if (nsamples == 1) {
            sample_vec = std::make_unique<Vector>(atomic_data.get_graph());
            sample_vec->set_random_normal(true);
            return;
        }
        sample_multi_vec = std::make_unique<MultiVector>(atomic_data.get_graph(), nsamples);
        sample_multi_vec->set_random_normal(true);
    }
};

/// Everything an observable needs that is not one of its own arguments.
struct ObservableContext {
    using Matrix = vbcsr::BlockSpMat<ObservableScalar>;

    setup::OperatorSet* operators = nullptr;
    vbcsr::atomic::AtomicData* atomic_data = nullptr;
    const SampleVectors* samples = nullptr;
    /// Leaders of each sub-communicator; per-k results are summed across it.
    MPI_Comm master_comm = MPI_COMM_NULL;
    /// This rank's sub-communicator; complete results are broadcast within it.
    MPI_Comm sub_comm = MPI_COMM_NULL;
    bool verbose = false;

    setup::OperatorSet& RequireOperators() const {
        if (operators == nullptr) {
            throw std::runtime_error(
                "RSATB is not set up yet. Call setup(...) before computing properties.");
        }
        return *operators;
    }

    /// Reduces a per-k-point result across ranks, in place.
    ///
    /// Sub-communicator leaders sum their disjoint k-point slices, then the
    /// complete array is broadcast within each sub-communicator. Summation is
    /// safe precisely because the slices are disjoint and unowned entries are
    /// zero.
    void GatherKPointResults(std::vector<double>* values) const {
        if (values->empty()) return;
        if (master_comm != MPI_COMM_NULL) {
            std::vector<double> reduced(values->size(), 0.0);
            MPI_Allreduce(values->data(), reduced.data(), static_cast<int>(values->size()),
                          MPI_DOUBLE, MPI_SUM, master_comm);
            values->swap(reduced);
        }
        if (sub_comm != MPI_COMM_NULL) {
            MPI_Bcast(values->data(), static_cast<int>(values->size()), MPI_DOUBLE, 0, sub_comm);
        }
    }

    /// Publishes a grid that is identical for every k-point, in place.
    ///
    /// Energy and frequency axes depend only on the spectral bounds, so a rank
    /// owning no k-point computes nothing and must be given the grid rather
    /// than contribute to it. This replaces summing the grid across ranks,
    /// which worked only because non-contributing ranks left it zero-filled.
    void BroadcastSharedGrid(std::vector<double>* values, bool owner_has_values) const {
        if (values->empty()) return;
        if (master_comm != MPI_COMM_NULL) {
            int master_rank = 0;
            int master_size = 1;
            MPI_Comm_rank(master_comm, &master_rank);
            MPI_Comm_size(master_comm, &master_size);
            const int candidate = owner_has_values ? master_rank : master_size;
            int root = candidate;
            MPI_Allreduce(&candidate, &root, 1, MPI_INT, MPI_MIN, master_comm);
            if (root < master_size) {
                MPI_Bcast(values->data(), static_cast<int>(values->size()), MPI_DOUBLE, root, master_comm);
            }
        }
        if (sub_comm != MPI_COMM_NULL) {
            MPI_Bcast(values->data(), static_cast<int>(values->size()), MPI_DOUBLE, 0, sub_comm);
        }
    }

    /// Builds the Chebyshev driver matching which operators this k-point has.
    std::unique_ptr<Chebyshev<ObservableScalar>> MakeChebyshev(
        const setup::OperatorSet::KPointOperatorState& state) const {
        Matrix* hk = state.hamiltonian.get();
        Matrix* sk = state.overlap.get();
        auto* preconditioner = state.preconditioner.get();
        const auto& ops = RequireOperators();

        if (!hk) {
            throw std::runtime_error("Hamiltonian is not available for the requested k-point.");
        }
        if (sk && preconditioner) {
            return std::make_unique<Chebyshev<ObservableScalar>>(
                *hk, *sk, *preconditioner, preconditioner->get_type(), ops.emin(), ops.emax(), verbose);
        }
        if (sk) {
            return std::make_unique<Chebyshev<ObservableScalar>>(*hk, *sk, ops.emin(), ops.emax(), verbose);
        }
        if (preconditioner) {
            return std::make_unique<Chebyshev<ObservableScalar>>(
                *hk, *preconditioner, preconditioner->get_type(), ops.emin(), ops.emax(), verbose);
        }
        return std::make_unique<Chebyshev<ObservableScalar>>(*hk, ops.emin(), ops.emax(), verbose);
    }

    /// Returns the global index of the Gamma point, agreed across all ranks.
    int SelectGammaKPointIndex() const;
};

inline bool IsGammaKPoint(const std::vector<double>& kpoint, double tol = 1e-12) {
    for (double value : kpoint) {
        if (std::abs(value) > tol) return false;
    }
    return true;
}

inline int ObservableContext::SelectGammaKPointIndex() const {
    const auto& ops = RequireOperators();
    int local_gamma = ops.global_nk();
    for (const auto& kp : ops.local_kpoints()) {
        if (IsGammaKPoint(kp.second)) {
            local_gamma = std::min(local_gamma, kp.first);
        }
    }

    int global_gamma = local_gamma;
    if (master_comm != MPI_COMM_NULL) {
        MPI_Allreduce(&local_gamma, &global_gamma, 1, MPI_INT, MPI_MIN, master_comm);
    }
    if (sub_comm != MPI_COMM_NULL) {
        MPI_Bcast(&global_gamma, 1, MPI_INT, 0, sub_comm);
    }
    if (global_gamma >= ops.global_nk()) {
        throw std::runtime_error(
            "This observable requires a Gamma point in the setup k-point list.");
    }
    return global_gamma;
}

/// Maps an axis name onto a Cartesian index.
inline int AxisToIndex(const std::string& axis) {
    if (axis == "x") return 0;
    if (axis == "y") return 1;
    if (axis == "z") return 2;
    throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
}

inline const op::PositionAxisData& SelectAxisData(
    const op::PositionTriplet& position, int axis_idx) {
    if (axis_idx == 0) return position.x;
    if (axis_idx == 1) return position.y;
    if (axis_idx == 2) return position.z;
    throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
}

inline const op::LatticeMomentAxis& SelectAxisData(
    const setup::LatticeMomentTriplet& first_lattice_moment, int axis_idx) {
    if (axis_idx == 0) return first_lattice_moment.x;
    if (axis_idx == 1) return first_lattice_moment.y;
    if (axis_idx == 2) return first_lattice_moment.z;
    throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
}

inline void ComputeDos(const ObservableContext& context, int nt, std::vector<double>& dos_out, std::vector<double>& omega_out) {
        // Output size: global_nk * 2*nt
        auto& ops = context.RequireOperators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }

        std::vector<double> local_dos(ops.global_nk() * 2 * nt, 0.0);
        
        std::complex<double> imI(0.0, 1.0);
        double scale = (ops.emax() - ops.emin()) / 2.0;
        double shift = (ops.emax() + ops.emin()) / 2.0;
        
        for (const auto& kp : ops.local_kpoints()) {
            int k_idx = kp.first;
            const auto& state = ops.kpoint_state(k_idx);
            
            auto cheby = context.MakeChebyshev(state);
            
            std::vector<std::complex<double>> cdos;
            const SampleVectors::Vector* null_vec = nullptr;
            std::vector<std::complex<double>>* null_res = nullptr;
            
            if (context.samples->nsamples == 1) {
                rsatb::calc::corr_dos(*cheby, *context.samples->sample_vec, nt, cdos, null_vec, null_res, context.verbose);
            } else {
                rsatb::calc::corr_dos(*cheby, *context.samples->sample_multi_vec, nt, cdos, null_vec, null_res, context.verbose);
            }
            
            // Transform per K-point
            std::vector<std::complex<double>> cdos_full(2 * nt + 1);
            cdos_full[0] = cdos[0];
            
            for(int i=1; i<=nt; ++i) {
                // std::complex<double> poscorr = std::exp(imI * shift * M_PI * (double)i);
                // std::complex<double> negcorr = std::exp(-imI * shift * M_PI * (double)i);
                double factor = 0.5 * (1 + std::cos(M_PI * (i-1) / nt));
                cdos_full[i] = cdos[i] * factor;
                cdos_full[2*nt - i] = std::conj(cdos[i]) * factor;
            }
            
            std::vector<std::complex<double>> temp_dos(2 * nt);
            rsatb::calc::ifft_1d(2 * nt, cdos_full.data(), temp_dos.data());
            
            // Store in local_dos at offset k_idx * 2*nt
            size_t offset = k_idx * 2 * nt;
            for(int i=0; i<nt; ++i) {
                local_dos[offset + i] = temp_dos[nt+i].real();
                local_dos[offset + nt + i] = temp_dos[i].real();
            }
        }
        
        dos_out = std::move(local_dos);
        context.GatherKPointResults(&dos_out);

        // Omega depends only on the spectral bounds, so every rank builds it.
        omega_out.resize(2 * nt);
        for(int i=0; i<nt; ++i) {
            omega_out[i] = (i - nt) * scale / nt + shift;
            omega_out[nt + i] = i * scale / nt + shift;
        }
    }

inline void ComputeAc(const ObservableContext& context, int nt, int n_kpm, double temp, double mu, std::vector<double>& ac_out, std::vector<double>& omega_out, std::string axis="ab") {
        // Output size: global_nk * (nt+1) * 6
        auto& ops = context.RequireOperators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        if (!ops.has_velocity()) {
            throw std::runtime_error("compute_ac requires a velocity-capable model/operator setup.");
        }
        
        std::vector<double> local_ac(ops.global_nk() * (nt + 1) * 6, 0.0);
        
        double scale = (ops.emax() - ops.emin()) / 2.0;
        double shift = (ops.emax() + ops.emin()) / 2.0;
        double beta = (temp > 0.0) ? (1.0 / (temp * 8.617333262e-5)) : std::numeric_limits<double>::infinity();
        double ac_prefactor = (double)context.atomic_data->norb() / context.atomic_data->volume(axis);
        double t_step = M_PI / scale;
        
        // Precompute omega
        std::vector<double> omega(nt + 1);
        for(int i=0; i<=nt; ++i) {
             omega[i] = i * scale / (nt + 1);
        }

        for (const auto& kp : ops.local_kpoints()) {
            int k_idx = kp.first;
            const auto& state = ops.kpoint_state(k_idx);
            const auto& velocity = ops.velocity_state(k_idx);
            auto cheby = context.MakeChebyshev(state);
            
            std::vector<std::complex<double>> cac;
            
            if (context.samples->nsamples == 1) {
                rsatb::calc::corr_ac(*cheby, *velocity.x, *velocity.y, *velocity.z, *context.samples->sample_vec, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, context.verbose);
            } else {
                rsatb::calc::corr_ac(*cheby, *velocity.x, *velocity.y, *velocity.z, *context.samples->sample_multi_vec, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, context.verbose);
            }

            
            // Transform and store in local_ac
            size_t k_offset = k_idx * (nt + 1) * 6;
            
            for( int iac=0; iac<6; ++iac ){
                for( int iw=0; iw<=nt; ++iw ){
                    size_t idx = k_offset + iw*6+iac;
                    double val = 0.0;
                    if (omega[iw] != 0.0){
                        for( int it=0; it<=nt; ++it ){
                            val += 2.0 * std::sin(omega[iw]*it*t_step) * std::exp(-2.0*std::pow((double)(it+1) / (nt+1), 2)) * cac[it*6+iac].imag();
                        }
                        const double thermal_factor =
                            (temp > 0.0) ? (std::exp(-beta * omega[iw]) - 1.0) : -1.0;
                        val = ac_prefactor * t_step * val * thermal_factor / omega[iw];
                    }
                    local_ac[idx] = val;
                }
            }
        }
        
        ac_out = std::move(local_ac);
        context.GatherKPointResults(&ac_out);

        omega_out = omega;
    }

    // Compute the full zero-temperature transport spectrum on the native
    // Chebyshev energy grid for every k-point handled by this RSATB instance.
    //
    // Output layout:
    // - `sigma_out` has shape [global_nk_, n_moments, 6] flattened in row-major
    //   order with tensor components stored as (xx, xy, xz, yy, yz, zz).
    // - `energy_out` has length `n_moments` and is shared by all k-points.
    //
    // `formulation="greenwood"` fills the diagonal components from the
    // longitudinal DC spectrum.
    // `formulation="bastin"` fills the full upper-triangular tensor and
    // includes Hall-type responses for complex Hamiltonians.
inline void ComputeDcSpectrum(
    const ObservableContext& context,
        int n_moments,
        std::vector<double>& sigma_out,
        std::vector<double>& energy_out,
        std::string formulation = "greenwood",
        std::string axis = "abc",
        int partition_sections = 0,
        size_t max_workspace_mb = 0,
        std::string kernel = "jackson"
    ) {
        auto& ops = context.RequireOperators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        if (!ops.has_velocity()) {
            throw std::runtime_error("compute_dc_spectrum requires a velocity-capable model/operator setup.");
        }

        const auto dc_formulation = parse_dc_formulation(formulation);
        const double trace_prefactor = static_cast<double>(context.atomic_data->norb()) / context.atomic_data->volume(axis);
        const double transport_prefactor =
            (dc_formulation == DCFormulation::Greenwood ? 4.0 : 8.0) * trace_prefactor;

        std::vector<double> local_sigma(ops.global_nk() * n_moments * 6, 0.0);
        std::vector<double> local_energy;

        for (const auto& kp : ops.local_kpoints()) {
            int k_idx = kp.first;
            const auto& state = ops.kpoint_state(k_idx);
            const auto& velocity = ops.velocity_state(k_idx);
            auto cheby = context.MakeChebyshev(state);

            SampleVectors::MultiVector sample_mv(context.atomic_data->get_graph(), context.samples->nsamples == 1 ? 1 : context.samples->nsamples);
            if (context.samples->nsamples == 1) {
                sample_mv.set_col(0, *context.samples->sample_vec);
            } else {
                sample_mv.copy_from(*context.samples->sample_multi_vec);
            }

            std::vector<double> sigma_k;
            std::vector<double> energy_k;
            rsatb::calc::dc_spectrum_fftcheb(
                *cheby, *velocity.x, *velocity.y, *velocity.z, sample_mv, n_moments, dc_formulation,
                transport_prefactor, partition_sections, max_workspace_mb,
                sigma_k, energy_k, context.verbose, parse_kpm_kernel(kernel)
            );

            if (local_energy.empty()) {
                local_energy = energy_k;
            }

            const size_t offset = static_cast<size_t>(k_idx) * static_cast<size_t>(n_moments) * 6;
            std::copy(sigma_k.begin(), sigma_k.end(), local_sigma.begin() + offset);
        }

        sigma_out = std::move(local_sigma);
        context.GatherKPointResults(&sigma_out);

        // A sub-communicator that owns no k-point computed no energy grid, so
        // it has to receive one rather than contribute zeros.
        const bool has_energy = !local_energy.empty();
        energy_out.assign(static_cast<size_t>(n_moments), 0.0);
        if (has_energy) {
            energy_out = local_energy;
        }
        context.BroadcastSharedGrid(&energy_out, has_energy);
    }

    // Convenience wrapper over `compute_dc_spectrum(...)`.
    // It evaluates the finite-temperature conductivity tensor by integrating
    // the zero-temperature spectrum against -df/dE at the target (mu, T).
inline void ComputeDc(
    const ObservableContext& context,
        int n_moments,
        double mu,
        double temp,
        std::vector<double>& dc_out,
        std::string formulation = "greenwood",
        std::string axis = "abc",
        int partition_sections = 0,
        size_t max_workspace_mb = 0,
        std::string kernel = "jackson"
    ) {
        std::vector<double> sigma_spectrum;
        std::vector<double> energy_out;
        ComputeDcSpectrum(context, n_moments, sigma_spectrum, energy_out, formulation, axis, partition_sections, max_workspace_mb, kernel);

        auto& ops = context.RequireOperators();
        dc_out.assign(ops.global_nk() * 6, 0.0);
        for (int k_idx = 0; k_idx < ops.global_nk(); ++k_idx) {
            std::vector<double> sigma_k(sigma_spectrum.begin() + static_cast<size_t>(k_idx) * static_cast<size_t>(n_moments) * 6,
                                        sigma_spectrum.begin() + static_cast<size_t>(k_idx + 1) * static_cast<size_t>(n_moments) * 6);
            std::vector<double> dc_k;
            rsatb::calc::integrate_dc_spectrum(sigma_k, energy_out, mu, temp, dc_k);
            std::copy(dc_k.begin(), dc_k.end(), dc_out.begin() + static_cast<size_t>(k_idx) * 6);
        }
    }

inline void ComputeMsdSpectrum(
    const ObservableContext& context,
        int n_steps,
        double dt_fs,
        int n_kpm,
        std::vector<double>& msd_out,
        std::vector<double>& energy_out,
        std::vector<double>& time_out,
        std::string axis = "x"
    ) {
        // MSD always uses the configured Gamma-point state. Once that state is
        // selected, the only remaining switch is which position representation
        // is valid for the requested axis.

        auto& ops = context.RequireOperators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        if (!ops.has_hamiltonian()) {
            throw std::runtime_error("compute_msd_spectrum requires a Hamiltonian-capable model/operator setup.");
        }

        const size_t per_k_size = static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm);
        std::vector<double> local_msd(per_k_size, 0.0);
        std::vector<double> local_energy;
        const int axis_idx = AxisToIndex(axis);
        const int gamma_k_idx = context.SelectGammaKPointIndex();
        const auto* gamma_state = ops.try_kpoint_state(gamma_k_idx);

        if (gamma_state != nullptr) {
            auto cheby = context.MakeChebyshev(*gamma_state);
            const auto& first_lattice_moment = ops.first_lattice_moment_state(gamma_k_idx);
            const auto& moment_axis = SelectAxisData(first_lattice_moment, axis_idx);
            const bool periodic_axis = static_cast<bool>(moment_axis.first_moment);

            auto run_msd_for_sample = [&](const auto& sample) {
                if (periodic_axis) {
                    if (ops.has_overlap()) {
                        throw std::runtime_error(
                            "compute_msd_spectrum periodic MSD currently supports orthogonal systems only.");
                    }
                    const auto& position = ops.position_state(gamma_k_idx);
                    const auto& position_axis = SelectAxisData(position, axis_idx);
                    if (!position_axis.orthogonal_diagonal) {
                        throw std::runtime_error("Orthogonal diagonal position data is missing for affine MSD.");
                    }

                    // MSD is defined from the Gamma-point wavepacket evolution. For periodic
                    // axes we switch to the affine route, where the first lattice moment
                    // accounts for wrap-around contributions and the orthogonal diagonal
                    // position provides the intra-cell coordinate.
                    rsatb::calc::corr_msd(
                        *cheby,
                        *moment_axis.first_moment,
                        *position_axis.orthogonal_diagonal,
                        sample,
                        n_steps,
                        dt_fs,
                        n_kpm,
                        local_msd,
                        local_energy,
                        context.verbose);
                    return;
                }

                if (!ops.has_position()) {
                    throw std::runtime_error("compute_msd_spectrum requires a position-capable model/operator setup.");
                }
                const auto& position = ops.position_state(gamma_k_idx);
                const auto& position_axis = SelectAxisData(position, axis_idx);
                if (ops.has_overlap()) {
                    if (!position_axis.matrix) {
                        throw std::runtime_error("Explicit position matrix data is missing for MSD.");
                    }
                    // Open-boundary MSD also uses the Gamma-point Hamiltonian. With overlap the
                    // position operator is not diagonal in the working basis, so we use the
                    // explicit real-space matrix route.
                    rsatb::calc::corr_msd(
                        *cheby,
                        *position_axis.matrix,
                        sample,
                        n_steps,
                        dt_fs,
                        n_kpm,
                        local_msd,
                        local_energy,
                        context.verbose);
                    return;
                }

                if (!position_axis.orthogonal_diagonal) {
                    throw std::runtime_error("Orthogonal diagonal position data is missing for MSD.");
                }

                // Open-boundary orthogonal MSD still runs at Gamma; the route switch here is
                // only about how the position operator is represented, not about which k-point
                // is used.
                rsatb::calc::corr_msd(
                    *cheby,
                    *position_axis.orthogonal_diagonal,
                    sample,
                    n_steps,
                    dt_fs,
                    n_kpm,
                    local_msd,
                    local_energy,
                    context.verbose);
            };

            if (context.samples->nsamples == 1) {
                run_msd_for_sample(*context.samples->sample_vec);
            } else {
                run_msd_for_sample(*context.samples->sample_multi_vec);
            }
        }

        // MSD is evaluated at Gamma only, so exactly one sub-communicator
        // contributes and the rest hold zeros.
        msd_out = std::move(local_msd);
        context.GatherKPointResults(&msd_out);

        // The energy grid used to be summed across leaders, which gave the
        // right answer only because every non-Gamma rank left it zero-filled.
        // Broadcasting from the rank that actually computed it says what is
        // meant and does not depend on that coincidence.
        const bool has_energy = !local_energy.empty();
        energy_out.assign(static_cast<size_t>(n_kpm), 0.0);
        if (has_energy) {
            energy_out = local_energy;
        }
        context.BroadcastSharedGrid(&energy_out, has_energy);

        time_out.resize(static_cast<size_t>(n_steps + 1), 0.0);
        for (int step = 0; step <= n_steps; ++step) {
            time_out[static_cast<size_t>(step)] = dt_fs * static_cast<double>(step);
        }
    }

inline void ComputeQuantumMetric(
    const ObservableContext& context,
        int nt,
        int n_kpm,
        double temp,
        double mu,
        std::vector<double>& gxx_out,
        std::vector<double>& sigma_xx_out,
        std::vector<double>& omega_out,
        std::string axis = "abc",
        double omega_min_ev = 0.0
    ) {
        std::vector<double> ac_out;
        ComputeAc(context, nt, n_kpm, temp, mu, ac_out, omega_out, axis);

        auto& ops = context.RequireOperators();
        const int nk = ops.global_nk();
        const size_t n_omega = static_cast<size_t>(nt + 1);

        constexpr double sigma_au_to_si = 2.4341348e6;
        constexpr double hbar_ev_s = 6.582119569e-16;
        constexpr double h_si = 6.62607015e-34;
        constexpr double e_si = 1.602176634e-19;

        sigma_xx_out.assign(static_cast<size_t>(nk) * n_omega, 0.0);
        gxx_out.assign(static_cast<size_t>(nk), 0.0);

        std::vector<double> omega_rad_s(n_omega, 0.0);
        std::vector<int> mask_indices;
        mask_indices.reserve(n_omega);
        for (size_t iw = 0; iw < n_omega; ++iw) {
            omega_rad_s[iw] = omega_out[iw] / hbar_ev_s;
            if (omega_out[iw] > std::max(omega_min_ev, 1e-12)) {
                mask_indices.push_back(static_cast<int>(iw));
            }
        }

        for (int k_idx = 0; k_idx < nk; ++k_idx) {
            const size_t k_offset_ac = static_cast<size_t>(k_idx) * n_omega * 6;
            const size_t k_offset_sigma = static_cast<size_t>(k_idx) * n_omega;
            for (size_t iw = 0; iw < n_omega; ++iw) {
                sigma_xx_out[k_offset_sigma + iw] = ac_out[k_offset_ac + iw * 6] * sigma_au_to_si;
            }

        }

        if (mask_indices.size() < 2) {
            return;
        }

        const double prefactor = h_si / (M_PI * e_si * e_si);
        for (int k_idx = 0; k_idx < nk; ++k_idx) {
            const size_t k_offset_sigma = static_cast<size_t>(k_idx) * n_omega;
            double integral = 0.0;
            for (size_t idx = 1; idx < mask_indices.size(); ++idx) {
                const int left = mask_indices[idx - 1];
                const int right = mask_indices[idx];
                const double w0 = omega_rad_s[static_cast<size_t>(left)];
                const double w1 = omega_rad_s[static_cast<size_t>(right)];
                if (w0 <= 0.0 || w1 <= 0.0) {
                    continue;
                }

                const double f0 = sigma_xx_out[k_offset_sigma + static_cast<size_t>(left)] / w0;
                const double f1 = sigma_xx_out[k_offset_sigma + static_cast<size_t>(right)] / w1;
                integral += 0.5 * (f0 + f1) * (w1 - w0);
            }
            gxx_out[static_cast<size_t>(k_idx)] = prefactor * integral;
        }
    }

// ---------------------------------------------------------------------------
// Occupation / band energy / Fermi level (trace-mode ACE-FD, occupation.hpp)
// ---------------------------------------------------------------------------

struct OccupationResult {
    // Stochastic trace estimates: Tr[f(H)] and Tr[H f(H)] averaged over
    // k-points. No spin-degeneracy factor is applied.
    double n_electrons = 0.0;
    double band_energy = 0.0;
    double mu = 0.0;
    int n_moments = 0;
    double indicator = 0.0;
    // The stopping indicator fired at every k-point -- a heuristic signal, not
    // an error bound. See FermiExpansionOptions::indicator_threshold.
    bool indicator_fired = true;
};

namespace detail {

// Scalar companion of ObservableContext::GatherKPointResults: reduce with
// `op` across sub-communicator leaders, then broadcast within each
// sub-communicator. Every rank of a sub-communicator holds identical local
// values (collective dots), so only leaders participate in the reduction.
inline void ReduceScalars(const ObservableContext& context, double* values,
                          int n, MPI_Op op) {
    if (context.master_comm != MPI_COMM_NULL) {
        std::vector<double> reduced(static_cast<size_t>(n), 0.0);
        MPI_Allreduce(values, reduced.data(), n, MPI_DOUBLE, op, context.master_comm);
        std::copy(reduced.begin(), reduced.end(), values);
    }
    if (context.sub_comm != MPI_COMM_NULL) {
        MPI_Bcast(values, n, MPI_DOUBLE, 0, context.sub_comm);
    }
}

// Computes the per-k trace moments for this rank's k-points, adaptively
// anchored at (temp, mu_anchor). Aggregates the adaptivity metadata.
inline std::vector<TraceMoments> LocalTraceMoments(
    const ObservableContext& context, double temp, double mu_anchor,
    const FermiExpansionOptions& opt, OccupationResult* meta) {
    auto& ops = context.RequireOperators();
    std::vector<TraceMoments> kmoms;
    for (const auto& kp : ops.local_kpoints()) {
        auto cheby = context.MakeChebyshev(ops.kpoint_state(kp.first));
        TraceMoments tm =
            (context.samples->nsamples == 1)
                ? adaptive_trace_moments(*cheby, *context.samples->sample_vec,
                                         temp, mu_anchor, opt)
                : adaptive_trace_moments(*cheby, *context.samples->sample_multi_vec,
                                         temp, mu_anchor, opt);
        if (meta) {
            meta->n_moments = std::max(meta->n_moments, tm.n_used);
            meta->indicator = std::max(meta->indicator, tm.indicator);
            meta->indicator_fired = meta->indicator_fired && tm.indicator_fired;
        }
        kmoms.push_back(std::move(tm));
    }
    return kmoms;
}

} // namespace detail

// Stochastic-trace electron count and band energy at one (mu, T).
//
// Requires spectral bounds and sample vectors (as ComputeDos). The degree is
// chosen adaptively per k-point; on extended systems it is set by the
// spectral regularity near mu and is essentially temperature independent.
inline OccupationResult ComputeOccupation(const ObservableContext& context,
                                          double mu, double temp,
                                          double indicator_threshold = 1e-4,
                                          int n_max = 1 << 16) {
    auto& ops = context.RequireOperators();
    if (!ops.has_bounds()) {
        throw std::runtime_error(
            "Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
    }

    FermiExpansionOptions opt;
    opt.indicator_threshold = indicator_threshold;
    opt.n_max = n_max;

    OccupationResult res;
    res.mu = mu;
    const std::vector<TraceMoments> kmoms =
        detail::LocalTraceMoments(context, temp, mu, opt, &res);

    double sums[2] = {0.0, 0.0};
    for (const TraceMoments& tm : kmoms) {
        sums[0] += fermi_trace_sum(tm, temp, mu, KpmKernel::CosTaper, false);
        sums[1] += fermi_trace_sum(tm, temp, mu, KpmKernel::CosTaper, true);
    }
    detail::ReduceScalars(context, sums, 2, MPI_SUM);

    const double trace_scale = static_cast<double>(context.atomic_data->norb()) /
                               static_cast<double>(ops.global_nk());
    res.n_electrons = trace_scale * sums[0];
    res.band_energy = trace_scale * sums[1];

    double flags[2] = {res.indicator_fired ? 1.0 : 0.0, res.indicator};
    detail::ReduceScalars(context, &flags[0], 1, MPI_MIN);
    detail::ReduceScalars(context, &flags[1], 1, MPI_MAX);
    res.indicator_fired = flags[0] > 0.5;
    res.indicator = flags[1];
    double nmom = static_cast<double>(res.n_moments);
    detail::ReduceScalars(context, &nmom, 1, MPI_MAX);
    res.n_moments = static_cast<int>(nmom);
    return res;
}

// Finds mu such that the stochastic-trace electron count equals
// n_electrons_target at temperature temp, then returns occupation and band
// energy at that mu. One moment set is computed per k-point (anchored at
// mu_guess, default band center); the entire search happens in coefficient
// space at zero additional matvecs. Bracketing uses the monotone Jackson
// summation; the final value is refined with the tapered summation, and the
// convergence indicator is re-checked at the found mu.
inline OccupationResult ComputeFermiLevel(
    const ObservableContext& context, double n_electrons_target, double temp,
    double indicator_threshold = 1e-4, int n_max = 1 << 16,
    double mu_guess = std::numeric_limits<double>::quiet_NaN()) {
    auto& ops = context.RequireOperators();
    if (!ops.has_bounds()) {
        throw std::runtime_error(
            "Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
    }

    const double anchor =
        std::isfinite(mu_guess) ? mu_guess : 0.5 * (ops.emin() + ops.emax());

    FermiExpansionOptions opt;
    opt.indicator_threshold = indicator_threshold;
    opt.n_max = n_max;

    OccupationResult res;
    const std::vector<TraceMoments> kmoms =
        detail::LocalTraceMoments(context, temp, anchor, opt, &res);

    const double trace_scale = static_cast<double>(context.atomic_data->norb()) /
                               static_cast<double>(ops.global_nk());
    auto trace_sum = [&](double mu_trial, KpmKernel kernel, bool energy_weighted) {
        double local = 0.0;
        for (const TraceMoments& tm : kmoms) {
            local += fermi_trace_sum(tm, temp, mu_trial, kernel, energy_weighted);
        }
        detail::ReduceScalars(context, &local, 1, MPI_SUM);
        return trace_scale * local;
    };

    // Monotone bracket, then tapered refinement inside a small window.
    const double lo = ops.emin() - 1.0;
    const double hi = ops.emax() + 1.0;
    const double mu_jackson = bisect_fermi_level(
        [&](double m) { return trace_sum(m, KpmKernel::Jackson, false); },
        lo, hi, n_electrons_target);

    const double kb = 8.617333262e-5;
    const double delta = std::max(10.0 * kb * std::max(temp, 0.0),
                                  (ops.emax() - ops.emin()) * 1e-6);
    double mu_star = mu_jackson;
    if (trace_sum(mu_jackson - delta, KpmKernel::CosTaper, false) < n_electrons_target &&
        trace_sum(mu_jackson + delta, KpmKernel::CosTaper, false) > n_electrons_target) {
        mu_star = bisect_fermi_level(
            [&](double m) { return trace_sum(m, KpmKernel::CosTaper, false); },
            mu_jackson - delta, mu_jackson + delta, n_electrons_target, 60);
    }

    res.mu = mu_star;
    res.n_electrons = trace_sum(mu_star, KpmKernel::CosTaper, false);
    res.band_energy = trace_sum(mu_star, KpmKernel::CosTaper, true);

    // Honesty: the adaptive anchor was mu_guess; re-check at the found mu.
    double ind = res.indicator;
    for (const TraceMoments& tm : kmoms) {
        ind = std::max(ind, fermi_trace_indicator(tm, temp, mu_star));
    }
    double flags[2] = {res.indicator_fired ? 1.0 : 0.0, ind};
    detail::ReduceScalars(context, &flags[0], 1, MPI_MIN);
    detail::ReduceScalars(context, &flags[1], 1, MPI_MAX);
    res.indicator = flags[1];
    res.indicator_fired = flags[0] > 0.5 && res.indicator < indicator_threshold;
    double nmom = static_cast<double>(res.n_moments);
    detail::ReduceScalars(context, &nmom, 1, MPI_MAX);
    res.n_moments = static_cast<int>(nmom);
    return res;
}

}  // namespace calc
}  // namespace rsatb

#endif  // RSATB_CALC_OBSERVABLES_HPP
