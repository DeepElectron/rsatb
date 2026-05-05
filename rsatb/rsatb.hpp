#ifndef RSATB_CORE_RSATB_HPP
#define RSATB_CORE_RSATB_HPP

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "model/operator_set.hpp"
#include "model/io.hpp"
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

using namespace rsatb::backend;
using namespace rsatb::calc;


class RSATB {
public:
    using Scalar = std::complex<double>;
    using Real = double;
    using Matrix = BlockSpMat<Scalar>;
    using Vector = DistVector<Scalar>;
    using MultiVector = DistMultiVector<Scalar>;

    RSATB(int npar,
          const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
          const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb,
          const std::vector<std::string>& type_symbols,
          const int *edge_shift_vec,
          const double *cell, const double *pos,
          bool verbose = false,
          MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        init_comms();
        atomic_data_ = std::make_unique<AtomicData>(
            n_atom, N_atom, atom_offset, n_edge, N_edge,
            atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
            cell, pos, sub_comm_
        );
        r_max_per_type_.clear();
        type_symbols_ = type_symbols;
        init_components();
    }

    std::string get_kernel_info() const {
        return Matrix::KernelType::name();
    }

    // constructor defines the npar and the atomic structure.

    RSATB(int npar, const std::string& filename, const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb, bool verbose = false, MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        init_comms();
        atomic_data_ = std::unique_ptr<AtomicData>(AtomicData::from_file(filename, r_max_per_type, type_norb, sub_comm_));
        r_max_per_type_ = r_max_per_type;
        type_symbols_ = backend::unique_type_symbols_from_atomic_numbers(
            backend::io::read_structure(filename).z);
        init_components();
    }

    RSATB(int npar, const std::vector<double>& pos, const std::vector<int>& z, 
          const std::vector<double>& cell, const std::vector<bool>& pbc, 
          const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb, bool verbose = false, MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        init_comms();
        atomic_data_ = std::unique_ptr<AtomicData>(AtomicData::from_points(pos, z, cell, pbc, r_max_per_type, type_norb, sub_comm_));
        r_max_per_type_ = r_max_per_type;
        type_symbols_ = backend::unique_type_symbols_from_atomic_numbers(z);
        init_components();
    }
    
    ~RSATB() {
        release_resources();
    }


    // Setup Function defines the operators of one of the supported models (LCAO, SKTB, DFTB) and the associated capabilities.
    void setup(const model::AnyModelConfig& model_config, const model::SetupOptions& options = model::SetupOptions{}) {
        validate_model_setup(model_config);
        const auto normalized_options = normalize_setup_options(options);
        operators_ = backend::build_operator_set(
            *atomic_data_,
            sub_comm_,
            verbose_,
            model_config,
            normalized_options,
            type_symbols_,
            make_operator_runtime_context());
        sampling_.initialize(*atomic_data_, normalized_options.nsamples);
    }

    void setup(const model::LCAOModelConfig& model_config, const model::SetupOptions& options = model::SetupOptions{}) {
        setup(model::AnyModelConfig(model_config), options);
    }

    void setup(const model::SKTBModelConfig& model_config, const model::SetupOptions& options = model::SetupOptions{}) {
        setup(model::AnyModelConfig(model_config), options);
    }

    void setup(const model::DFTBModelConfig& model_config, const model::SetupOptions& options = model::SetupOptions{}) {
        setup(model::AnyModelConfig(model_config), options);
    }

    void test_solve() {
        LinearSolver<Scalar, Matrix::KernelType> solver(1e-8, 1000, verbose_);
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

    void compute_dos(int nt, std::vector<double>& dos_out, std::vector<double>& omega_out) {
        // Output size: global_nk * 2*nt
        auto& ops = require_operators();
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
            
            auto cheby = make_chebyshev(state);
            
            std::vector<std::complex<double>> cdos;
            const Vector* null_vec = nullptr;
            std::vector<std::complex<double>>* null_res = nullptr;
            
            if (sampling_.nsamples == 1) {
                rsatb::calc::corr_dos(*cheby, *sampling_.sample_vec, nt, cdos, null_vec, null_res, verbose_);
            } else {
                rsatb::calc::corr_dos(*cheby, *sampling_.sample_multi_vec, nt, cdos, null_vec, null_res, verbose_);
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
        
        // Gather all K-points results
        dos_out.assign(ops.global_nk() * 2 * nt, 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_dos.data(), dos_out.data(), local_dos.size(), MPI_DOUBLE, MPI_SUM, master_comm_);
        } else {
            dos_out = local_dos;
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(dos_out.data(), dos_out.size(), MPI_DOUBLE, 0, sub_comm_);
        }
        
        // Omega is same for all K-points
        omega_out.resize(2 * nt);
        for(int i=0; i<nt; ++i) {
            omega_out[i] = (i - nt) * scale / nt + shift;
            omega_out[nt + i] = i * scale / nt + shift;
        }
    }

    void compute_ac(int nt, int n_kpm, double temp, double mu, std::vector<double>& ac_out, std::vector<double>& omega_out, std::string axis="ab") {
        // Output size: global_nk * (nt+1) * 6
        auto& ops = require_operators();
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
        double ac_prefactor = (double)atomic_data_->norb() / atomic_data_->volume(axis);
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
            auto cheby = make_chebyshev(state);
            
            std::vector<std::complex<double>> cac;
            
            if (sampling_.nsamples == 1) {
                rsatb::calc::corr_ac(*cheby, *velocity.x, *velocity.y, *velocity.z, *sampling_.sample_vec, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, verbose_);
            } else {
                rsatb::calc::corr_ac(*cheby, *velocity.x, *velocity.y, *velocity.z, *sampling_.sample_multi_vec, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, verbose_);
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
        
        // Gather
        ac_out.assign(ops.global_nk() * (nt + 1) * 6, 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_ac.data(), ac_out.data(), local_ac.size(), MPI_DOUBLE, MPI_SUM, master_comm_);
        } else {
            ac_out = local_ac;
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(ac_out.data(), ac_out.size(), MPI_DOUBLE, 0, sub_comm_);
        }
        
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
    void compute_dc_spectrum(
        int n_moments,
        std::vector<double>& sigma_out,
        std::vector<double>& energy_out,
        std::string formulation = "greenwood",
        std::string axis = "abc",
        int partition_sections = 0,
        size_t max_workspace_mb = 0
    ) {
        auto& ops = require_operators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        if (!ops.has_velocity()) {
            throw std::runtime_error("compute_dc_spectrum requires a velocity-capable model/operator setup.");
        }

        const auto dc_formulation = parse_dc_formulation(formulation);
        const double trace_prefactor = static_cast<double>(atomic_data_->norb()) / atomic_data_->volume(axis);
        const double transport_prefactor =
            (dc_formulation == DCFormulation::Greenwood ? 4.0 : 8.0) * trace_prefactor;

        std::vector<double> local_sigma(ops.global_nk() * n_moments * 6, 0.0);
        std::vector<double> local_energy;

        for (const auto& kp : ops.local_kpoints()) {
            int k_idx = kp.first;
            const auto& state = ops.kpoint_state(k_idx);
            const auto& velocity = ops.velocity_state(k_idx);
            auto cheby = make_chebyshev(state);

            MultiVector sample_mv(atomic_data_->get_graph(), sampling_.nsamples == 1 ? 1 : sampling_.nsamples);
            if (sampling_.nsamples == 1) {
                sample_mv.set_col(0, *sampling_.sample_vec);
            } else {
                sample_mv.copy_from(*sampling_.sample_multi_vec);
            }

            std::vector<double> sigma_k;
            std::vector<double> energy_k;
            rsatb::calc::dc_spectrum_fftcheb(
                *cheby, *velocity.x, *velocity.y, *velocity.z, sample_mv, n_moments, dc_formulation,
                transport_prefactor, partition_sections, max_workspace_mb,
                sigma_k, energy_k, verbose_
            );

            if (local_energy.empty()) {
                local_energy = energy_k;
            }

            const size_t offset = static_cast<size_t>(k_idx) * static_cast<size_t>(n_moments) * 6;
            std::copy(sigma_k.begin(), sigma_k.end(), local_sigma.begin() + offset);
        }

        sigma_out.assign(local_sigma.size(), 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_sigma.data(), sigma_out.data(), static_cast<int>(local_sigma.size()), MPI_DOUBLE, MPI_SUM, master_comm_);
        } else {
            sigma_out = local_sigma;
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(sigma_out.data(), static_cast<int>(sigma_out.size()), MPI_DOUBLE, 0, sub_comm_);
        }

        energy_out.resize(n_moments, 0.0);
        if (!local_energy.empty()) {
            energy_out = local_energy;
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(energy_out.data(), n_moments, MPI_DOUBLE, 0, sub_comm_);
        }
    }

    // Convenience wrapper over `compute_dc_spectrum(...)`.
    // It evaluates the finite-temperature conductivity tensor by integrating
    // the zero-temperature spectrum against -df/dE at the target (mu, T).
    void compute_dc(
        int n_moments,
        double mu,
        double temp,
        std::vector<double>& dc_out,
        std::string formulation = "greenwood",
        std::string axis = "abc",
        int partition_sections = 0,
        size_t max_workspace_mb = 0
    ) {
        std::vector<double> sigma_spectrum;
        std::vector<double> energy_out;
        compute_dc_spectrum(n_moments, sigma_spectrum, energy_out, formulation, axis, partition_sections, max_workspace_mb);

        auto& ops = require_operators();
        dc_out.assign(ops.global_nk() * 6, 0.0);
        for (int k_idx = 0; k_idx < ops.global_nk(); ++k_idx) {
            std::vector<double> sigma_k(sigma_spectrum.begin() + static_cast<size_t>(k_idx) * static_cast<size_t>(n_moments) * 6,
                                        sigma_spectrum.begin() + static_cast<size_t>(k_idx + 1) * static_cast<size_t>(n_moments) * 6);
            std::vector<double> dc_k;
            rsatb::calc::integrate_dc_spectrum(sigma_k, energy_out, mu, temp, dc_k);
            std::copy(dc_k.begin(), dc_k.end(), dc_out.begin() + static_cast<size_t>(k_idx) * 6);
        }
    }

    void compute_msd_spectrum(
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

        auto& ops = require_operators();
        if (!ops.has_bounds()) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        if (!ops.has_hamiltonian()) {
            throw std::runtime_error("compute_msd_spectrum requires a Hamiltonian-capable model/operator setup.");
        }

        const size_t per_k_size = static_cast<size_t>(n_steps + 1) * static_cast<size_t>(n_kpm);
        std::vector<double> local_msd(per_k_size, 0.0);
        std::vector<double> local_energy;
        const int axis_idx = axis_to_index(axis);
        const int gamma_k_idx = select_gamma_kpoint_index(ops);
        const auto* gamma_state = ops.try_kpoint_state(gamma_k_idx);

        if (gamma_state != nullptr) {
            auto cheby = make_chebyshev(*gamma_state);
            const auto& first_lattice_moment = ops.first_lattice_moment_state(gamma_k_idx);
            const auto& moment_axis = select_axis_data(first_lattice_moment, axis_idx);
            const bool periodic_axis = static_cast<bool>(moment_axis.first_moment);

            auto run_msd_for_sample = [&](const auto& sample) {
                if (periodic_axis) {
                    if (ops.has_overlap()) {
                        throw std::runtime_error(
                            "compute_msd_spectrum periodic MSD currently supports orthogonal systems only.");
                    }
                    const auto& position = ops.position_state(gamma_k_idx);
                    const auto& position_axis = select_axis_data(position, axis_idx);
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
                        verbose_);
                    return;
                }

                if (!ops.has_position()) {
                    throw std::runtime_error("compute_msd_spectrum requires a position-capable model/operator setup.");
                }
                const auto& position = ops.position_state(gamma_k_idx);
                const auto& position_axis = select_axis_data(position, axis_idx);
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
                        verbose_);
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
                    verbose_);
            };

            if (sampling_.nsamples == 1) {
                run_msd_for_sample(*sampling_.sample_vec);
            } else {
                run_msd_for_sample(*sampling_.sample_multi_vec);
            }
        }

        msd_out.assign(local_msd.size(), 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_msd.data(), msd_out.data(), static_cast<int>(local_msd.size()), MPI_DOUBLE, MPI_SUM, master_comm_);
        } else {
            msd_out = local_msd;
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(msd_out.data(), static_cast<int>(msd_out.size()), MPI_DOUBLE, 0, sub_comm_);
        }

        energy_out.assign(static_cast<size_t>(n_kpm), 0.0);
        if (!local_energy.empty()) {
            energy_out = local_energy;
        }
        if (master_comm_ != MPI_COMM_NULL) {
            std::vector<double> reduced_energy(static_cast<size_t>(n_kpm), 0.0);
            MPI_Allreduce(
                energy_out.data(),
                reduced_energy.data(),
                n_kpm,
                MPI_DOUBLE,
                MPI_SUM,
                master_comm_);
            energy_out.swap(reduced_energy);
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(energy_out.data(), n_kpm, MPI_DOUBLE, 0, sub_comm_);
        }

        time_out.resize(static_cast<size_t>(n_steps + 1), 0.0);
        for (int step = 0; step <= n_steps; ++step) {
            time_out[static_cast<size_t>(step)] = dt_fs * static_cast<double>(step);
        }
    }

    void compute_quantum_metric(
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
        compute_ac(nt, n_kpm, temp, mu, ac_out, omega_out, axis);

        auto& ops = require_operators();
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


    // Test Helpers
    Matrix* get_H_k(int k_idx) {
        return operators_ ? operators_->hamiltonian_k(k_idx) : nullptr;
    }

    backend::OperatorSet* operators() {
        return operators_.get();
    }

    const backend::OperatorSet* operators() const {
        return operators_.get();
    }
    
    AtomicData* atomic_data() {
        return atomic_data_.get();
    }

    const std::vector<std::string>& type_symbols() const {
        return type_symbols_;
    }

private:
    struct SamplingState {
        int nsamples = 1;
        std::unique_ptr<Vector> sample_vec;
        std::unique_ptr<MultiVector> sample_multi_vec;

        void reset() {
            sample_vec.reset();
            sample_multi_vec.reset();
            nsamples = 1;
        }

        void initialize(AtomicData& atomic_data, int requested_nsamples) {
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

    backend::OperatorRuntimeContext make_operator_runtime_context() const {
        backend::OperatorRuntimeContext runtime;
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

    void validate_pairwise_model_setup(
        const backend::SKTBModel& sktb_model,
        const std::string& model_name,
        bool require_overlap = false
    ) const {
        if (!sktb_model.enabled()) {
            return;
        }
        if (require_overlap && !sktb_model.has_overlap) {
            throw std::runtime_error(model_name + " native setup requires overlap parameterization.");
        }

        const auto prepared = backend::prepare_sktb_atomic_data(sktb_model);

        if (prepared.type_norb.size() != atomic_data_->type_norb.size()) {
            throw std::runtime_error(model_name + " basis/type count does not match the current AtomicData.");
        }
        for (size_t type = 0; type < prepared.type_norb.size(); ++type) {
            if (prepared.type_norb[type] != atomic_data_->type_norb[type]) {
                throw std::runtime_error(
                    model_name + " basis-derived orbital count does not match AtomicData for species " +
                    prepared.type_symbols[type] + ".");
            }
        }

        if (r_max_per_type_.empty()) {
            return;
        }
        if (r_max_per_type_.size() != prepared.r_max_per_type.size()) {
            throw std::runtime_error("Stored AtomicData cutoffs do not match the " + model_name + " type count.");
        }

        for (const auto& pair : sktb_model.atom_pairs) {
            const double required_pair_cutoff = pair.max_cutoff();
            if (required_pair_cutoff <= 0.0) {
                continue;
            }
            const size_t row_type = static_cast<size_t>(pair.row_type);
            const size_t col_type = static_cast<size_t>(pair.col_type);
            const double available_pair_cutoff =
                r_max_per_type_[row_type] + r_max_per_type_[col_type];
            if (available_pair_cutoff + 1e-12 < required_pair_cutoff) {
                throw std::runtime_error(
                    "Current AtomicData graph cutoff is too small for " + model_name + " pair (" +
                    prepared.type_symbols[row_type] + ", " + prepared.type_symbols[col_type] + ").");
            }
        }
    }

    void validate_model_setup(const model::AnyModelConfig& model_config) const {
        std::visit(
            [&](const auto& cfg) {
                using Config = std::decay_t<decltype(cfg)>;
                if constexpr (std::is_same_v<Config, model::SKTBModelConfig>) {
                    validate_pairwise_model_setup(cfg.native, "SKTB");
                } else if constexpr (std::is_same_v<Config, model::DFTBModelConfig>) {
                    validate_pairwise_model_setup(cfg.native, "DFTB", true);
                }
            },
            model_config);
    }

    backend::OperatorSet& require_operators() {
        if (!operators_) {
            throw std::runtime_error("RSATB is not set up yet. Call setup(...) before computing properties.");
        }
        return *operators_;
    }

    const backend::OperatorSet& require_operators() const {
        if (!operators_) {
            throw std::runtime_error("RSATB is not set up yet. Call setup(...) before computing properties.");
        }
        return *operators_;
    }

    std::unique_ptr<Chebyshev<Scalar>> make_chebyshev(const backend::OperatorSet::KPointOperatorState& state) const {
        Matrix* Hk = state.hamiltonian.get();
        Matrix* Sk = state.overlap.get();
        auto* P = state.preconditioner.get();
        const auto& ops = require_operators();

        if (!Hk) {
            throw std::runtime_error("Hamiltonian is not available for the requested k-point.");
        }

        if (Sk && P) {
            return std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, *P, P->get_type(), ops.emin(), ops.emax(), verbose_);
        }
        if (Sk) {
            return std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, ops.emin(), ops.emax(), verbose_);
        }
        if (P) {
            return std::make_unique<Chebyshev<Scalar>>(*Hk, *P, P->get_type(), ops.emin(), ops.emax(), verbose_);
        }
        return std::make_unique<Chebyshev<Scalar>>(*Hk, ops.emin(), ops.emax(), verbose_);
    }

    static const backend::PositionAxisData& select_axis_data(
        const backend::PositionTriplet& position,
        int axis_idx
    ) {
        if (axis_idx == 0) return position.x;
        if (axis_idx == 1) return position.y;
        if (axis_idx == 2) return position.z;
        throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
    }

    static const backend::FirstLatticeMomentAxisData& select_axis_data(
        const backend::FirstLatticeMomentTriplet& first_lattice_moment,
        int axis_idx
    ) {
        if (axis_idx == 0) return first_lattice_moment.x;
        if (axis_idx == 1) return first_lattice_moment.y;
        if (axis_idx == 2) return first_lattice_moment.z;
        throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
    }

    static int axis_to_index(const std::string& axis) {
        if (axis == "x") return 0;
        if (axis == "y") return 1;
        if (axis == "z") return 2;
        throw std::runtime_error("compute_msd_spectrum axis must be one of: x, y, z");
    }

    static bool is_gamma_kpoint(const std::vector<double>& kpoint, double tol = 1e-12) {
        for (double value : kpoint) {
            if (std::abs(value) > tol) {
                return false;
            }
        }
        return true;
    }

    int select_gamma_kpoint_index(const backend::OperatorSet& ops) const {
        int local_gamma_k_idx = ops.global_nk();
        for (const auto& kp : ops.local_kpoints()) {
            if (is_gamma_kpoint(kp.second)) {
                local_gamma_k_idx = std::min(local_gamma_k_idx, kp.first);
            }
        }

        int global_gamma_k_idx = local_gamma_k_idx;
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(&local_gamma_k_idx, &global_gamma_k_idx, 1, MPI_INT, MPI_MIN, master_comm_);
        }
        if (sub_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(&global_gamma_k_idx, 1, MPI_INT, 0, sub_comm_);
        }

        if (global_gamma_k_idx >= ops.global_nk()) {
            throw std::runtime_error(
                "compute_msd_spectrum requires a Gamma-point in setup.");
        }
        return global_gamma_k_idx;
    }

    MPI_Comm world_comm_;
    MPI_Comm sub_comm_;
    MPI_Comm master_comm_;
    int world_rank_, world_size_;
    int sub_rank_, sub_size_;
    int npar_;
    bool verbose_;
    
    std::unique_ptr<AtomicData> atomic_data_;
    std::unique_ptr<backend::OperatorSet> operators_;
    SamplingState sampling_;
    std::vector<std::string> type_symbols_;
    std::vector<double> r_max_per_type_;
};

} // namespace rsatb

#endif
