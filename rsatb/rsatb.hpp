#ifndef RSATB_CORE_RSATB_HPP
#define RSATB_CORE_RSATB_HPP

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "backend/operator_construction.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "calc/eigen.hpp"
#include "calc/dos.hpp"
#include "calc/ac.hpp"
#include "calc/fermi_dirac.hpp"
#include "calc/isqrtm.hpp"
#include "calc/preconditioner.hpp"
#include "calc/solver.hpp" 
#include "calc/cheby.hpp"
#include "calc/kpm.hpp"
#include "calc/fft.hpp"

#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <map>
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

    // Constructors
    RSATB(int npar,
          const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
          const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
          const double *cell, const double *pos,
          bool verbose = false,
          MPI_Comm comm = MPI_COMM_WORLD) 
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        
        // Initialize threading for BLAS kernels
        Matrix::KernelType::init_threading();

        init_comms();
        atomic_data_ = std::make_unique<AtomicData>(
            n_atom, N_atom, atom_offset, n_edge, N_edge,
            atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
            cell, pos, sub_comm_
        );
        init_components();
    }

    std::string get_kernel_info() const {
        return Matrix::KernelType::name();
    }

    RSATB(int npar, const std::string& filename, const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb, bool verbose = false, MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        Matrix::KernelType::init_threading();
        init_comms();
        atomic_data_ = std::unique_ptr<AtomicData>(AtomicData::from_file(filename, r_max_per_type, type_norb, sub_comm_));
        init_components();
    }

    RSATB(int npar, const std::vector<double>& pos, const std::vector<int>& z, 
          const std::vector<double>& cell, const std::vector<bool>& pbc, 
          const std::vector<double>& r_max_per_type, const std::vector<int>& type_norb, bool verbose = false, MPI_Comm comm = MPI_COMM_WORLD)
        : world_comm_(comm), npar_(npar), verbose_(verbose) {
        Matrix::KernelType::init_threading();
        init_comms();
        atomic_data_ = std::unique_ptr<AtomicData>(AtomicData::from_points(pos, z, cell, pbc, r_max_per_type, type_norb, sub_comm_));
        init_components();
    }
    
    ~RSATB() {
        // Destroy components that might hold DBCSR objects before finalization
        H_images_.reset();
        S_images_.reset();
        H_k_map_.clear();
        S_k_map_.clear();
        precond_map_.clear();
        sample_vec_.reset();
        sample_multi_vec_.reset();
        op_construction_.reset();
        atomic_data_.reset();
        
        // c_dbcsr_finalize_lib();
        if (sub_comm_ != MPI_COMM_NULL) MPI_Comm_free(&sub_comm_);
        if (master_comm_ != MPI_COMM_NULL) MPI_Comm_free(&master_comm_);
    }



    // Setup Function
    void setup(const std::string& Hfile, const std::vector<std::vector<double>>& kpoints,
               const std::vector<std::string>& orb_names, const std::string& orb_dirs,
               const std::vector<std::string>& psd_names, const std::string& psd_dirs,
               double tol=1e-7, double sptol=1e-5, int maxiter=20, const std::string& method="graph",
               const std::string& precond = "H/S", int nsamples = 1, bool compute_bounds = true) {
        
        nsamples_ = nsamples;
        global_nk_ = kpoints.size();

        // 1. Setup H/S Images
        if (!Hfile.empty()) {
            rsatb::io::HDF5Handler h5(Hfile);
            H_images_ = op_construction_->build_from_hdf5(*atomic_data_, h5);
        }
        
        if (!orb_names.empty() && !orb_dirs.empty()) {
             int ntype = atomic_data_->n_type();
             op_construction_->init(ntype, 1, false, orb_dirs, orb_names, psd_dirs, psd_names, "op2c.log");
             S_images_ = op_construction_->build_overlap(*atomic_data_);
        }

        // 2. K-point Partition
        int my_color = (world_rank_ * npar_) / world_size_;
        my_kpoints_.clear();
        for (size_t i = 0; i < kpoints.size(); ++i) {
            if (i % npar_ == my_color) {
                my_kpoints_.push_back({static_cast<int>(i), kpoints[i]});
            }
        }
        
        // 3. Loop K-points
        double local_emin = 1e30, local_emax = -1e30;
        
        for (const auto& kp : my_kpoints_) {
            int k_idx = kp.first;
            const std::vector<double>& K = kp.second;
            
            if (H_images_) {
                H_k_map_[k_idx] = std::unique_ptr<Matrix>(H_images_->sample_k(K, PhaseConvention::R_ONLY));
            }
            
            Matrix* Sk = nullptr;
            if (S_images_) {
                S_k_map_[k_idx] = std::unique_ptr<Matrix>(S_images_->sample_k(K, PhaseConvention::R_ONLY));
                Sk = S_k_map_[k_idx].get();
            }
            
            // Compute Preconditioner
            if (Sk) {
                PrimmeEigenSolver<Scalar, Matrix::KernelType> esolver;
                std::vector<double> evals;
                std::vector<Vector> evecs;
                
                // esolver.solve_smallest(*Sk, nullptr, nullptr, 1, 1e-3, evals, evecs, false);
                double max_eig = 1;
                // std::cout << "max_eig = " << max_eig << std::endl;
                esolver.solve_largest(*Sk, nullptr, nullptr, 1, 1e-3, evals, evecs, verbose_);
                max_eig = evals[0];
                
                double factor = 1.0 / std::sqrt(max_eig + 0.1);
                
                if (precond == "H" || precond == "S"){
                    auto P = std::make_unique<ISQRTMPreconditioner<Scalar>>(*Sk, maxiter, tol, sptol, method, factor, verbose_);
                    precond_map_[k_idx] = std::move(P);
                } else {
                    // assert precond is None
                    if (precond != "None") {
                        throw std::runtime_error("Preconditioner support H, S or None");
                    }
                }
                
                if (precond == "H") {
                    S_k_map_.erase(k_idx);
                    Sk = nullptr;
                }
            }
            
            // Compute eigenvalues of H
            if (!Hfile.empty() && compute_bounds) {
                
                PrimmeEigenSolver<Scalar> esolver;
                std::vector<double> evals_min, evals_max;
                std::vector<Vector> evecs;
                
                Matrix* Hk = H_k_map_[k_idx].get();
                Preconditioner<Scalar>* P = (precond_map_.count(k_idx)) ? precond_map_[k_idx].get() : nullptr;
                
                esolver.solve(*Hk, Sk, P, 1, primme_smallest, 1e-3, evals_min, evecs, verbose_, P->get_type());
                esolver.solve(*Hk, Sk, P, 1, primme_largest, 1e-3, evals_max, evecs, verbose_, P->get_type()); // eigensolver will automatically switch type according to Sk and P's value.
                
                if (!evals_min.empty()) local_emin = std::min(local_emin, evals_min[0]);
                if (!evals_max.empty()) local_emax = std::max(local_emax, evals_max[0]);

            }
        }
        
        if (!Hfile.empty() && compute_bounds) {
            // 4. Global Reduction for emin/emax
            double global_emin = local_emin;
            double global_emax = local_emax;
            
            if (master_comm_ != MPI_COMM_NULL) {
                MPI_Allreduce(&local_emin, &global_emin, 1, MPI_DOUBLE, MPI_MIN, master_comm_);
                MPI_Allreduce(&local_emax, &global_emax, 1, MPI_DOUBLE, MPI_MAX, master_comm_);
            }
            
            MPI_Bcast(&global_emin, 1, MPI_DOUBLE, 0, sub_comm_);
            MPI_Bcast(&global_emax, 1, MPI_DOUBLE, 0, sub_comm_);

            emin_ = global_emin - 0.5; // Buffer
            emax_ = global_emax + 0.5;
        }
            
        // 5. Initialize Random Samples
        if (nsamples_ == 1) {
            sample_vec_ = std::make_unique<Vector>(atomic_data_->get_graph());
            sample_vec_->set_random_normal(true);

            sample_multi_vec_ = nullptr;
        } else {
            sample_multi_vec_ = std::make_unique<MultiVector>(atomic_data_->get_graph(), nsamples_);
            sample_multi_vec_->set_random_normal(true);
            sample_vec_ = nullptr;
        }
    }

    void test_solve() {
        LinearSolver<Scalar, Matrix::KernelType> solver(1e-8, 1000, verbose_);
        for (const auto& kp : my_kpoints_) {
            int k_idx = kp.first;
            Preconditioner<Scalar>* P = (precond_map_.count(k_idx)) ? precond_map_[k_idx].get() : nullptr;
            Matrix* Sk = (S_k_map_.count(k_idx)) ? S_k_map_[k_idx].get() : nullptr;
            if (nsamples_ == 1) {
                Vector y = sample_vec_->duplicate();
                solver.solve(
                    *Sk, *sample_vec_, y, *P, P->get_type()
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
        if (emin_ > emax_) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }

        std::vector<double> local_dos(global_nk_ * 2 * nt, 0.0);
        
        std::complex<double> imI(0.0, 1.0);
        double scale = (emax_ - emin_) / 2.0;
        double shift = (emax_ + emin_) / 2.0;
        
        for (const auto& kp : my_kpoints_) {
            int k_idx = kp.first;
            Matrix* Hk = H_k_map_[k_idx].get();
            Matrix* Sk = (S_k_map_.count(k_idx)) ? S_k_map_[k_idx].get() : nullptr;
            Preconditioner<Scalar>* P = (precond_map_.count(k_idx)) ? precond_map_[k_idx].get() : nullptr;
            
            std::unique_ptr<Chebyshev<Scalar>> cheby;
            if (Sk && P) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, *P, P->get_type(), emin_, emax_, verbose_);
            } else if (Sk) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, emin_, emax_, verbose_);
            } else if (P) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *P, P->get_type(), emin_, emax_, verbose_);
            } else {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, emin_, emax_, verbose_);
            }
            
            std::vector<std::complex<double>> cdos;
            const Vector* null_vec = nullptr;
            std::vector<std::complex<double>>* null_res = nullptr;
            
            if (nsamples_ == 1) {
                rsatb::calc::corr_dos(*cheby, *sample_vec_, nt, cdos, null_vec, null_res, verbose_);
            } else {
                rsatb::calc::corr_dos(*cheby, *sample_multi_vec_, nt, cdos, null_vec, null_res, verbose_);
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
        dos_out.assign(global_nk_ * 2 * nt, 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_dos.data(), dos_out.data(), local_dos.size(), MPI_DOUBLE, MPI_SUM, master_comm_);
        }
        MPI_Bcast(dos_out.data(), dos_out.size(), MPI_DOUBLE, 0, sub_comm_);
        
        // Omega is same for all K-points
        omega_out.resize(2 * nt);
        for(int i=0; i<nt; ++i) {
            omega_out[i] = (i - nt) * scale / nt + shift;
            omega_out[nt + i] = i * scale / nt + shift;
        }
    }

    void compute_ac(int nt, int n_kpm, double temp, double mu, std::vector<double>& ac_out, std::vector<double>& omega_out, std::string axis="ab") {
        // Output size: global_nk * (nt+1) * 6
        if (emin_ > emax_) {
            throw std::runtime_error("Spectral bound is not computed yet. Set it true during Setup stage for random state method support.");
        }
        
        std::vector<double> local_ac(global_nk_ * (nt + 1) * 6, 0.0);
        
        double scale = (emax_ - emin_) / 2.0;
        double shift = (emax_ + emin_) / 2.0;
        double beta = 1.0 / (temp * 8.617333262e-5);
        double ac_prefactor = (double)atomic_data_->norb() / atomic_data_->volume(axis);
        double t_step = M_PI / scale;
        
        // Precompute omega
        std::vector<double> omega(nt + 1);
        for(int i=0; i<=nt; ++i) {
             omega[i] = i * scale / (nt + 1);
        }

        //check if orthogonal, if not
            // get three-body graph
        DistGraph* graphv;
        if (S_images_) {
            int ntype = atomic_data_->n_type();
            std::vector<double> orbrmax(ntype, 0.0), psdrmax(ntype, 0.0);
            for(int i=0; i<ntype; ++i) {
                orbrmax[i] = op_construction_->op2c_->tcbd.orb_->rcut_max(i);
                psdrmax[i] = op_construction_->op2c_->tcbd.beta_->rcut_max(i);
            }
            graphv = atomic_data_->get_graph3b(orbrmax, psdrmax);
        } else {
            graphv = atomic_data_->get_graph();
        }
        
        for (const auto& kp : my_kpoints_) {
            int k_idx = kp.first;
            const std::vector<double>& K = kp.second;
            
            Matrix* Hk = H_k_map_[k_idx].get();
            Matrix* Sk = (S_k_map_.count(k_idx)) ? S_k_map_[k_idx].get() : nullptr;
            Preconditioner<Scalar>* P = (precond_map_.count(k_idx)) ? precond_map_[k_idx].get() : nullptr;
            
            auto Vx = std::make_unique<Matrix>(graphv);
            auto Vy = std::make_unique<Matrix>(graphv);
            auto Vz = std::make_unique<Matrix>(graphv);
            
            if (S_images_) {
                auto img_V = op_construction_->build_overlap_velocity(Vx.get(), Vy.get(), Vz.get(), *atomic_data_, K[0], K[1], K[2]);
            } else {
                if (world_rank_ == 0) {
                    std::cout << "Orthogonal Velocity Matrix is not supported yet, but will be done very soon. Since it is easy.";
                }
            }

            Vx->assemble();
            Vy->assemble();
            Vz->assemble();
            
            std::unique_ptr<Chebyshev<Scalar>> cheby;
            if (Sk && P) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, *P, P->get_type(), emin_, emax_, verbose_);
            } else if (Sk) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *Sk, emin_, emax_, verbose_);
            } else if (P) {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, *P, P->get_type(), emin_, emax_, verbose_);
            } else {
                cheby = std::make_unique<Chebyshev<Scalar>>(*Hk, emin_, emax_, verbose_);
            }
            
            std::vector<std::complex<double>> cac;
            
            if (nsamples_ == 1) {
                rsatb::calc::corr_ac(*cheby, *Vx, *Vy, *Vz, *sample_vec_, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, verbose_);
            } else {
                rsatb::calc::corr_ac(*cheby, *Vx, *Vy, *Vz, *sample_multi_vec_, nt, temp, mu, scale, shift, n_kpm, 1e-14, &cac, verbose_);
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
                        val = ac_prefactor * t_step * val * (std::exp(-beta * omega[iw]) - 1) / omega[iw];
                    }
                    local_ac[idx] = val;
                }
            }
        }
        
        // Gather
        ac_out.assign(global_nk_ * (nt + 1) * 6, 0.0);
        if (master_comm_ != MPI_COMM_NULL) {
            MPI_Allreduce(local_ac.data(), ac_out.data(), local_ac.size(), MPI_DOUBLE, MPI_SUM, master_comm_);
        }
        MPI_Bcast(ac_out.data(), ac_out.size(), MPI_DOUBLE, 0, sub_comm_);
        
        omega_out = omega;
    }

    // Test Helpers
    void set_H_images(std::unique_ptr<ImageContainer<double>> img) {
        H_images_ = std::move(img);
    }
    
    void set_S_images(std::unique_ptr<ImageContainer<double>> img) {
        S_images_ = std::move(img);
    }
    
    Matrix* get_H_k(int k_idx) {
        if (H_k_map_.count(k_idx)) return H_k_map_[k_idx].get();
        return nullptr;
    }

    OperatorConstruction* get_operator_construction() {
        return op_construction_.get();
    }
    
    AtomicData* atomic_data() {
        return atomic_data_.get();
    }

private:
    void init_comms() {
        MPI_Comm_rank(world_comm_, &world_rank_);
        MPI_Comm_size(world_comm_, &world_size_);

        if (npar_ > world_size_) npar_ = world_size_;

        int color = (world_rank_ * npar_) / world_size_;
        std::cerr << "Rank " << world_rank_ << " init_comms: npar=" << npar_ << " size=" << world_size_ << " color=" << color << std::endl;
        MPI_Comm_split(world_comm_, color, world_rank_, &sub_comm_);
        
        MPI_Comm_rank(sub_comm_, &sub_rank_);
        MPI_Comm_size(sub_comm_, &sub_size_);
        std::cerr << "Rank " << world_rank_ << " sub_comm size=" << sub_size_ << std::endl;
        
        int master_color = (sub_rank_ == 0) ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(world_comm_, master_color, world_rank_, &master_comm_);
    }

    void init_components() {
        op_construction_ = std::make_unique<OperatorConstruction>(sub_comm_);
        
        // int color = (world_rank_ * npar_) / world_size_;
        // int io_unit = (sub_rank_ == 0 && color == 0) ? 6 : 0; 
        // c_dbcsr_init_lib(sub_comm_, &io_unit);
    }

    MPI_Comm world_comm_;
    MPI_Comm sub_comm_;
    MPI_Comm master_comm_;
    int world_rank_, world_size_;
    int sub_rank_, sub_size_;
    int npar_;
    bool verbose_;
    
    std::unique_ptr<AtomicData> atomic_data_;
    std::unique_ptr<OperatorConstruction> op_construction_;
    
    std::unique_ptr<ImageContainer<double>> H_images_ = nullptr;
    std::unique_ptr<ImageContainer<double>> S_images_ = nullptr;
    
    std::vector<std::pair<int, std::vector<double>>> my_kpoints_;
    
    std::map<int, std::unique_ptr<Matrix>> H_k_map_;
    std::map<int, std::unique_ptr<Matrix>> S_k_map_;
    std::map<int, std::unique_ptr<Preconditioner<Scalar>>> precond_map_;
    
    double emin_ = -1.0;
    double emax_ = 1.0;
    
    int nsamples_ = 1;
    int global_nk_ = 0;
    std::unique_ptr<Vector> sample_vec_;
    std::unique_ptr<MultiVector> sample_multi_vec_;
};

} // namespace rsatb

#endif