#ifndef RSATB_CORE_RSATB_HPP
#define RSATB_CORE_RSATB_HPP

#include "backend/atomic_data.hpp"
#include "backend/block_csr.hpp"
#include "backend/dist_vector.hpp"
#include "backend/dist_multivector.hpp"
#include "backend/operator_construction.hpp"
#include "backend/dbcsr_conversions.hpp"
#include "calc_/eigen.hpp"
#include "calc_/dos.hpp"
#include "calc_/ac.hpp"
#include "calc_/fermi_dirac.hpp"
#include "calc_/isqrtm.hpp"
#include "calc_/preconditioner.hpp"
#include "calc_/solver.hpp" // For LinearSolver
#include "calc_/cheby.hpp"

#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <mpi.h>
#include <dbcsr.h>

namespace rsatb {

using namespace rsatb::backend;
using namespace rsatb::calc;

// Main class for RSATB simulation
// Uses std::complex<double> for all matrices to support k-points and general operators.
class RSATB {
public:
    using Scalar = std::complex<double>;
    using Real = double;
    using Matrix = BlockSpMat<Scalar>;
    using Vector = DistVector<Scalar>;
    using MultiVector = DistMultiVector<Scalar>;

    // Constructor
    RSATB(const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
          const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
          const double *cell, const double *pos,
          MPI_Comm comm = MPI_COMM_WORLD) 
        : comm_(comm) {
        
        MPI_Comm_rank(comm, &rank_);
        
        // Initialize AtomicData
        atomic_data_ = std::make_unique<AtomicData>(
            n_atom, N_atom, atom_offset, n_edge, N_edge,
            atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
            cell, pos, comm
        );

        // Initialize DistGraph
        graph_ = std::make_unique<DistGraph>(comm);
        
        // Initialize OperatorConstruction
        op_construction_ = std::make_unique<OperatorConstruction>(comm);
        
        // Initialize DBCSR library
        int io_unit = (rank_ == 0) ? 6 : 0; // stdout for rank 0
        c_dbcsr_init_lib(comm, &io_unit);
    }
    
    ~RSATB() {
        c_dbcsr_finalize_lib();
    }

    // Initialize OperatorConstruction with basis paths
    void init_basis(size_t ntype, int nspin, bool lspinorb,
                    const std::string& orb_dir, const std::vector<std::string>& orb_name, 
                    const std::string& psd_dir, const std::vector<std::string>& psd_name,
                    const std::string& log_file) {
        op_construction_->init(ntype, nspin, lspinorb, orb_dir, orb_name, psd_dir, psd_name, log_file);
    }

    // Setup K-point
    void setup_K(double Kx, double Ky, double Kz) {
        Kx_ = Kx; Ky_ = Ky; Kz_ = Kz;
        kpoint_ready_ = true;
    }

    // Setup Hamiltonian from HDF5
    void setup_H(const std::string &HFile) {
        if (!kpoint_ready_) throw std::runtime_error("Setup K-point first.");
        
        H_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        rsatb::io::HDF5Handler h5(HFile);
        op_construction_->build_from_hdf5(*H_, *atomic_data_, h5, Kx_, Ky_, Kz_);
        hamiltonian_ready_ = true;
    }

    void setup_S(const std::string &SFile) {
        if (!kpoint_ready_) throw std::runtime_error("Setup K-point first.");
        
        S_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        rsatb::io::HDF5Handler h5(SFile);
        op_construction_->build_from_hdf5(*S_, *atomic_data_, h5, Kx_, Ky_, Kz_);
        overlap_ready_ = true;
    }

    void setup_S() {
        if (!kpoint_ready_) throw std::runtime_error("Setup K-point first.");
        
        S_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        op_construction_->build_overlap_position(*S_, nullptr, nullptr, nullptr, *atomic_data_, Kx_, Ky_, Kz_);
        overlap_ready_ = true;
    }

    // Setup Velocity Operators
    void setup_V() {
        if (!kpoint_ready_) throw std::runtime_error("Setup K-point first.");
        
        // If S is not ready, we build it here too
        if (!S_) S_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        
        Vx_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        Vy_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        Vz_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        
        op_construction_->build_velocity(*S_, Vx_.get(), Vy_.get(), Vz_.get(), *atomic_data_, Kx_, Ky_, Kz_);
        
        overlap_ready_ = true;
        velocity_ready_ = true;
    }

    // Initialize matrices manually (for testing or custom filling)
    void init_matrices_manual() {
        if (!kpoint_ready_) throw std::runtime_error("Setup K-point first.");
        H_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        S_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        hamiltonian_ready_ = true;
        overlap_ready_ = true;
    }

    void isqrtm() {
        if (!overlap_ready_) throw std::runtime_error("S must be ready.");
        
        Sr_ = std::make_unique<Matrix>(atomic_data_.get(), graph_.get());
        
        // Estimate norm/scaling for S
        PrimmeEigenSolver<Scalar> esolver;
        std::vector<double> evals;
        std::vector<Vector> evecs;
        esolver.solve_largest(*S_, nullptr, nullptr, 1, 1e-3, evals, evecs, verbose);
        double max_eig = evals[0];
        double factor = 1.0 / std::sqrt(max_eig + 0.1); 
        
        // Compute Sr = S^-1/2
        rsatb::calc::isqrtm(*S_, *Sr_, tol, 1e-6, maxit, factor, verbose);

        // make Sr a preconditioner
        rsatb::calc::ISQRTMPreconditioner(*Sr_);

        isqrtm_ready_ = true;
    }

    // Orthogonalize H: H' = S^-1/2 H S^-1/2
    // Also transforms V if present.
    void orthogonalize_H(bool explicit = false, bool verbose = false) {
        if (!hamiltonian_ready_ || !overlap_ready_) throw std::runtime_error("H and S must be ready.");
        
        if (!isqrtm_ready_) isqrtm();

        // drop overlap
        free(S_);
        S_ = nullptr;
        
        if (explicit) {
            // Transform H -> Sr * H * Sr
            transform_matrix(*H_, *Sr_);
        }
        
        // Transform V if ready
        if (velocity_ready_) {
            transform_matrix(*Vx_, *Sr_);
            transform_matrix(*Vy_, *Sr_);
            transform_matrix(*Vz_, *Sr_);
        }
        
        orthogonalized_ = true;
    }

    // Setup Random Samples
    void setup_samples(int n) {
        samples_ = std::make_unique<MultiVector>(graph_.get(), n);
        samples_->set_random(); // Gaussian random
    }

    // Compute DOS
    void compute_dos(int nt, int n_proj, const int* orb_indices, double* dos_out, double* pdos_out, double* omega_out, bool verbose = false) {
        if (!scaled_) throw std::runtime_error("Scale and shift H first.");
        if (!samples_) throw std::runtime_error("Setup samples first.");
        
        // Create Chebyshev object
        // If orthogonalized, S is null. If not, S is S_.
        Matrix* S_ptr = orthogonalized_ ? nullptr : S_.get();
        
        // LinearSolver needed if S is present
        std::unique_ptr<LinearSolver<Scalar>> solver;
        if (S_ptr) {
            solver = std::make_unique<LinearSolver<Scalar>>(comm_);
        }
        
        Chebyshev<Scalar> cheby(*H_, S_ptr, Sr_.get(), solver.get(), scale_, shift_);
        
        std::vector<std::complex<double>> cdos;
        std::vector<std::complex<double>> pcdos;
        
        rsatb::calc::corr_dos(cheby, *samples_, nt, cdos, nullptr, nullptr, verbose);
        
        // Output moments (real part)
        for(int i=0; i<=nt; ++i) {
            if (dos_out) dos_out[i] = cdos[i].real();
        }
    }
    
    // Compute AC Conductivity
    void compute_ac(int nt, int n_kpm, double temp, double mu, double* ac_out, double* omega_out) {
        if (!scaled_) throw std::runtime_error("Scale and shift H first.");
        if (!velocity_ready_) throw std::runtime_error("Setup velocity first.");
        
        Matrix* S_ptr = orthogonalized_ ? nullptr : S_.get();
        std::unique_ptr<LinearSolver<Scalar>> solver;
        if (S_ptr) {
            solver = std::make_unique<LinearSolver<Scalar>>(comm_);
        }
        
        Chebyshev<Scalar> cheby(*H_, S_ptr, Sr_.get(), solver.get(), scale_, shift_);
        
        std::vector<std::complex<double>> cac((nt + 1) * 6);
        
        rsatb::calc::corr_ac(cheby, *Vx_, *Vy_, *Vz_, *samples_, nt, temp, mu, scale_, shift_, n_kpm, 1e-14, cac.data(), true);
        
        // Copy raw correlation (real part)
         for(int i=0; i<(nt+1)*6; ++i) {
            if (ac_out) ac_out[i] = cac[i].real();
        }
    }

    // Compute Eigenvalues
    void compute_eigenvalues(int num, double tol, std::vector<double>& evals, bool verbose = false) {
        PrimmeEigenSolver<Scalar> esolver;
        std::vector<Vector> evecs;
        esolver.solve(*H_, nullptr, nullptr, num, primme_smallest, tol, evals, evecs, verbose);
    }

    // Getters
    Matrix& H() { return *H_; }
    Matrix& S() { return *S_; }
    AtomicData& atomic_data() { return *atomic_data_; }

private:
    // Helper to transform M -> Z^H * M * Z using DBCSR
    void transform_matrix(Matrix& M, Matrix& Z) {
        dbcsr_matrix dbcsr_M, dbcsr_Z, dbcsr_tmp, dbcsr_res;
        
        // Convert to DBCSR
        // Assume Hermitian M for optimization if possible, but general works
        BlockSpMat2DBCSR(M, &dbcsr_M, 'N');
        BlockSpMat2DBCSR(Z, &dbcsr_Z, 'N');
        
        // Create temps
        dbcsr_distribution dist;
        c_dbcsr_get_distribution(dbcsr_M, &dist);
        int NGlobal = c_dbcsr_nblkrows_total(dbcsr_M);
        std::vector<int> blksize(NGlobal);
        c_dbcsr_get_col_blk_size(dbcsr_M, blksize.data(), NGlobal);
        int data_type = dbcsr_type_complex_8;
        bool mutable_work = true;
        
        c_dbcsr_create_new(&dbcsr_tmp, "tmp", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
        c_dbcsr_finalize(dbcsr_tmp);
        c_dbcsr_create_new(&dbcsr_res, "res", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
        c_dbcsr_finalize(dbcsr_res);
        
        // tmp = M * Z
        double alpha = 1.0, beta = 0.0;
        long long cflop = 0;
        double sptol = 1e-6;
        bool retain = false;
        
        c_dbcsr_multiply_z('N', 'N', std::complex<double>(1.0), dbcsr_M, dbcsr_Z, std::complex<double>(0.0), dbcsr_tmp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain, &sptol, &cflop);
        
        // res = Z^H * tmp
        c_dbcsr_multiply_z('C', 'N', std::complex<double>(1.0), dbcsr_Z, dbcsr_tmp, std::complex<double>(0.0), dbcsr_res, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain, &sptol, &cflop);
        
        // Convert back to M
        // This replaces M's content
        DBCSR2BlockSpMat(dbcsr_res, M);
        
        c_dbcsr_release(&dbcsr_M);
        c_dbcsr_release(&dbcsr_Z);
        c_dbcsr_release(&dbcsr_tmp);
        c_dbcsr_release(&dbcsr_res);
    }

    MPI_Comm comm_;
    int rank_;
    
    std::unique_ptr<AtomicData> atomic_data_;
    std::unique_ptr<DistGraph> graph_;
    std::unique_ptr<OperatorConstruction> op_construction_;
    
    std::unique_ptr<Matrix> H_, S_, Sr_;
    std::unique_ptr<Matrix> Rx_, Ry_, Rz_;
    std::unique_ptr<Matrix> Vx_, Vy_, Vz_;
    
    std::unique_ptr<MultiVector> samples_;
    
    double Kx_ = 0, Ky_ = 0, Kz_ = 0;
    double scale_ = 1.0, shift_ = 0.0;
    double emin_ = 0.0, emax_ = 0.0;
    
    bool kpoint_ready_ = false;
    bool hamiltonian_ready_ = false;
    bool overlap_ready_ = false;
    bool velocity_ready_ = false;
    bool orthogonalized_ = false;
    bool scaled_ = false;
};

} // namespace rsatb

#endif
