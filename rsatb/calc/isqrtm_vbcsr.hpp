#ifndef RSATB_CALC_ISQRTM_VBCSR_HPP
#define RSATB_CALC_ISQRTM_VBCSR_HPP

#include "vbcsr/core/block_csr.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <set>

// LAPACK Declarations
extern "C" {
    // Double precision symmetric eigendecomposition
    void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info);
    
    // Complex double Hermitian eigendecomposition
    void zheev_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info);
}

namespace rsatb {
namespace calc {

using namespace vbcsr;

// Helper to compute M^{-1/2} for a dense matrix M
template <typename T>
void dense_isqrt(int n, std::vector<T>& M) {
    if (n == 0) return;
    
    std::vector<double> w(n); // Eigenvalues
    int info;
    int lwork = -1;
    
    if constexpr (std::is_same<T, double>::value) {
        double wkopt;
        dsyev_("V", "U", &n, (double*)M.data(), &n, w.data(), &wkopt, &lwork, &info);
        lwork = (int)wkopt;
        std::vector<double> work(lwork);
        dsyev_("V", "U", &n, (double*)M.data(), &n, w.data(), work.data(), &lwork, &info);
    } else {
        std::complex<double> wkopt;
        std::vector<double> rwork(3 * n - 2);
        zheev_("V", "U", &n, (std::complex<double>*)M.data(), &n, w.data(), &wkopt, &lwork, rwork.data(), &info);
        lwork = (int)wkopt.real();
        std::vector<std::complex<double>> work(lwork);
        zheev_("V", "U", &n, (std::complex<double>*)M.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &info);
    }
    
    if (info != 0) {
        std::cerr << "Error in LAPACK eigendecomposition (info=" << info << ")" << std::endl;
        return; // Handle error?
    }
    
    // M now contains eigenvectors V. w contains eigenvalues.
    // M^{-1/2} = V * diag(w^{-1/2}) * V^H
    
    // 1. Scale columns of V by w^{-1/4}
    for (int j = 0; j < n; ++j) {
        double val = (w[j] > 1e-12) ? std::pow(w[j], -0.25) : 0.0;
        for (int i = 0; i < n; ++i) {
            M[i + j * n] *= val;
        }
    }
    
    // 2. Compute Res = M * M^H
    std::vector<T> Res(n * n, T(0));
    
    if constexpr (std::is_same<T, double>::value) {
        vbcsr::BLASKernel::gemm(n, n, n, 1.0, (double*)M.data(), n, (double*)M.data(), n, 0.0, (double*)Res.data(), n, vbcsr::CblasNoTrans, vbcsr::CblasConjTrans); // ConjTrans for double is Trans
    } else {
        vbcsr::BLASKernel::gemm(n, n, n, std::complex<double>(1.0), (std::complex<double>*)M.data(), n, (std::complex<double>*)M.data(), n, std::complex<double>(0.0), (std::complex<double>*)Res.data(), n, vbcsr::CblasNoTrans, vbcsr::CblasConjTrans);
    }
    
    M = Res;
}

// Matrix Inverse Square Root using Newton-Schulz Iteration
// Computes S^{-1/2}
template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>* S_inv_sqrt, 
            double tol = 1e-7, double sptol = 1e-6, int maxit = 20, 
            bool verbose = false) {
    
    DistGraph* graph = S_mat.graph;
    int rank = graph->rank;

    // 1. Initialize Z0 (S_inv_sqrt)
    if (rank == 0 && verbose) std::cout << "isqrtm: Initializing Z0 using subgraph method..." << std::endl;

    // check S_inv_sqrt allocation
    if (S_inv_sqrt->val.size() == 0 || S_inv_sqrt == nullptr || S_inv_sqrt == NULL) {
        *S_inv_sqrt = S_mat.duplicate();
    }
    
    // Clear S_inv_sqrt
    std::fill(S_inv_sqrt->val.begin(), S_inv_sqrt->val.end(), T(0));
    
    int n_owned = graph->owned_global_indices.size();
    
    // Iterate over local rows
    for (int i = 0; i < n_owned; ++i) {
        int global_row = graph->owned_global_indices[i];
        
        // Identify Neighborhood C_i
        std::vector<int> neighbors;
        int start = S_mat.row_ptr[i];
        int end = S_mat.row_ptr[i+1];
        for (int k = start; k < end; ++k) {
            int col_lid = S_mat.col_ind[k];
            int col_gid = graph->get_global_index(col_lid);
            neighbors.push_back(col_gid);
        }
        // Ensure global_row is in neighbors (diagonal)
        bool has_diag = false;
        for(int gid : neighbors) if(gid == global_row) has_diag = true;
        if(!has_diag) neighbors.push_back(global_row);
        
        // Sort for consistency
        std::sort(neighbors.begin(), neighbors.end());
        
        // Extract Submatrix
        BlockSpMat<T, Kernel> sub_mat = S_mat.extract_submatrix(neighbors);
        
        // Convert to Dense
        std::vector<T> M = sub_mat.to_dense();
        
        // M is row-major, size (total_dim) x (total_dim)
        int total_dim = 0;
        for(size_t k=0; k<neighbors.size(); ++k) total_dim += sub_mat.graph->block_sizes[k];
        
        // Compute M^{-1/2}
        dense_isqrt(total_dim, M);
        
        // Insert Row corresponding to global_row into S_inv_sqrt
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), global_row);
        int block_idx = std::distance(neighbors.begin(), it);
        
        // Calculate offset in dense matrix M
        int row_offset = 0;
        for(int k=0; k<block_idx; ++k) row_offset += sub_mat.graph->block_sizes[k];
        int r_dim = sub_mat.graph->block_sizes[block_idx];
        
        // Iterate over columns (neighbors)
        int col_offset = 0;
        for(size_t k=0; k<neighbors.size(); ++k) {
            int col_gid = neighbors[k];
            int c_dim = sub_mat.graph->block_sizes[k];
            
            // Extract block (r_dim x c_dim) from M at (row_offset, col_offset)
            std::vector<T> block_data(r_dim * c_dim);
            for(int c=0; c<c_dim; ++c) {
                for(int r=0; r<r_dim; ++r) {
                    // M is RowMajor
                    block_data[c * r_dim + r] = M[(col_offset + c) * total_dim + (row_offset + r)];
                }
            }
            
            // Insert
            S_inv_sqrt->add_block(global_row, col_gid, block_data.data(), r_dim, c_dim, AssemblyMode::ADD, MatrixLayout::ColMajor);
            
            col_offset += c_dim;
        }
    }
    
    S_inv_sqrt->assemble();
    // S_inv_sqrt.shift(T(factor));
    
    // 2. Newton-Schulz Iteration
    // BlockSpMat<T, Kernel> Z = S_inv_sqrt.duplicate(); // Working copy
    BlockSpMat<T, Kernel> tmp1(graph), tmp2(graph);
    
    // Initial Check
    // tmp2 = Z^H * S
    tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false); // Z^T * S
    // tmp2 = tmp2 * Z
    tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);
    // tmp2 = tmp2 - I
    tmp2.shift(T(-1.0));

    // Z, X, XX
    
    // Norm
    double norm = 0.0;
    {
        const auto& norms = tmp2.get_block_norms();
        for(double n : norms) norm = std::max(norm, n);
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        norm = global_norm;
    }
    if (rank == 0) std::cout << "isqrtm: Initial norm=" << norm << std::endl;
    
    // Restore tmp2
    tmp2.shift(T(1.0));
    
    double norm_pre = 0.0;
    int count = 0;
    
    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        if (rank == 0 && verbose) {
            std::cout << "Recursive Factorization of S: current error: maxabs norm - " << norm << " with density " << S_inv_sqrt->get_block_density() << std::endl;
        }
        
        if (norm > 100.0) {
            if (rank == 0) std::cout << "isqrtm: DIVERGENCE DETECTED! norm=" << norm << std::endl;
            break;
        }
        
        norm_pre = norm;
        
        // tmp1 = tmp2 * tmp2
        tmp1 = tmp2.spmm(tmp2, sptol);
        
        // tmp1 = 0.375 * tmp1 - 1.25 * tmp2 + 1.875 * I
        tmp1.axpby(T(-1.25), tmp2, T(0.375));
        tmp1.shift(T(1.875));
        tmp1.filter_blocks(sptol);
        
        // Z = Z * tmp1
        *S_inv_sqrt = S_inv_sqrt->spmm(tmp1, sptol);
        
        // --- Debug Z Hermitian ---
        if (verbose) {
            BlockSpMat<T, Kernel> Z_H = S_inv_sqrt->transpose();
            BlockSpMat<T, Kernel> diff_Z = S_inv_sqrt->duplicate();
            diff_Z.axpby(T(-1.0), Z_H, T(1.0));
            double d_norm = 0.0;
            const auto& norms = diff_Z.get_block_norms();
            for(double n : norms) d_norm = std::max(d_norm, n);
            double global_d_norm;
            MPI_Allreduce(&d_norm, &global_d_norm, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
            if (rank == 0) std::cout << "DEBUG: Z Hermitian Diff: " << global_d_norm << std::endl;
        }
        // -------------------------
        // *S_inv_sqrt = S_inv_sqrt->spmm(S_inv_sqrt->transpose(), sptol);
        // S_inv_sqrt->axpby(T(0.5), S_inv_sqrt->transpose(), T(0.5));

        // Approximate update?
        // tmp3 = tmp1^H * tmp2
        // tmp2 = tmp3 * tmp1
        if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
            tmp2 = tmp1.spmm(tmp2, sptol, true, false);
            tmp2 = tmp2.spmm(tmp1, sptol, false, false);

            if (rank == 0 && verbose) {
                std::cout << "Approximate update: " << std::endl;
            }
        } else {
            // Update tmp2 (X_{k+1})
            tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false);
            tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);

            if (rank == 0 && verbose) {
                std::cout << "Exact update" << std::endl;
            }
        }
        
        tmp2.shift(T(-1.0));
        
        // Norm
        norm = 0.0;
        {
            const auto& norms = tmp2.get_block_norms();
            for(double n : norms) norm = std::max(norm, n);
            double global_norm;
            MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
            norm = global_norm;
        }
        
        tmp2.shift(T(1.0));
        
        count++;
    }
    
    if (rank == 0 && verbose) {
        std::cout << "Recursive Factorization of S: Converge at maxabs norm - " << norm << " with density " << S_inv_sqrt->get_block_density() << std::endl;
    }
}

} // namespace calc
} // namespace rsatb

#endif
