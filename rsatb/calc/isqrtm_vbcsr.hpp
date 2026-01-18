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
#if !defined(RSATB_USE_MKL) && !defined(VBCSR_USE_MKL)
extern "C" {
    // Double precision symmetric eigendecomposition
    void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info);
    
    // Complex double Hermitian eigendecomposition
    void zheev_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info);

}
#endif

#if defined(RSATB_USE_OPENBLAS) || defined(VBCSR_USE_OPENBLAS)
extern "C" void openblas_set_num_threads(int num_threads);
#endif

#if defined(RSATB_USE_MKL) || defined(VBCSR_USE_MKL)
extern "C" {
    void mkl_set_num_threads_local_(int* num_threads);
}
#endif

using namespace vbcsr;

namespace rsatb {
namespace calc {

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
        #if defined(RSATB_USE_MKL) || defined(VBCSR_USE_MKL)
        MKL_Complex16 wkopt_mkl;
        std::vector<double> rwork(3 * n - 2);
        zheev_("V", "U", &n, reinterpret_cast<MKL_Complex16*>(M.data()), &n, w.data(), &wkopt_mkl, &lwork, rwork.data(), &info);
        lwork = (int)wkopt_mkl.real;
        std::vector<MKL_Complex16> work(lwork);
        zheev_("V", "U", &n, reinterpret_cast<MKL_Complex16*>(M.data()), &n, w.data(), work.data(), &lwork, rwork.data(), &info);
        #else
        std::complex<double> wkopt;
        std::vector<double> rwork(3 * n - 2);
        zheev_("V", "U", &n, (std::complex<double>*)M.data(), &n, w.data(), &wkopt, &lwork, rwork.data(), &info);
        lwork = (int)wkopt.real();
        std::vector<std::complex<double>> work(lwork);
        zheev_("V", "U", &n, (std::complex<double>*)M.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &info);
        #endif
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
        vbcsr::BLASKernel::gemm(n, n, n, 1.0, (double*)M.data(), n, (double*)M.data(), n, 0.0, (double*)Res.data(), n, CblasNoTrans, CblasConjTrans); // ConjTrans for double is Trans
    } else {
        vbcsr::BLASKernel::gemm(n, n, n, std::complex<double>(1.0), (std::complex<double>*)M.data(), n, (std::complex<double>*)M.data(), n, std::complex<double>(0.0), (std::complex<double>*)Res.data(), n, CblasNoTrans, CblasConjTrans);
    }
    
    M = Res;
}


template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm_init(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>* S_inv_sqrt, bool verbose = false) {
    
    DistGraph* graph = S_mat.graph;
    int rank = graph->rank;
    int size = graph->size;
    MPI_Comm comm = graph->comm;

    // 1. Initialize Z0 (S_inv_sqrt)
    if (rank == 0 && verbose) std::cout << "isqrtm: Initializing Z0 using subgraph method..." << std::endl;

    // check S_inv_sqrt allocation
    if (S_inv_sqrt == nullptr || S_inv_sqrt == NULL) {
        *S_inv_sqrt = S_mat.duplicate();
    }
    
    // Clear S_inv_sqrt
    S_inv_sqrt->fill(T(0));
    
    int n_owned = graph->owned_global_indices.size();
    int n_owned_max = 0;
    MPI_Allreduce(&n_owned, &n_owned_max, 1, MPI_INT, MPI_MAX, comm);
    
    // batch size
    int batch_size = std::max(1, omp_get_max_threads() / size);
    
    int nbatch = n_owned_max / batch_size;
    if (n_owned_max % batch_size != 0) nbatch++;

    std::vector<std::vector<int>> batch_indices(batch_size);

    for (int b = 0; b < nbatch; ++b) {
        // Avoid thread oversubscription in BLAS/LAPACK
        #ifdef RSATB_USE_MKL
        int one = 1;
        mkl_set_num_threads_local_(&one);
        #elif defined(RSATB_USE_OPENBLAS)
        openblas_set_num_threads(1);
        #endif

        batch_indices.clear();
        batch_indices.resize(batch_size);
        
        // Store neighbors for parallel phase
        std::vector<std::vector<int>> batch_neighbors(batch_size);

        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            
            if (idx < n_owned) {
                int global_row = graph->owned_global_indices[idx];
                // Identify Neighborhood C_i
                std::vector<int> neighbors;
                int start = S_mat.row_ptr[idx]; // Use local index idx
                int end = S_mat.row_ptr[idx+1];
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

                batch_indices[i] = neighbors;
            } else {
                // Padding: Empty request
                batch_indices[i] = {};
            }
        }
        
        auto batch_blocks = S_mat.fetch_blocks(batch_indices);

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            if (idx >= n_owned) continue;
            
            int global_row = graph->owned_global_indices[idx];
            const auto& neighbors = batch_indices[i];

            BlockSpMat<T, Kernel> sub_mat = S_mat.construct_submatrix(neighbors, batch_blocks);

            // Convert to Dense
            std::vector<T> M = sub_mat.to_dense();
        
            // M is row-major, size (total_dim) x (total_dim)
            int total_dim = 0;
            for(size_t k=0; k<neighbors.size(); ++k) total_dim += sub_mat.graph->block_sizes[k];

            if (rank==0 || verbose) {
                std::cout << "working on atom: "<< idx << " total dim: " << total_dim << std::endl;
            }
            
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
                
                S_inv_sqrt->add_block(global_row, col_gid, block_data.data(), r_dim, c_dim, AssemblyMode::ADD, MatrixLayout::ColMajor);
                
                col_offset += c_dim;
                    
            }
        }
    }
    
    if (rank == 0 && verbose) std::cout << "isqrtm: Z0 Loop Finished. Assembling..." << std::endl;
    if (rank == 0) std::cout << "isqrtm: remote_blocks size: " << S_inv_sqrt->remote_blocks.size() << std::endl;

    S_inv_sqrt->assemble();
    
}
// Matrix Inverse Square Root using Newton-Schulz Iteration
// Computes S^{-1/2}
template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>* S_inv_sqrt, 
            double tol = 1e-7, double sptol = 1e-6, int maxit = 20, std::string init_method = "graph", double factor = 1.0,
            bool verbose = false) {
    
    DistGraph* graph = S_mat.graph;
    int rank = graph->rank;
    
    if (init_method == "graph") {
        isqrtm_init(S_mat, S_inv_sqrt, verbose);
    } else if (init_method == "simple") {
        S_inv_sqrt->shift(T(factor));   
    } else if (init_method == "taylor") {
        S_inv_sqrt->shift(T(1.5));
        S_inv_sqrt->axpy(T(0.5), S_mat);
    } else {
        throw std::runtime_error("Invalid init_method");
    }

    // ZSZ = (1-0.5D)(1+D)(1-0.5D)=(1+0.5D-0.5D^2)(1-0.5D)=1-0.75D^2+0.25D^3

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
    if (rank == 0) std::cout << "isqrtm: Initial norm=" << norm <<  std::endl;
    
    // Restore tmp2
    tmp2.shift(T(1.0));
    
    double norm_pre = 0.0;
    int count = 0;
    
    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        double density = S_inv_sqrt->get_block_density();
        if (rank == 0 && verbose) {
            std::cout << "Recursive Factorization of S: current error: maxabs norm - " << norm << " with density " << density << std::endl;
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
        S_inv_sqrt->axpby(T(0.5), S_inv_sqrt->transpose(), T(0.5));
        S_inv_sqrt->filter_blocks(sptol);

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
    
    double final_density = S_inv_sqrt->get_block_density();
    if (rank == 0 && verbose) {
        std::cout << "Recursive Factorization of S: Converge at maxabs norm - " << norm << " with density " << final_density << std::endl;
    }
}

} // namespace calc
} // namespace rsatb

#endif
