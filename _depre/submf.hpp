#ifndef RSATB_CALC_LANMF_HPP
#define RSATB_CALC_LANMF_HPP

#include "lapack_types.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <set>
#include <functional>

using namespace vbcsr;

namespace rsatb {
namespace calc {
namespace detail {

    // C = alpha * A * B + beta * C
    template <typename T>
    void dense_gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc, bool transA = false, bool transB = false) {
        const char* ta = transA ? (std::is_same<T, std::complex<double>>::value ? "C" : "T") : "N";
        const char* tb = transB ? (std::is_same<T, std::complex<double>>::value ? "C" : "T") : "N";
        
        rsatb_lapack_int m_ = m;
        rsatb_lapack_int n_ = n;
        rsatb_lapack_int k_ = k;
        rsatb_lapack_int lda_ = lda;
        rsatb_lapack_int ldb_ = ldb;
        rsatb_lapack_int ldc_ = ldc;

        if constexpr (std::is_same<T, double>::value) {
            dgemm_(ta, tb, &m_, &n_, &k_, &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_);
        } else if constexpr (std::is_same<T, std::complex<double>>::value) {
             zgemm_(ta, tb, &m_, &n_, &k_, 
                    reinterpret_cast<const rsatb_complex_double*>(&alpha), 
                    reinterpret_cast<const rsatb_complex_double*>(A), &lda_, 
                    reinterpret_cast<const rsatb_complex_double*>(B), &ldb_, 
                    reinterpret_cast<const rsatb_complex_double*>(&beta), 
                    reinterpret_cast<rsatb_complex_double*>(C), &ldc_);
        }
    }

} // namespace detail


// Helper to compute f(M) for a dense matrix M
template <typename T>
void dense_matrix_function(int n_in, std::vector<T>& M, std::function<T(double)> func, int k_cols = -1, int col_start_idx = 0) {
    rsatb_lapack_int n = n_in;
    if (n == 0) return;
    
    int k = (k_cols <= 0) ? n : k_cols;
    
    std::vector<double> w(n); // Eigenvalues
    rsatb_lapack_int info;
    rsatb_lapack_int lwork = -1;
    
    if constexpr (std::is_same<T, double>::value) {
        double wkopt;
        rsatb_lapack_int iwkopt;
        rsatb_lapack_int lwork_query = -1;
        rsatb_lapack_int liwork_query = -1;
        dsyevd_("V", "U", &n, (double*)M.data(), &n, w.data(), &wkopt, &lwork_query, &iwkopt, &liwork_query, &info);
        lwork = (rsatb_lapack_int)wkopt;
        rsatb_lapack_int liwork = iwkopt;
        std::vector<double> work(lwork);
        std::vector<rsatb_lapack_int> iwork(liwork);
        dsyevd_("V", "U", &n, (double*)M.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else {
        rsatb_complex_double wkopt;
        double rwkopt;
        rsatb_lapack_int iwkopt;
        rsatb_lapack_int lwork_query = -1;
        rsatb_lapack_int lrwork_query = -1;
        rsatb_lapack_int liwork_query = -1;
        rsatb_complex_double* a_ptr = reinterpret_cast<rsatb_complex_double*>(M.data());
        zheevd_("V", "U", &n, a_ptr, &n, w.data(), &wkopt, &lwork_query, &rwkopt, &lrwork_query, &iwkopt, &liwork_query, &info);
        lwork = (rsatb_lapack_int)wkopt.real();
        rsatb_lapack_int lrwork = (rsatb_lapack_int)rwkopt;
        rsatb_lapack_int liwork = iwkopt;
        std::vector<rsatb_complex_double> work(lwork);
        std::vector<double> rwork(lrwork);
        std::vector<rsatb_lapack_int> iwork(liwork);
        zheevd_("V", "U", &n, a_ptr, &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }
    
    if (info != 0) {
        std::cerr << "Error in LAPACK eigendecomposition (info=" << info << ")" << std::endl;
        return;
    }
    
    // M now contains eigenvectors V. w contains eigenvalues.
    // Result_subset = (V * diag(f(w))) * (V^H)[:, col_start_idx : col_start_idx + k]
    std::vector<T> V_scaled = M;
    for (int j = 0; j < n; ++j) {
        T val = func(w[j]);
        for (int i = 0; i < n; ++i) {
            V_scaled[i + j * n] *= val;
        }
    }
    
    std::vector<T> Res(n * k, T(0));
    // C(n x k) = V_scaled(n x n) * M(k x n)^H
    // Select rows [col_start_idx, col_start_idx + k) of V (M).
    // Pointer arithmetic `M.data() + col_start_idx` with stride `n` correctly slices the rows.
    detail::dense_gemm(n, k, n, T(1.0), V_scaled.data(), n, M.data() + col_start_idx, n, T(0.0), Res.data(), n, false, true);
    M = Res;
}

// Block Lanczos Method for Matrix Function (First k columns) - Dense MVP Version
template <typename T, typename Kernel = DefaultKernel<T>>
void lanczos_matrix_function_dense(const std::vector<T>& dense_S, DistMultiVector<T>& X, std::function<T(double)> func, int n_cols, DistGraph* graph, double tol = 1e-7, int max_iter = 50, bool verbose = false, bool reorth = true, int global_col_start = 0) {
    int rank = graph->rank;
    
    int m_owned = X.local_rows;
    int m_local = X.local_rows + X.ghost_rows;
    
    int n_global = 0;
    for (int sz : graph->block_sizes) n_global += sz;

    // Initial Block Size
    int b_curr = n_cols;
    
    // Contiguous Basis Storage
    int max_basis_cols = (max_iter + 1) * n_cols;
    std::vector<T> V_basis(m_local * max_basis_cols, T(0));
    std::vector<int> V_offsets; 
    std::vector<int> V_sizes;   
    
    std::vector<std::vector<T>> A_blocks; 
    std::vector<std::vector<T>> B_blocks; 
    
    // Workspace Pooling
    std::vector<T> W_data(m_local * n_cols);
    std::vector<T> overlaps(max_basis_cols * n_cols);
    
    V_offsets.push_back(0);
    V_sizes.push_back(b_curr);
    
    std::vector<int> global_block_offsets(graph->block_sizes.size() + 1, 0);
    for(size_t i=0; i<graph->block_sizes.size(); ++i) {
        global_block_offsets[i+1] = global_block_offsets[i] + graph->block_sizes[i];
    }

    int n_owned_blocks = graph->owned_global_indices.size();
    int local_row_offset = 0;
    for(int i=0; i<n_owned_blocks; ++i) {
        int blk_gid = graph->owned_global_indices[i];
        int blk_size = graph->block_sizes[blk_gid];
        int global_row_start = global_block_offsets[blk_gid];
        for(int r=0; r<blk_size; ++r) {
            int global_r = global_row_start + r;
            // Check if global_r is in [global_col_start, global_col_start + n_cols)
            if (global_r >= global_col_start && global_r < global_col_start + n_cols) {
                int c = global_r - global_col_start;
                V_basis[(local_row_offset + r) + c * m_local] = T(1.0);
            }
        }
        local_row_offset += blk_size;
    }
    
    double t_spmm = 0, t_reorth = 0, t_norm = 0, t_other = 0;

    for (int j = 0; j < max_iter; ++j) {
        if (rank==0 && verbose) std::cout << "Lanczos Iter " << j << " Block Size: " << b_curr << std::endl;
        if (b_curr == 0) break;

        // W = S * V_j (Dense GEMM)
        double t0 = MPI_Wtime();
        // dense_S is Row-Major (m_owned x n_global). 
        // In dense_gemm, we pass transA=true to treat it as Col-Major (n_global x m_owned) transposed.
        // lda for dense_S is n_global.
        detail::dense_gemm(m_owned, b_curr, n_global, T(1.0), dense_S.data(), n_global, 
                           &V_basis[V_offsets[j] * m_local], m_local, T(0.0), W_data.data(), m_local, true, false);
        t_spmm += MPI_Wtime() - t0;
        
        double t1 = MPI_Wtime();
        if (j > 0) {
            int b_prev = V_sizes[j-1];
            detail::dense_gemm(m_local, b_curr, b_prev, T(-1.0), 
                               &V_basis[V_offsets[j-1] * m_local], m_local, 
                               B_blocks.back().data(), b_curr, T(1.0), W_data.data(), m_local, false, true);
        }
        
        std::vector<T> Aj(b_curr * b_curr);
        detail::dense_gemm(b_curr, b_curr, m_owned, T(1.0), 
                           &V_basis[V_offsets[j] * m_local], m_local, 
                           W_data.data(), m_local, T(0.0), Aj.data(), b_curr, true, false);
        A_blocks.push_back(Aj);
        
        detail::dense_gemm(m_local, b_curr, b_curr, T(-1.0), 
                           &V_basis[V_offsets[j] * m_local], m_local, 
                           Aj.data(), b_curr, T(1.0), W_data.data(), m_local, false, false);
        t_other += MPI_Wtime() - t1;
        
        if (reorth) {
            double t2 = MPI_Wtime();
            int total_cols_so_far = V_offsets[j] + V_sizes[j];
            auto cgs_step = [&]() {
                detail::dense_gemm(total_cols_so_far, b_curr, m_owned, T(1.0), V_basis.data(), m_local, W_data.data(), m_local, T(0.0), overlaps.data(), total_cols_so_far, true, false);
                detail::dense_gemm(m_local, b_curr, total_cols_so_far, T(-1.0), V_basis.data(), m_local, overlaps.data(), total_cols_so_far, T(1.0), W_data.data(), m_local, false, false);
            };
            cgs_step(); cgs_step();
            t_reorth += MPI_Wtime() - t2;
        }
        
        double t3 = MPI_Wtime();
        std::vector<T> G(b_curr * b_curr);
        detail::dense_gemm(b_curr, b_curr, m_owned, T(1.0), W_data.data(), m_local, W_data.data(), m_local, T(0.0), G.data(), b_curr, true, false);
        
        int info = 0;
        if constexpr (std::is_same<T, double>::value) {
            rsatb_lapack_int n_int = b_curr;
            dpotrf_("U", &n_int, G.data(), &n_int, &info);
        } else {
            rsatb_lapack_int n_int = b_curr;
            zpotrf_("U", &n_int, reinterpret_cast<rsatb_complex_double*>(G.data()), &n_int, &info);
        }
        
        int next_rank;
        std::vector<T> Bj;
        if (info == 0) {
            next_rank = b_curr;
            std::vector<T> R(b_curr * b_curr);
            for(int j_idx=0; j_idx<b_curr; ++j_idx) {
                for(int i=0; i<=j_idx; ++i) R[i + j_idx*b_curr] = G[i + j_idx*b_curr];
                for(int i=j_idx+1; i<b_curr; ++i) R[i + j_idx*b_curr] = T(0.0);
            }
            rsatb_lapack_int m_int = m_local;
            rsatb_lapack_int n_int = b_curr;
            if constexpr (std::is_same<T, double>::value) {
                double alpha = 1.0;
                dtrsm_("R", "U", "N", "N", &m_int, &n_int, &alpha, R.data(), &n_int, W_data.data(), &m_int);
            } else {
                rsatb_complex_double alpha = 1.0;
                ztrsm_("R", "U", "N", "N", &m_int, &n_int, &alpha, reinterpret_cast<const rsatb_complex_double*>(R.data()), &n_int, reinterpret_cast<rsatb_complex_double*>(W_data.data()), &m_int);
            }
            Bj.resize(b_curr * b_curr);
            for(int r=0; r<b_curr; ++r) {
                for(int c=0; c<b_curr; ++c) Bj[r + c*b_curr] = R[r + c*b_curr];
            }
        } else {
            // SVD Fallback
            std::vector<T> U(m_local * b_curr);
            std::vector<T> VT(b_curr * b_curr);
            std::vector<double> s(b_curr);
            rsatb_lapack_int m_ = m_local, n_ = b_curr, info_ = 0, lwork = -1;
            std::vector<T> W_copy = W_data;
            if constexpr (std::is_same<T, double>::value) {
                double wkopt;
                dgesvd_("S", "S", &m_, &n_, (double*)W_copy.data(), &m_, s.data(), (double*)U.data(), &m_, (double*)VT.data(), &n_, &wkopt, &lwork, &info_);
                lwork = (rsatb_lapack_int)wkopt; std::vector<double> work(lwork);
                dgesvd_("S", "S", &m_, &n_, (double*)W_copy.data(), &m_, s.data(), (double*)U.data(), &m_, (double*)VT.data(), &n_, work.data(), &lwork, &info_);
            } else {
                rsatb_complex_double wkopt; std::vector<double> rwork(5 * b_curr);
                zgesvd_("S", "S", &m_, &n_, reinterpret_cast<rsatb_complex_double*>(W_copy.data()), &m_, s.data(), reinterpret_cast<rsatb_complex_double*>(U.data()), &m_, reinterpret_cast<rsatb_complex_double*>(VT.data()), &n_, &wkopt, &lwork, rwork.data(), &info_);
                lwork = (rsatb_lapack_int)wkopt.real(); std::vector<rsatb_complex_double> work(lwork);
                zgesvd_("S", "S", &m_, &n_, reinterpret_cast<rsatb_complex_double*>(W_copy.data()), &m_, s.data(), reinterpret_cast<rsatb_complex_double*>(U.data()), &m_, reinterpret_cast<rsatb_complex_double*>(VT.data()), &n_, work.data(), &lwork, rwork.data(), &info_);
            }
            double max_sv = 0; for(double val : s) max_sv = std::max(max_sv, val);
            std::vector<int> keep_indices; for(int i=0; i<b_curr; ++i) if (s[i] > max_sv * 1e-10) keep_indices.push_back(i);
            next_rank = keep_indices.size(); if (next_rank == 0) break;
            Bj.resize(next_rank * b_curr);
            for(int r=0; r<next_rank; ++r) {
                int k = keep_indices[r];
                for(int c=0; c<b_curr; ++c) Bj[r + c*next_rank] = s[k] * VT[k + c*b_curr];
            }
            W_data.resize(m_local * next_rank);
            for(int c=0; c<next_rank; ++c) {
                int k = keep_indices[c];
                std::copy(U.begin() + k * m_local, U.begin() + (k + 1) * m_local, W_data.begin() + c * m_local);
            }
        }
        B_blocks.push_back(Bj);
        int next_offset = V_offsets.back() + V_sizes.back();
        V_offsets.push_back(next_offset); V_sizes.push_back(next_rank);
        std::copy(W_data.begin(), W_data.begin() + next_rank * m_local, V_basis.begin() + next_offset * m_local);
        b_curr = next_rank; t_norm += MPI_Wtime() - t3;
        
        // Convergence Check
        if ((j + 1) % 5 == 0) {
            // Construct current Tm
            int m_iters_curr = A_blocks.size();
            int total_dim_curr = 0;
            for(int k=0; k<m_iters_curr; ++k) total_dim_curr += V_sizes[k];
            
            std::vector<T> Tm_curr(total_dim_curr * total_dim_curr, T(0));
            int row_off_c = 0, col_off_c = 0;
            for(int k=0; k<m_iters_curr; ++k) {
                int b_k = V_sizes[k];
                for(int c=0; c<b_k; ++c) for(int r=0; r<b_k; ++r) Tm_curr[(row_off_c + r) + (col_off_c + c)*total_dim_curr] = A_blocks[k][r + c*b_k];
                if (k < m_iters_curr - 1) {
                    int b_next = V_sizes[k+1];
                    for(int c=0; c<b_k; ++c) for(int r=0; r<b_next; ++r) {
                        Tm_curr[(row_off_c + b_k + r) + (col_off_c + c)*total_dim_curr] = B_blocks[k][r + c*b_next];
                        T val = B_blocks[k][r + c*b_next];
                        if constexpr (std::is_same<T, std::complex<double>>::value) val = std::conj(val);
                        Tm_curr[(row_off_c + c) + (col_off_c + b_k + r)*total_dim_curr] = val;
                    }
                }
                row_off_c += b_k; col_off_c += b_k;
            }
            
            // Compute f(Tm)
            // Use n_cols for optimized calculation if available
            dense_matrix_function(total_dim_curr, Tm_curr, func, n_cols);
            
            // Extract Y_curr = f(Tm)[:, 0:n_cols]
            // Tm_curr is ColMajor (or we treat it as such for BLAS)
            // Actually dense_matrix_function modifies Tm_curr in place to be f(Tm).
            // We want the first n_cols columns.
            
            static std::vector<T> Y_prev;
            static int dim_prev = 0;
            
            if (j + 1 == 5) {
                // First check, just store
                dim_prev = total_dim_curr;
                Y_prev.resize(dim_prev * n_cols);
                for(int c=0; c<n_cols; ++c) {
                    for(int r=0; r<dim_prev; ++r) {
                        Y_prev[r + c*dim_prev] = Tm_curr[r + c*total_dim_curr];
                    }
                }
            } else {
                // Compare with Y_prev
                double diff_sq = 0.0;
                double norm_sq = 0.0;
                
                for(int c=0; c<n_cols; ++c) {
                    for(int r=0; r<total_dim_curr; ++r) {
                        T val_curr = Tm_curr[r + c*total_dim_curr];
                        T val_prev = (r < dim_prev) ? Y_prev[r + c*dim_prev] : T(0.0);
                        
                        double abs_diff = std::abs(val_curr - val_prev);
                        double abs_val = std::abs(val_curr);
                        diff_sq += abs_diff * abs_diff;
                        norm_sq += abs_val * abs_val;
                    }
                }
                
                double rel_err = std::sqrt(diff_sq) / std::sqrt(norm_sq);
                if (rank == 0 && verbose) std::cout << "  Iter " << j << " Rel Err: " << rel_err << std::endl;
                
                if (rel_err < tol) {
                    if (rank == 0 && verbose) std::cout << "  Converged at iter " << j << std::endl;
                    break;
                }
                
                // Update Y_prev
                dim_prev = total_dim_curr;
                Y_prev.resize(dim_prev * n_cols);
                for(int c=0; c<n_cols; ++c) {
                    for(int r=0; r<dim_prev; ++r) {
                        Y_prev[r + c*dim_prev] = Tm_curr[r + c*total_dim_curr];
                    }
                }
            }
        }
    }
    
    double t_tm_isqrt = 0, t_final_gemm = 0;
    
    // Construct T_m
    int m_iters = A_blocks.size();
    int total_dim = 0;
    for(int j=0; j<m_iters; ++j) total_dim += V_sizes[j];
    std::vector<T> Tm(total_dim * total_dim, T(0));
    int row_off = 0, col_off = 0;
    for(int j=0; j<m_iters; ++j) {
        int b_j = V_sizes[j];
        for(int c=0; c<b_j; ++c) for(int r=0; r<b_j; ++r) Tm[(row_off + r) + (col_off + c)*total_dim] = A_blocks[j][r + c*b_j];
        if (j < m_iters - 1) {
            int b_next = V_sizes[j+1];
            for(int c=0; c<b_j; ++c) for(int r=0; r<b_next; ++r) {
                Tm[(row_off + b_j + r) + (col_off + c)*total_dim] = B_blocks[j][r + c*b_next];
                T val = B_blocks[j][r + c*b_next];
                if constexpr (std::is_same<T, std::complex<double>>::value) val = std::conj(val);
                Tm[(row_off + c) + (col_off + b_j + r)*total_dim] = val;
            }
        }
        row_off += b_j; col_off += b_j;
    }
    
    double t4 = MPI_Wtime();
    dense_matrix_function(total_dim, Tm, func, n_cols);
    t_tm_isqrt = MPI_Wtime() - t4;
    
    X.bind_to_graph(graph); X.set_constant(T(0));
    
    double t5 = MPI_Wtime();
    detail::dense_gemm(m_local, n_cols, total_dim, T(1.0), V_basis.data(), m_local, Tm.data(), total_dim, T(0.0), X.data.data(), m_local, false, false);
    t_final_gemm = MPI_Wtime() - t5;

    if (rank == 0 && verbose) {
        std::cout << "  Timing Breakdown (Dense MVP):" << std::endl;
        std::cout << "    GEMM (MVP):  " << t_spmm << " s" << std::endl;
        std::cout << "    Reorth:      " << t_reorth << " s" << std::endl;
        std::cout << "    Norm:        " << t_norm << " s" << std::endl;
        std::cout << "    Tm ISQRT:    " << t_tm_isqrt << " s" << std::endl;
        std::cout << "    Final GEMM:  " << t_final_gemm << " s" << std::endl;
        std::cout << "    Other:       " << t_other << " s" << std::endl;
    }
}

// Block Lanczos Method for Matrix Function (First k columns)
template <typename T, typename Kernel = DefaultKernel<T>>
void lanczos_matrix_function(BlockSpMat<T, Kernel>& S, DistMultiVector<T>& X, std::function<T(double)> func, int n_cols, double tol = 1e-7, int max_iter = 50, bool verbose = false, bool reorth = true, int global_col_start = 0) {
    double t_start = MPI_Wtime();
    std::vector<T> dense_S = S.to_dense();
    double t_to_dense = MPI_Wtime() - t_start;
    
    // if (S.graph->rank == 0 && verbose) {
    //     std::cout << "  to_dense time: " << t_to_dense << " s" << std::endl;
    // }
    
    lanczos_matrix_function_dense(dense_S, X, func, n_cols, S.graph, tol, max_iter, verbose, reorth, global_col_start);
}
} // namespace calc
} // namespace rsatb

#endif // RSATB_CALC_LANMF_HPP