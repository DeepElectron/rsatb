#ifndef RSATB_CALC_SOLVER_HPP
#define RSATB_CALC_SOLVER_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_vector.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include "preconditioner.hpp"
#include <iostream>
#include <cmath>

namespace rsatb {
namespace calc {

using namespace vbcsr;

// Linear Solver class
// Currently implements Conjugate Gradient (CG) for SPD matrices
template <typename T, typename Kernel = DefaultKernel<T>>
class LinearSolver {
public:
    double tol;
    int max_iter;
    bool verbose;

    LinearSolver(double tolerance = 1e-8, int max_iterations = 1000, bool verbose_output = false)
        : tol(tolerance), max_iter(max_iterations), verbose(verbose_output) {}

    // Generic Solve for any MatrixType that supports mult_optimized and has graph member
    template <typename MatrixType>
    void solve(MatrixType& A, const DistVector<T>& b, DistVector<T>& x) {
        // CG Algorithm:
        // r0 = b - A*x0
        
        DistVector<T> r = b.duplicate(); // r = b
        DistVector<T> Ax = b.duplicate(); // Temp for A*x
        
        // Compute initial residual r = b - A*x
        A.mult_optimized(x, Ax);
        r.axpby(-1.0, Ax, 1.0); // r = b - Ax
        
        DistVector<T> p = r.duplicate();
        
        T r_norm_sq = r.dot(r); // (r, r)
        double initial_error = std::sqrt(std::abs(r_norm_sq));
        
        if (verbose && A.graph->rank == 0) {
            std::cout << "CG Start: Initial Residual = " << initial_error << std::endl;
        }
        
        if (initial_error < tol) return;
        
        DistVector<T> Ap = p.duplicate(); // Temp for A*p
        
        for (int k = 0; k < max_iter; ++k) {
            // Ap = A * p
            A.mult_optimized(p, Ap);
            
            // alpha = (r, r) / (p, Ap)
            T p_Ap = p.dot(Ap);
            
            if (std::abs(p_Ap) < 1e-16) {
                if (verbose && A.graph->rank == 0) std::cout << "CG Breakdown: p_Ap too small" << std::endl;
                break;
            }
            
            T alpha = r_norm_sq / p_Ap;
            
            // x = x + alpha * p
            x.axpy(alpha, p);
            
            // r = r - alpha * Ap
            r.axpy(-alpha, Ap);
            
            T r_new_norm_sq = r.dot(r);
            
            double error = std::sqrt(std::abs(r_new_norm_sq));
            if (verbose && A.graph->rank == 0 && (k % 10 == 0)) {
                std::cout << "CG Iter " << k << ": Residual = " << error << std::endl;
            }
            
            if (error < tol) {
                if (verbose && A.graph->rank == 0) std::cout << "CG Converged in " << k+1 << " iterations." << std::endl;
                break;
            }
            
            T beta = r_new_norm_sq / r_norm_sq;
            
            // p = r + beta * p
            p.axpby(1.0, r, beta);
            
            r_norm_sq = r_new_norm_sq;
        }
    }
    
    // Overload for Preconditioned Solve
    // P should approximate A^-1 for Left/Right, or A^-1/2 for Symmetric
    template <typename MatrixType>
    void solve(MatrixType& A, const DistVector<T>& b, DistVector<T>& x, Preconditioner<T, Kernel>& P, PreconditionerType type) {
        if (type == PreconditionerType::Symmetric) {
            // Explicit Symmetric Preconditioning: P * A * P * y = P * b, x = P * y
            // Requires P approx A^-1/2
            SymPrecondMat<MatrixType, Preconditioner<T, Kernel>, T> PHAP(A, P);
            DistVector<T> PHb = b.duplicate();
            P.apply_adjoint(b, PHb);
            DistVector<T> y = x.duplicate();
            y.set_constant(0.0); // Assume x0=0 or handle initial guess mapping, since PHAP approx Identity, y=PHb is the good initial guess maybe?
            solve(PHAP, PHb, y);
            P.apply(y, x);
        } else {
            // Standard Preconditioned CG (PCG)
            // Solves A x = b with preconditioner M^-1 = P
            // Requires P approx A^-1
            // This algorithm is mathematically equivalent to Left Preconditioning but preserves symmetry
            
            DistVector<T> r = b.duplicate();
            DistVector<T> Ax = b.duplicate();
            
            // r = b - A x
            A.mult_optimized(x, Ax);
            r.axpby(-1.0, Ax, 1.0);
            
            DistVector<T> z = r.duplicate();
            P.apply(r, z); // z = M^-1 r
            
            DistVector<T> p = z.duplicate(); // p = z
            
            T rho = r.dot(z); // rho = r . z
            
            double initial_error = std::sqrt(std::abs(r.dot(r)));
             if (verbose && A.graph->rank == 0) {
                std::cout << "PCG Start: Initial Residual = " << initial_error << std::endl;
            }
            
            if (initial_error < tol) return;
            
            DistVector<T> Ap = p.duplicate();
            
            for (int k = 0; k < max_iter; ++k) {
                // Ap = A * p
                A.mult_optimized(p, Ap);
                
                // alpha = rho / (p . Ap)
                T p_Ap = p.dot(Ap);
                
                if (std::abs(p_Ap) < 1e-16) {
                    if (verbose && A.graph->rank == 0) std::cout << "PCG Breakdown: p_Ap too small" << std::endl;
                    break;
                }
                
                T alpha = rho / p_Ap;
                
                // x = x + alpha * p
                x.axpy(alpha, p);
                
                // r = r - alpha * Ap
                r.axpy(-alpha, Ap);
                
                // Check convergence
                T r_norm_sq = r.dot(r);
                double error = std::sqrt(std::abs(r_norm_sq));
                
                if (verbose && A.graph->rank == 0 && (k % 10 == 0)) {
                    std::cout << "PCG Iter " << k << ": Residual = " << error << std::endl;
                }
                
                if (error < tol) {
                    if (verbose && A.graph->rank == 0) std::cout << "PCG Converged in " << k+1 << " iterations." << std::endl;
                    break;
                }
                
                // z = M^-1 r
                P.apply(r, z);
                
                T rho_new = r.dot(z);
                T beta = rho_new / rho;
                
                // p = z + beta * p
                p.axpby(1.0, z, beta);
                
                rho = rho_new;
            }
        }
    }

    // Solve AX = B for multivector (Batched CG)
    template <typename MatrixType>
    void solve(MatrixType& A, const DistMultiVector<T>& B, DistMultiVector<T>& X) {
        // We can treat each column independently, but process them together in MatMults.
        
        int num_vecs = B.num_vectors;
        
        DistMultiVector<T> R = B; // Copy B
        DistMultiVector<T> AX = B; // Temp
        
        // R = B - A*X
        A.mult_dense(X, AX);
        R.axpby(-1.0, AX, 1.0); // R = B - AX
        
        DistMultiVector<T> P = R; // P = R
        
        std::vector<T> R_norm_sq;
        R.dot(R, R_norm_sq); // Column-wise dot
        
        DistMultiVector<T> AP = P; // Temp
        
        // Check convergence per vector? Or max residual?
        // Let's track max residual.
        
        for (int k = 0; k < max_iter; ++k) {
            // AP = A * P
            A.mult_dense(P, AP);
            
            // alpha = (R, R) / (P, AP)
            std::vector<T> P_AP;
            P.dot(AP, P_AP);
            
            // Update each vector
            // We need to apply different alpha for each column.
            // DistMultiVector doesn't support column-specific scaling easily in one call unless we add it.
            // But we can do it manually on local data.
            
            #pragma omp parallel for
            for (int c = 0; c < num_vecs; ++c) {
                T alpha = R_norm_sq[c] / P_AP[c];
                
                // X_c = X_c + alpha * P_c
                // R_c = R_c - alpha * AP_c
                
                T* X_ptr = X.col_data(c);
                const T* P_ptr = P.col_data(c);
                T* R_ptr = R.col_data(c);
                const T* AP_ptr = AP.col_data(c);
                
                for (int i = 0; i < X.local_rows; ++i) {
                    X_ptr[i] += alpha * P_ptr[i];
                    R_ptr[i] -= alpha * AP_ptr[i];
                }
            }
            
            std::vector<T> R_new_norm_sq;
            R.dot(R, R_new_norm_sq);
            
            // Check convergence
            double max_err = 0.0;
            for(auto v : R_new_norm_sq) max_err = std::max(max_err, std::sqrt(std::abs(v)));
            
            if (verbose && A.graph->rank == 0 && (k % 10 == 0)) {
                std::cout << "Batched CG Iter " << k << ": Max Residual = " << max_err << std::endl;
            }
            
            if (max_err < tol) {
                if (verbose && A.graph->rank == 0) std::cout << "Batched CG Converged in " << k+1 << " iterations." << std::endl;
                break;
            }
            
            // Update P
            #pragma omp parallel for
            for (int c = 0; c < num_vecs; ++c) {
                T beta = R_new_norm_sq[c] / R_norm_sq[c];
                
                // P_c = R_c + beta * P_c
                T* P_ptr = P.col_data(c);
                const T* R_ptr = R.col_data(c);
                
                for (int i = 0; i < P.local_rows; ++i) {
                    P_ptr[i] = R_ptr[i] + beta * P_ptr[i];
                }
            }
            
            R_norm_sq = R_new_norm_sq;
        }
    }
    
    // Overload for Preconditioned Batched Solve
    // Overload for Preconditioned Batched Solve
    template <typename MatrixType>
    void solve(MatrixType& A, const DistMultiVector<T>& B, DistMultiVector<T>& X, Preconditioner<T, Kernel>& P, PreconditionerType type) {
        if (type == PreconditionerType::Symmetric) {
            // Explicit Symmetric Preconditioning: PH * A * P * Y = PH * B, X = P * Y
            SymPrecondMat<MatrixType, Preconditioner<T, Kernel>, T> PHAP(A, P);
            DistMultiVector<T> PHb = B;
            P.apply_adjoint(B, PHb);
            DistMultiVector<T> Y = X;
            std::fill(Y.data.begin(), Y.data.end(), 0.0);
            solve(PHAP, PHb, Y);
            P.apply(Y, X);
        } else {
            // Batched PCG
            // Solves A X = B with preconditioner M^-1 = P
            
            int num_vecs = B.num_vectors;
            
            DistMultiVector<T> R = B; // Copy B
            DistMultiVector<T> AX = B; // Temp
            
            // R = B - A*X
            A.mult_dense(X, AX);
            R.axpby(-1.0, AX, 1.0); // R = B - AX
            
            DistMultiVector<T> Z = R; // Z = M^-1 R
            P.apply(R, Z);
            
            DistMultiVector<T> P_vec = Z; // P = Z
            
            std::vector<T> Rho;
            R.dot(Z, Rho); // Rho = R . Z (column-wise)
            
            std::vector<T> R_norm_sq;
            R.dot(R, R_norm_sq); // Track residual norm for convergence check
            
            double initial_max_err = 0.0;
            for(auto v : R_norm_sq) initial_max_err = std::max(initial_max_err, std::sqrt(std::abs(v)));
            
            if (verbose && A.graph->rank == 0) {
                std::cout << "Batched PCG Start: Max Residual = " << initial_max_err << std::endl;
            }
            
            if (initial_max_err < tol) return;
            
            DistMultiVector<T> AP = P_vec; // Temp
            
            for (int k = 0; k < max_iter; ++k) {
                // AP = A * P
                A.mult_dense(P_vec, AP);
                
                // alpha = Rho / (P . AP)
                std::vector<T> P_AP;
                P_vec.dot(AP, P_AP);
                
                // Update X, R
                #pragma omp parallel for
                for (int c = 0; c < num_vecs; ++c) {
                    T alpha = Rho[c] / P_AP[c];
                    
                    T* X_ptr = X.col_data(c);
                    const T* P_ptr = P_vec.col_data(c);
                    T* R_ptr = R.col_data(c);
                    const T* AP_ptr = AP.col_data(c);
                    
                    for (int i = 0; i < X.local_rows; ++i) {
                        X_ptr[i] += alpha * P_ptr[i];
                        R_ptr[i] -= alpha * AP_ptr[i];
                    }
                }
                
                // Check convergence
                R.dot(R, R_norm_sq);
                double max_err = 0.0;
                for(auto v : R_norm_sq) max_err = std::max(max_err, std::sqrt(std::abs(v)));
                
                if (verbose && A.graph->rank == 0 && (k % 10 == 0)) {
                    std::cout << "Batched PCG Iter " << k << ": Max Residual = " << max_err << std::endl;
                }
                
                if (max_err < tol) {
                    if (verbose && A.graph->rank == 0) std::cout << "Batched PCG Converged in " << k+1 << " iterations." << std::endl;
                    break;
                }
                
                // Z = M^-1 R
                P.apply(R, Z);
                
                std::vector<T> Rho_new;
                R.dot(Z, Rho_new);
                
                // Update P
                #pragma omp parallel for
                for (int c = 0; c < num_vecs; ++c) {
                    T beta = Rho_new[c] / Rho[c];
                    
                    T* P_ptr = P_vec.col_data(c);
                    const T* Z_ptr = Z.col_data(c);
                    
                    for (int i = 0; i < P_vec.local_rows; ++i) {
                        P_ptr[i] = Z_ptr[i] + beta * P_ptr[i];
                    }
                }
                
                Rho = Rho_new;
            }
        }
    }
};

} // namespace calc
} // namespace rsatb

#endif
