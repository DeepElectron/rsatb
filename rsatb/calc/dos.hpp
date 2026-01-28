#ifndef RSATB_CALC_DOS_HPP
#define RSATB_CALC_DOS_HPP

#include "cheby.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

using namespace vbcsr;

namespace rsatb {
namespace calc {

// Compute Time Correlation Function (for DOS)
// cdos(t) = <psi(0) | psi(t)>
// Propagates psi(t) = e^{-i H t} psi(0) step-by-step.
template <typename T, typename Kernel = DefaultKernel<T>>
void corr_dos(Chebyshev<T, Kernel>& cheby, 
              const DistVector<T>& psi_in, 
              int nt, 
              std::vector<std::complex<double>>& cdos, 
              const DistVector<T>* proj_mask = nullptr, 
              std::vector<std::complex<double>>* pcdos = nullptr,
              bool verbose = false) {
    
    DistGraph* graph = psi_in.graph;
    int rank = graph->rank;
    int n = graph->block_offsets[-1];
    double dt = M_PI;

    // 1. Generate Time Propagation Coefficients for dt
    std::vector<std::complex<double>> coeffs;
    // Precision 1e-14 as in old code
    timeprop_coeff(dt, 1e-14, true, coeffs);

    // 2. Prepare Vectors
    DistVector<T> psi_curr(graph);
    DistVector<T> psi_out(graph);
    DistVector<T> v_tmp(graph); // For projection

    psi_curr.copy_from(psi_in);
    
    // Resize output arrays
    // Old code stores [0..nt-1] for positive times?
    // Old code: cdos[nt] = 1 (t=0). cdos[nt+it+1] (t>0). cdos[nt-it-1] (t<0).
    // It seems it stores symmetric time window [-T, T].
    // Let's follow a simpler convention: [0, nt].
    // Or match old code if user expects it.
    // The user said "implement its function".
    // Old code signature: `PetscComplex *cdos`.
    // Here we use `std::vector`.
    // Let's store 0 to nt.
    cdos.resize(nt + 1);
    if (pcdos) pcdos->resize(nt + 1);

    // t=0
    std::complex<double> overlap;
    overlap = psi_curr.dot(psi_in);
    cdos[0] = overlap;
    
    if (pcdos && proj_mask) {
        // Projected overlap: <psi_curr | P | psi_in> = <psi_curr | mask * psi_in>
        // Assuming mask is real (0/1).
        // v_tmp = psi_in * mask
        v_tmp.copy_from(psi_in);
        v_tmp.pointwise_mult(*proj_mask);
        overlap = psi_curr.dot(v_tmp);
        (*pcdos)[0] = overlap * static_cast<double>(n);
    }

    // 3. Time Propagation Loop
    for (int it = 0; it < nt; ++it) {
        if (rank == 0 && verbose) {
            std::cout << "corr_dos calculation: " << it + 1 << " / " << nt << "\n";
        }

        // Propagate: psi_out = e^{-i H dt} psi_curr
        cheby.propagate(psi_curr, psi_out, coeffs, ChebyMode::NORMAL);
        
        // Correct phase for shift: e^{-i H dt} = e^{-i (H_scaled * half_width + center) dt}
        // propagate applies T_n(H_scaled).
        // timeprop_coeff generates coefficients for e^{-i H_scaled * (half_width * dt)}.
        // So we just need to multiply by phase e^{-i center dt}.
        // Wait, timeprop_coeff generates coeffs for e^{-i \tau x}. Here \tau = dt * half_width.
        // So propagate computes e^{-i H_scaled * half_width * dt}.
        // We need e^{-i H dt} = e^{-i (H_scaled * half_width + center) dt}
        // = e^{-i center dt} * e^{-i H_scaled * half_width * dt}
        
        // std::complex<double> phase = std::exp(std::complex<double>(0, -cheby.center * dt / cheby.half_width));
        // psi_out.scale(phase);

        // Compute Overlap <psi_out | psi_in> ?
        // Old code: VecDot(Psi_out, Psi_in, cdos + ...).
        // Note: VecDot(x, y) in PETSc is y^H x.
        // So <Psi_in | Psi_out>.
        // cdos(t) = <psi(0) | psi(t)>.
        // My DistVector::dot(other) computes this->dot(other) = this^H * other.
        // So psi_in.dot(psi_out) gives <psi_in | psi_out>.
        
        overlap = psi_in.dot(psi_out);
        cdos[it + 1] = overlap;

        if (pcdos && proj_mask) {
            v_tmp.copy_from(psi_in);
            v_tmp.pointwise_mult(*proj_mask);
            overlap = v_tmp.dot(psi_out); // <P psi_in | psi_out> = <psi_in | P psi_out>
            (*pcdos)[it + 1] = overlap * static_cast<double>(n);
        }

        // Update for next step
        psi_curr.copy_from(psi_out);
    }
}

// Multivector Version
template <typename T, typename Kernel = DefaultKernel<T>>
void corr_dos(Chebyshev<T, Kernel>& cheby, 
              const DistMultiVector<T>& psi_in, 
              int nt, 
              std::vector<std::complex<double>>& cdos, 
              const DistVector<T>* proj_mask = nullptr, 
              std::vector<std::complex<double>>* pcdos = nullptr,
              bool verbose = false) {
    
    DistGraph* graph = psi_in.graph;
    int rank = graph->rank;
    int num_vecs = psi_in.num_vectors;
    int n = graph->block_offsets[-1];
    double dt = M_PI;

    std::vector<std::complex<double>> coeffs;
    timeprop_coeff(dt, 1e-14, true, coeffs);

    DistMultiVector<T> psi_curr(graph, num_vecs);
    DistMultiVector<T> psi_out(graph, num_vecs);
    DistMultiVector<T> v_tmp(graph, num_vecs);

    psi_curr.copy_from(psi_in);
    
    cdos.resize(nt + 1);
    if (pcdos) pcdos->resize(nt + 1);

    // t=0
    std::vector<T> dots(num_vecs);
    psi_in.dot(psi_curr, dots);
    std::complex<double> sum_overlap = 0;
    for(auto d : dots) sum_overlap += d;
    cdos[0] = sum_overlap / static_cast<double>(num_vecs); // Average over vectors?
    // Old code: cdos[...] += temp; ... cdos[...] /= nrhs;
    // Yes, average.

    if (pcdos && proj_mask) {
        v_tmp.copy_from(psi_in);
        v_tmp.pointwise_mult(*proj_mask);
        
        v_tmp.dot(psi_curr, dots);
        sum_overlap = 0;
        for(auto d : dots) sum_overlap += d;
        (*pcdos)[0] = sum_overlap / static_cast<double>(num_vecs) * static_cast<double>(n);
    }

    for (int it = 0; it < nt; ++it) {
        if (rank == 0 && verbose) {
            std::cout << "corr_dos calculation: " << it + 1 << " / " << nt << "\n";
        }

        cheby.propagate(psi_curr, psi_out, coeffs, ChebyMode::NORMAL);
        
        // std::complex<double> phase = std::exp(std::complex<double>(0, -cheby.center * dt / cheby.half_width));
        // psi_out.scale(phase);

        psi_in.dot(psi_out, dots);
        sum_overlap = 0;
        for(auto d : dots) sum_overlap += d;
        cdos[it + 1] = sum_overlap / static_cast<double>(num_vecs);

        if (pcdos && proj_mask) {
            v_tmp.copy_from(psi_in);
            v_tmp.pointwise_mult(*proj_mask);
            
            v_tmp.dot(psi_out, dots);
            sum_overlap = 0;
            for(auto d : dots) sum_overlap += d;
            (*pcdos)[it + 1] = sum_overlap / static_cast<double>(num_vecs) * static_cast<double>(n);
        }

        psi_curr.copy_from(psi_out);
    }
}

} // namespace calc
} // namespace rsatb

#endif
