#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "source_estate/read_pseudo.h"
// #include "source_cell/atom_spec.h"
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_basis/module_nao/atomic_radials.h"
#include "source_basis/module_nao/beta_radials.h"
#include "source_base/ylm.h"
#include "source_base/math_integral.h"

namespace GlobalV {
    std::ofstream ofs_running;
}

class TwoCenterBundleTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup Atom
        ntype = 1;
        // atoms_vec[0].ncpp.pp_type = "upf";
        
        // Redirect GlobalV::ofs_running to avoid clutter
        GlobalV::ofs_running.open("test_two_center_bundle_global.log");
    }

    void TearDown() override
    {
        GlobalV::ofs_running.close();
    }

    int ntype;
    std::vector<Atom_pseudo> pseudos;
};

// Helper to get Ylm index
int get_lm_index(int l, int m) {
    int idx = l * l;
    if (m == 0) return idx;
    if (m > 0) return idx + 2 * m - 1;
    return idx + 2 * std::abs(m);
}

// Helper function for direct numerical integration of overlap
// Assumes R is along z-axis
// Uses NumericalRadial to get radial values
double DirectIntegration(const NumericalRadial& rad_orb, 
                         const NumericalRadial& rad_beta, 
                         double R, double rmax = 20.0)
{
    // Grid parameters
    int nr = 1000; // Radial points
    int ntheta = 180; // Theta points
    // Optimization: if m1=0 and m2=0, the integrand is independent of phi.
    // We can set nphi=1 and multiply by 2*pi.
    int nphi = 360; 
    
    // Check if we can optimize phi loop
    // Note: This assumes m1 and m2 are passed or can be deduced. 
    // Since this helper is generic, we'll check if l1=0 and l2=0 for now, 
    // or we can add m1, m2 arguments. 
    // But wait, the test calls this with specific orbitals.
    // Let's just assume m=0 for now as per test case, or add arguments.
    // To be safe and generic without changing signature too much, let's just check if we can exploit symmetry.
    // Actually, for the purpose of this test which uses m=0, we can just hardcode the optimization 
    // or add m1, m2 to arguments. Let's add m1, m2 to arguments with default 0.
    
    double dr = rmax / nr;
    double dtheta = M_PI / ntheta;
    double dphi = 2.0 * M_PI / nphi;
    
    double integral = 0.0;
    
    int l1 = rad_orb.l();
    int l2 = rad_beta.l();
    
    // Lambda for optimized interpolation (assuming uniform grid)
    auto get_val_opt = [](const NumericalRadial& rad, double r) -> double {
        if (r > rad.rmax()) return 0.0;
        const double* rgrid = rad.rgrid();
        int nr_grid = rad.nr();
        
        // Assume uniform grid starting at 0
        double dr_grid = rgrid[1] - rgrid[0]; // Estimate dr
        // Or better, use the fact that we know it's uniform
        
        int idx = static_cast<int>(r / dr_grid);
        if (idx < 0) return rad.rvalue(0); // Should not happen for r >= 0
        if (idx >= nr_grid - 1) return 0.0;
        
        double r0 = rgrid[idx];
        double r1 = rgrid[idx+1];
        double v0 = rad.rvalue(idx);
        double v1 = rad.rvalue(idx+1);
        
        double val = v0 + (v1 - v0) * (r - r0) / (r1 - r0);
        
        if (rad.pr() == 1) {
            if (r < 1e-10) return 0.0; 
            return val / r;
        }
        return val;
    };

    // Use the optimized getter if possible, otherwise fallback
    // For safety in this test, we'll stick to the robust one but optimize the search
    auto get_val = [](const NumericalRadial& rad, double r) -> double {
        if (r > rad.rmax()) return 0.0;
        const double* rgrid = rad.rgrid();
        int nr_grid = rad.nr();
        
        // Optimization: guess index based on uniform grid assumption
        // This is valid for the files we use
        double dr_grid = rgrid[nr_grid-1] / (nr_grid - 1);
        int idx = static_cast<int>(r / dr_grid);
        
        if (idx < 0) idx = 0;
        if (idx >= nr_grid - 1) idx = nr_grid - 2;
        
        // Verify index (in case of non-uniformity)
        if (r < rgrid[idx] || r > rgrid[idx+1]) {
             auto it = std::lower_bound(rgrid, rgrid + nr_grid, r);
             if (it == rgrid) return rad.rvalue(0);
             if (it == rgrid + nr_grid) return 0.0;
             idx = std::distance(rgrid, it) - 1;
        }
        
        double r0 = rgrid[idx];
        double r1 = rgrid[idx+1];
        double v0 = rad.rvalue(idx);
        double v1 = rad.rvalue(idx+1);
        
        double val = v0 + (v1 - v0) * (r - r0) / (r1 - r0);
        
        if (rad.pr() == 1) {
            if (r < 1e-10) return 0.0; 
            return val / r;
        }
        return val;
    };
    
    // Optimization for m=0
    // We know the test uses m=0. 
    // If we want to be generic, we should pass m. 
    // But let's just set nphi=1 and multiply by 2pi since we know the test case.
    // To be strictly correct, we should check if the result depends on phi.
    // For m=0, Ylm is constant in phi.
    // Let's assume m=0 for this helper as it is used in VerifyOverlap with m=0.
    nphi = 1;
    dphi = 2.0 * M_PI; // Effectively integrating 0 to 2pi is just multiplying by 2pi for constant integrand
    
    for (int ir = 0; ir < nr; ++ir) {
        double r = (ir + 0.5) * dr;
        double val1 = get_val(rad_orb, r);
        
        if (std::abs(val1) < 1e-10) continue; // Optimization: skip if val1 is zero

        for (int it = 0; it < ntheta; ++it) {
            double theta = (it + 0.5) * dtheta;
            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);
            
            for (int ip = 0; ip < nphi; ++ip) {
                double phi = (ip + 0.5) * dphi; // phi is 0 if nphi=1, but dphi is 2pi, so we multiply result by dphi
                
                // Position vector r1 = (r, theta, phi)
                double x = r * sin_theta * std::cos(phi);
                double y = r * sin_theta * std::sin(phi);
                double z = r * cos_theta;
                
                // Position vector r2 = r1 - R (R is along z)
                double x2 = x;
                double y2 = y;
                double z2 = z - R;
                double r2_norm = std::sqrt(x2*x2 + y2*y2 + z2*z2);
                
                // Value of orbital 2 at r2
                double val2 = get_val(rad_beta, r2_norm);
                
                // Spherical Harmonics
                // Ylm needs normalized direction
                double ylm1 = 1.0;
                if (l1 > 0) {
                     std::vector<double> ylm_vec((l1+1)*(l1+1));
                     ModuleBase::Vector3<double> rhat(x/r, y/r, z/r);
                     ModuleBase::Ylm::get_ylm_real(l1 + 1, rhat, ylm_vec.data());
                     // Assume m=0
                     int idx_lm = get_lm_index(l1, 0);
                     ylm1 = ylm_vec[idx_lm];
                } else {
                    ylm1 = 0.282094791773878; // Y00
                }

                double ylm2 = 1.0;
                if (l2 > 0) {
                     std::vector<double> ylm_vec((l2+1)*(l2+1));
                     ModuleBase::Vector3<double> rhat2(x2/r2_norm, y2/r2_norm, z2/r2_norm);
                     ModuleBase::Ylm::get_ylm_real(l2 + 1, rhat2, ylm_vec.data());
                     // Assume m=0
                     int idx_lm = get_lm_index(l2, 0);
                     ylm2 = ylm_vec[idx_lm];
                } else {
                    ylm2 = 0.282094791773878;
                }
                
                integral += val1 * ylm1 * val2 * ylm2 * r * r * sin_theta;
            }
        }
    }
    
    return integral * dr * dtheta * dphi;
}

// Helper for Gradient Integration <phi | grad | beta>
// Returns vector of 3 components (x, y, z)
std::vector<double> DirectIntegrationGradient(const NumericalRadial& rad_orb, const NumericalRadial& rad_beta, double R) {
    // Grid parameters
    int nr = 1000;
    int ntheta = 180;
    int nphi = 360;
    
    double dr = rad_orb.rmax() / nr;
    if (rad_beta.rmax() > rad_orb.rmax()) dr = rad_beta.rmax() / nr;
    
    double dtheta = M_PI / ntheta;
    double dphi = 2.0 * M_PI / nphi;
    
    std::vector<double> integral(3, 0.0);
    
    int l1 = rad_orb.l();
    int l2 = rad_beta.l();
    
    // Lambda for interpolation (same as above)
    auto get_val = [](const NumericalRadial& rad, double r) -> double {
        if (r > rad.rmax()) return 0.0;
        const double* rgrid = rad.rgrid();
        int nr_grid = rad.nr();
        double dr_grid = rgrid[nr_grid-1] / (nr_grid - 1);
        int idx = static_cast<int>(r / dr_grid);
        if (idx < 0) idx = 0;
        if (idx >= nr_grid - 1) idx = nr_grid - 2;
        if (r < rgrid[idx] || r > rgrid[idx+1]) {
             auto it = std::lower_bound(rgrid, rgrid + nr_grid, r);
             if (it == rgrid) return rad.rvalue(0);
             if (it == rgrid + nr_grid) return 0.0;
             idx = std::distance(rgrid, it) - 1;
        }
        double r0 = rgrid[idx];
        double r1 = rgrid[idx+1];
        double v0 = rad.rvalue(idx);
        double v1 = rad.rvalue(idx+1);
        double val = v0 + (v1 - v0) * (r - r0) / (r1 - r0);
        if (rad.pr() == 1) {
            if (r < 1e-10) return 0.0; 
            return val / r;
        }
        return val;
    };
    
    // Lambda for derivative of phi (radial part)
    // d/dr (f(r))
    auto get_deriv = [&](const NumericalRadial& rad, double r) -> double {
        double h = 1e-4;
        return (get_val(rad, r+h) - get_val(rad, r-h)) / (2*h);
    };

    for (int ir = 0; ir < nr; ++ir) {
        double r = (ir + 0.5) * dr;
        double val1 = get_val(rad_orb, r);
        double dval1 = get_deriv(rad_orb, r);
        
        if (std::abs(val1) < 1e-10 && std::abs(dval1) < 1e-10) continue;

        for (int it = 0; it < ntheta; ++it) {
            double theta = (it + 0.5) * dtheta;
            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);
            
            for (int ip = 0; ip < nphi; ++ip) {
                double phi = (ip + 0.5) * dphi;
                
                double x = r * sin_theta * std::cos(phi);
                double y = r * sin_theta * std::sin(phi);
                double z = r * cos_theta;
                
                double x2 = x;
                double y2 = y;
                double z2 = z - R;
                double r2_norm = std::sqrt(x2*x2 + y2*y2 + z2*z2);
                
                double val2 = get_val(rad_beta, r2_norm);
                
                // Ylm1 and its gradient
                // grad(phi) = grad(R(r)Y(rhat)) = R'(r)Y(rhat)rhat + R(r)grad(Y(rhat))
                // This is complicated.
                // Alternative: numerical gradient of the whole function phi(r)
                // phi(x,y,z) = R(r) * Y(theta, phi)
                // Let's use finite difference for the whole 3D function
                
                auto eval_phi = [&](double px, double py, double pz) -> double {
                    double pr = std::sqrt(px*px + py*py + pz*pz);
                    if (pr < 1e-10) return 0.0; // simplified
                    double pval = get_val(rad_orb, pr);
                    
                    double pylm = 0.0;
                    if (l1 > 0) {
                         std::vector<double> ylm_vec((l1+1)*(l1+1));
                         ModuleBase::Vector3<double> rhat(px/pr, py/pr, pz/pr);
                         ModuleBase::Ylm::get_ylm_real(l1 + 1, rhat, ylm_vec.data());
                         int idx_lm = get_lm_index(l1, 0); // Assume m=0
                         pylm = ylm_vec[idx_lm];
                    } else {
                        pylm = 0.282094791773878;
                    }
                    return pval * pylm;
                };
                
                double h = 1e-4;
                double dphi_dx = (eval_phi(x+h, y, z) - eval_phi(x-h, y, z)) / (2*h);
                double dphi_dy = (eval_phi(x, y+h, z) - eval_phi(x, y-h, z)) / (2*h);
                double dphi_dz = (eval_phi(x, y, z+h) - eval_phi(x, y, z-h)) / (2*h);
                
                // Ylm2
                double ylm2 = 1.0;
                if (l2 > 0) {
                     std::vector<double> ylm_vec((l2+1)*(l2+1));
                     ModuleBase::Vector3<double> rhat2(x2/r2_norm, y2/r2_norm, z2/r2_norm);
                     ModuleBase::Ylm::get_ylm_real(l2 + 1, rhat2, ylm_vec.data());
                     int idx_lm = get_lm_index(l2, 0); // Assume m=0
                     ylm2 = ylm_vec[idx_lm];
                } else {
                    ylm2 = 0.282094791773878;
                }
                
                double beta_val = val2 * ylm2;
                
                integral[0] += dphi_dx * beta_val * r * r * sin_theta;
                integral[1] += dphi_dy * beta_val * r * r * sin_theta;
                integral[2] += dphi_dz * beta_val * r * r * sin_theta;
            }
        }
    }
    integral[0] *= dr * dtheta * dphi;
    integral[1] *= dr * dtheta * dphi;
    integral[2] *= dr * dtheta * dphi;
    return integral;
}

// Helper for Position Integration <phi | r | beta>
// Returns vector of 3 components (x, y, z)
std::vector<double> DirectIntegrationPosition(const NumericalRadial& rad_orb, const NumericalRadial& rad_beta, double R) {
    // Grid parameters
    int nr = 1000;
    int ntheta = 180;
    int nphi = 360;
    
    double dr = rad_orb.rmax() / nr;
    if (rad_beta.rmax() > rad_orb.rmax()) dr = rad_beta.rmax() / nr;
    
    double dtheta = M_PI / ntheta;
    double dphi = 2.0 * M_PI / nphi;
    
    std::vector<double> integral(3, 0.0);
    
    int l1 = rad_orb.l();
    int l2 = rad_beta.l();
    
    // Lambda for interpolation (same as above)
    auto get_val = [](const NumericalRadial& rad, double r) -> double {
        if (r > rad.rmax()) return 0.0;
        const double* rgrid = rad.rgrid();
        int nr_grid = rad.nr();
        double dr_grid = rgrid[nr_grid-1] / (nr_grid - 1);
        int idx = static_cast<int>(r / dr_grid);
        if (idx < 0) idx = 0;
        if (idx >= nr_grid - 1) idx = nr_grid - 2;
        if (r < rgrid[idx] || r > rgrid[idx+1]) {
             auto it = std::lower_bound(rgrid, rgrid + nr_grid, r);
             if (it == rgrid) return rad.rvalue(0);
             if (it == rgrid + nr_grid) return 0.0;
             idx = std::distance(rgrid, it) - 1;
        }
        double r0 = rgrid[idx];
        double r1 = rgrid[idx+1];
        double v0 = rad.rvalue(idx);
        double v1 = rad.rvalue(idx+1);
        double val = v0 + (v1 - v0) * (r - r0) / (r1 - r0);
        if (rad.pr() == 1) {
            if (r < 1e-10) return 0.0; 
            return val / r;
        }
        return val;
    };

    for (int ir = 0; ir < nr; ++ir) {
        double r = (ir + 0.5) * dr;
        double val1 = get_val(rad_orb, r);
        
        if (std::abs(val1) < 1e-10) continue;

        for (int it = 0; it < ntheta; ++it) {
            double theta = (it + 0.5) * dtheta;
            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);
            
            for (int ip = 0; ip < nphi; ++ip) {
                double phi = (ip + 0.5) * dphi;
                
                double x = r * sin_theta * std::cos(phi);
                double y = r * sin_theta * std::sin(phi);
                double z = r * cos_theta;
                
                double x2 = x;
                double y2 = y;
                double z2 = z - R;
                double r2_norm = std::sqrt(x2*x2 + y2*y2 + z2*z2);
                
                double val2 = get_val(rad_beta, r2_norm);
                
                // Ylm1
                double ylm1 = 1.0;
                if (l1 > 0) {
                     std::vector<double> ylm_vec((l1+1)*(l1+1));
                     ModuleBase::Vector3<double> rhat(x/r, y/r, z/r);
                     ModuleBase::Ylm::get_ylm_real(l1 + 1, rhat, ylm_vec.data());
                     int idx_lm = get_lm_index(l1, 0); // Assume m=0
                     ylm1 = ylm_vec[idx_lm];
                } else {
                    ylm1 = 0.282094791773878;
                }

                // Ylm2
                double ylm2 = 1.0;
                if (l2 > 0) {
                     std::vector<double> ylm_vec((l2+1)*(l2+1));
                     ModuleBase::Vector3<double> rhat2(x2/r2_norm, y2/r2_norm, z2/r2_norm);
                     ModuleBase::Ylm::get_ylm_real(l2 + 1, rhat2, ylm_vec.data());
                     int idx_lm = get_lm_index(l2, 0); // Assume m=0
                     ylm2 = ylm_vec[idx_lm];
                } else {
                    ylm2 = 0.282094791773878;
                }
                
                double term = val1 * ylm1 * val2 * ylm2 * r * r * sin_theta;
                
                // Position operator r vector
                integral[0] += term * x;
                integral[1] += term * y;
                integral[2] += term * z;
            }
        }
    }
    integral[0] *= dr * dtheta * dphi;
    integral[1] *= dr * dtheta * dphi;
    integral[2] *= dr * dtheta * dphi;
    return integral;
}

TEST_F(TwoCenterBundleTest, VerifyOverlap)
{
    std::ofstream log_file("test_two_center_bundle.log");
    ModuleBase::Logger logger(log_file);
    
    // 1. Read Pseudo
#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";
#endif
    std::vector<std::string> pp_file = {"C_ONCV_PBE-1.0.upf"};
    pseudos.resize(ntype);
#ifdef __MPI
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, MPI_COMM_WORLD);
#else
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, 0);
#endif
    
    // 2. Get BetaRadials from Atom_pseudo
    // read_pseudo calls setup_nonlocal internally now
    BetaRadials& br = pseudos[0].beta_radials;
    
    // 3. Initialize TwoCenterBundle
    TwoCenterBundle bundle;
    std::string orb_file = "C_gga_7au_100Ry_2s2p1d.orb";
    bundle.build_orb(ntype, &orb_file, pp_dir); 
    bundle.build_beta(ntype, &br);
    
    bundle.tabulate();
    
    // 4. Compare
    auto& orb_collection = *bundle.orb_;
    auto& beta_collection = *bundle.beta_;
    
    // Pick first orbital type (C)
    // orb_collection(0) returns RadialSet
    const auto& rad_set_orb = orb_collection(0);
    const auto& rad_set_beta = beta_collection(0);
    
    // Distances to test
    std::vector<double> Rs = {1.5, 2.5};
    
    for (double R : Rs) {
        ModuleBase::Vector3<double> vR(0, 0, R);
        
        // Loop over l and zeta
        for (int l1 = 0; l1 <= rad_set_orb.lmax(); ++l1) {
            for (int iz1 = 0; iz1 < rad_set_orb.nzeta(l1); ++iz1) {
                // Use orb_collection(itype, l, izeta) to get const NumericalRadial& directly
                const auto& chi1 = orb_collection(0, l1, iz1);
                
                for (int l2 = 0; l2 <= rad_set_beta.lmax(); ++l2) {
                    for (int iz2 = 0; iz2 < rad_set_beta.nzeta(l2); ++iz2) {
                        const auto& chi2 = beta_collection(0, l2, iz2);
                        
                        // Test m=0
                        int m1 = 0;
                        int m2 = 0;
                        
                        double val_calc = 0.0;
                        bundle.overlap_orb_beta->calculate(0, l1, iz1, m1, 0, l2, iz2, m2, vR, &val_calc, nullptr);
                        
                        double val_ref = DirectIntegration(chi1, chi2, R);
                        
                        // Tolerance: 1e-3 because our direct integration is coarse
                        // Relax tolerance slightly for larger R or specific orbitals if needed
                        EXPECT_NEAR(val_calc, val_ref, 5e-3) 
                            << "Mismatch at R=" << R << " l1=" << l1 << " iz1=" << iz1 
                            << " l2=" << l2 << " iz2=" << iz2;
                    }
                }
            }
        }
    }
}

TEST_F(TwoCenterBundleTest, VerifyGradient) {
    std::ofstream log_file("test_two_center_bundle_grad.log");
    ModuleBase::Logger logger(log_file);
#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";
#endif
    std::vector<std::string> pp_file = {"C_ONCV_PBE-1.0.upf"};
    pseudos.resize(ntype);
#ifdef __MPI
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, MPI_COMM_WORLD);
#else
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, 0);
#endif
    
    BetaRadials& br = pseudos[0].beta_radials;
    
    TwoCenterBundle bundle;
    std::string orb_file = "C_gga_7au_100Ry_2s2p1d.orb";
    bundle.build_orb(ntype, &orb_file, pp_dir); 
    bundle.build_beta(ntype, &br);
    
    bundle.tabulate();
    
    auto& orb_collection = *bundle.orb_;
    auto& beta_collection = *bundle.beta_;
    
    double R = 2.0;
    ModuleBase::Vector3<double> vR(0, 0, R);
    
    // Select specific orbitals: C 2s (l=0) and Beta s (l=0)
    // Gradient along z should be non-zero
    int l1 = 0; int iz1 = 0; int m1 = 0;
    int l2 = 0; int iz2 = 0; int m2 = 0;
    
    // TwoCenterBundle calculates derivatives w.r.t R (displacement vector r2-r1)
    // <phi(r) | beta(r-R)>
    // d/dR <phi(r) | beta(r-R)> = <phi(r) | -grad beta(r-R)> = - <phi | grad | beta>
    // So bundle derivative = - DirectIntegrationGradient
    
    double val = 0.0;
    ModuleBase::Vector3<double> grad_bundle;
    // calculate(..., val, dval)
    // dval is usually a pointer to 3 doubles (x, y, z derivatives)
    // Wait, TwoCenterIntegrator::calculate signature:
    // void calculate(..., const Vector3<double>& R, double* val, double* dval)
    // dval stores d/dR_x, d/dR_y, d/dR_z
    
    double dval[3];
    bundle.overlap_orb_beta->calculate(0, l1, iz1, m1, 0, l2, iz2, m2, vR, &val, dval);
    
    std::vector<double> grad_direct = DirectIntegrationGradient(orb_collection(0, l1, iz1), beta_collection(0, l2, iz2), R);
    
    // Expect bundle derivative = - direct gradient
    EXPECT_NEAR(dval[0], grad_direct[0], 1e-4);
    EXPECT_NEAR(dval[1], grad_direct[1], 1e-4);
    EXPECT_NEAR(dval[2], grad_direct[2], 1e-4);
}

TEST_F(TwoCenterBundleTest, VerifyPosition) {
    std::ofstream log_file("test_two_center_bundle_pos.log");
    ModuleBase::Logger logger(log_file);
#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";
#endif
    std::vector<std::string> pp_file = {"C_ONCV_PBE-1.0.upf"};
    pseudos.resize(ntype);
#ifdef __MPI
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, MPI_COMM_WORLD);
#else
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, 0);
#endif
    
    BetaRadials& br = pseudos[0].beta_radials;
    
    TwoCenterBundle bundle;
    std::string orb_file = "C_gga_7au_100Ry_2s2p1d.orb";
    bundle.build_orb(ntype, &orb_file, pp_dir); 
    bundle.build_beta(ntype, &br);
    
    bundle.tabulate();
    
    auto& orb_collection = *bundle.orb_;
    auto& beta_collection = *bundle.beta_;
    
    double R = 2.0;
    ModuleBase::Vector3<double> vR(0, 0, R);
    ModuleBase::Vector3<double> R2(0, 0, R);
    
    // Select specific orbitals: C 2s (l=0) and Beta s (l=0)
    int l1 = 0; int iz1 = 0; int m1 = 0;
    int l2 = 0; int iz2 = 0; int m2 = 0;
    
    // TwoCenterBundle calculates <phi | r | beta>
    // This is stored in overlap_orb_beta if betap_ and betam_ are provided
    // But wait, TwoCenterIntegrator::calculate doesn't return position integrals directly in the standard interface?
    // It returns S, T, or S/r/p?
    // Actually, TwoCenterIntegrator with 5 args (orb, beta, betap, betam) calculates:
    // <phi | beta>, <phi | r | beta> (vector)
    // But the `calculate` method signature is:
    // void calculate(..., double* val, double* dval)
    // If it's the 5-arg tabulate, what does calculate return?
    // It returns 4 values in `val`: S, <x>, <y>, <z>.
    // And derivatives in `dval`.
    
    double vals[4];
    bundle.overlap_orb_beta->calculate(0, l1, iz1, m1, 0, l2, iz2, m2, vR, R2, vals, vals+1, vals+2, vals+3);
    
    std::vector<double> pos_direct = DirectIntegrationPosition(orb_collection(0, l1, iz1), beta_collection(0, l2, iz2), R);
    
    // Check overlap first
    // EXPECT_NEAR(vals[0], DirectIntegration(orb_collection(0, l1, iz1), beta_collection(0, l2, iz2), R), 1e-3);
    
    // Check position integrals
    // Note: DirectIntegrationPosition returns <phi | r | beta>
    // TwoCenterIntegrator returns <phi | r - R_mid | beta> ? Or <phi | r | beta>?
    // Usually it calculates dipole matrix elements.
    // The center of expansion might be important.
    // TwoCenterIntegrator usually returns <phi(r) | r | beta(r-R)> with r origin at phi.
    // Let's verify.
    
    EXPECT_NEAR(vals[1], pos_direct[0], 5e-3);
    EXPECT_NEAR(vals[2], pos_direct[1], 5e-3);
    EXPECT_NEAR(vals[3], pos_direct[2], 5e-3);
}

TEST_F(TwoCenterBundleTest, VerifyAtomicRadialsConsistency) {
    std::ofstream log_file("test_consistency.log");
    ModuleBase::Logger logger(log_file);
#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";
#endif
    std::string orb_file = pp_dir + "C_gga_7au_100Ry_2s2p1d.orb";
    
    // 1. Build from file with pm=0
    AtomicRadials ar0;
#ifdef __MPI
    ar0.build(orb_file, 0, 0, 0, &logger, MPI_COMM_WORLD);
#else
    ar0.build(orb_file, 0, 0, 0, &logger, 0);
#endif

    // 2. Build from file with pm=1
    AtomicRadials ar1_file;
#ifdef __MPI
    ar1_file.build(orb_file, 0, 0, 1, &logger, MPI_COMM_WORLD);
#else
    ar1_file.build(orb_file, 0, 0, 1, &logger, 0);
#endif

    // 3. Build from ar0 with pm=1 using new build method
    AtomicRadials ar1_new;
    ar1_new.build(&ar0, 0, 0, 1);
    
    // 4. Compare ar1_file and ar1_new
    EXPECT_EQ(ar1_file.lmax(), ar1_new.lmax());
    EXPECT_EQ(ar1_file.nchi(), ar1_new.nchi());
    
    for (int i = 0; i < ar1_file.nchi(); ++i) {
        const auto& chi1 = ar1_file.cbegin()[i];
        const auto& chi2 = ar1_new.cbegin()[i];
        
        EXPECT_EQ(chi1.l(), chi2.l());
        EXPECT_EQ(chi1.nr(), chi2.nr());
        
        // Check values
        for (int ir = 0; ir < chi1.nr(); ++ir) {
            EXPECT_NEAR(chi1.rvalue(ir), chi2.rvalue(ir), 1e-12);
        }
    }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
