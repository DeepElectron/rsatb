#include "source_basis/module_nao/two_center_bundle.h"

#include "source_base/memory.h"
#include "source_base/parallel_common.h"
#include "source_base/ylm.h"
#include "source_basis/module_nao/real_gaunt_table.h"
#include <memory>

void TwoCenterBundle::build_orb(int ntype, const std::string* file_orb0, const std::string& orbital_dir, MPI_Comm comm)
{
    std::vector<std::string> file_orb(ntype);
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif
    if (rank == 0)
    {
        std::transform(file_orb0, file_orb0 + ntype, file_orb.begin(), [&](const std::string& file) {
            return orbital_dir + file;
        });
    }
#ifdef __MPI
    Parallel_Common::bcast_string(file_orb.data(), ntype, comm);
#endif
    
    orb_ = std::unique_ptr<RadialCollection>(new RadialCollection);
    orbp_ = std::unique_ptr<RadialCollection>(new RadialCollection);
    orbm_ = std::unique_ptr<RadialCollection>(new RadialCollection);

    orb_->build(ntype, file_orb.data(), '\0', 0, 0, comm); // automatically detect file type
    orbp_->build(ntype, file_orb.data(), '\0', -1, 1, comm);
    orbm_->build(ntype, file_orb.data(), '\0', -1, -1, comm);

    // std::cout << "rank: " << rank << " 2CBuddle build_orb done" << std::endl;
}

void TwoCenterBundle::tabulate()
{
    ModuleBase::SphericalBesselTransformer sbt(true);
    orb_->set_transformer(sbt);
    if (beta_) { 
        beta_->set_transformer(sbt); 
        if (betap_) {
            betap_->set_transformer(sbt);
            betam_->set_transformer(sbt);
        }
    }
    if (orbp_) {
        orbp_->set_transformer(sbt);
        orbm_->set_transformer(sbt);
    }
    //================================================================
    //              build two-center integration tables
    //================================================================
    // set up a universal radial grid
    double rmax = orb_->rcut_max();
    if (beta_) { rmax = std::max(rmax, beta_->rcut_max()); }
    double dr = 0.01;
    double cutoff = 2.0 * rmax;
    int nr = static_cast<int>(rmax / dr) + 1;

    orb_->set_uniform_grid(true, nr, cutoff, 'i', true);
    if (beta_) { 
        beta_->set_uniform_grid(true, nr, cutoff, 'i', true); 
        if (betap_) {
            betap_->set_uniform_grid(true, nr, cutoff, 'i', true);
            betam_->set_uniform_grid(true, nr, cutoff, 'i', true);
        }
    }
    if (orbp_) { orbp_->set_uniform_grid(true, nr, cutoff, 'i', true); }
    if (orbm_) { orbm_->set_uniform_grid(true, nr, cutoff, 'i', true); }

    // build TwoCenterIntegrator objects
    // kinetic_orb = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
    // kinetic_orb->tabulate(*orb_, *orb_, 'T', nr, cutoff);
    // ModuleBase::Memory::record("TwoCenterTable: Kinetic", kinetic_orb->table_memory());

    overlap_orb = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
    overlap_orb->tabulate(*orb_, *orbp_, *orbm_, nr, cutoff);
    ModuleBase::Memory::record("TwoCenterTable: Overlap & Position Op", overlap_orb->table_memory());

    if (beta_)
    {
        overlap_orb_beta = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
        if (betap_) {
            overlap_orb_beta->tabulate(*orb_, *beta_, *betap_, *betam_, nr, cutoff);
        } else {
            overlap_orb_beta->tabulate(*orb_, *beta_, 'S', nr, cutoff);
        }
        ModuleBase::Memory::record("TwoCenterTable: Nonlocal", overlap_orb_beta->table_memory());
    }

    ModuleBase::Memory::record("RealGauntTable", RealGauntTable::instance().memory());

    sbt.clear();
}

void TwoCenterBundle::tabulate(const double lcao_ecut,
                               const double lcao_dk,
                               const double lcao_dr,
                               const double lcao_rmax)
{
    ModuleBase::SphericalBesselTransformer sbt(true);
    orb_->set_transformer(sbt);
    if (beta_) { 
        beta_->set_transformer(sbt); 
        if (betap_) {
            betap_->set_transformer(sbt);
            betam_->set_transformer(sbt);
        }
    }
    if (orbp_) {
        orbp_->set_transformer(sbt);
        orbm_->set_transformer(sbt);
    }

    //================================================================
    //              build two-center integration tables
    //================================================================

    // old formula for the number of k-space grid points
    int nk = static_cast<int>(sqrt(lcao_ecut) / lcao_dk) + 4;
    nk += 1 - nk % 2; // make nk odd

    std::vector<double> kgrid(nk);
    for (int ik = 0; ik < nk; ++ik)
    {
        kgrid[ik] = ik * lcao_dk;
    }

    orb_->set_grid(false, nk, kgrid.data(), 't');
    if (beta_) { 
        beta_->set_grid(false, nk, kgrid.data(), 't'); 
        if (betap_) {
            betap_->set_grid(false, nk, kgrid.data(), 't');
            betam_->set_grid(false, nk, kgrid.data(), 't');
        }
    }
    if (orbp_) { orbp_->set_grid(false, nk, kgrid.data(), 't'); }
    if (orbm_) { orbm_->set_grid(false, nk, kgrid.data(), 't'); }

    // "st" stands for overlap (s) and kinetic (t)
    const double cutoff_st = std::min(lcao_rmax, 2.0 * orb_->rcut_max());
    const int nr_st = static_cast<int>(cutoff_st / lcao_dr) + 5;

    // kinetic_orb = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
    // kinetic_orb->tabulate(*orb_, *orb_, 'T', nr_st, cutoff_st);
    // ModuleBase::Memory::record("TwoCenterTable: Kinetic", kinetic_orb->table_memory());

    overlap_orb = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
    overlap_orb->tabulate(*orb_, *orbp_, *orbm_, nr_st, cutoff_st);
    ModuleBase::Memory::record("TwoCenterTable: Overlap & Position Op", overlap_orb->table_memory());
    ModuleBase::Memory::record("RealGauntTable", RealGauntTable::instance().memory());

    if (beta_)
    {
        // overlap between orbital and beta (for nonlocal potential)
        const double cutoff_nl = std::min(lcao_rmax, orb_->rcut_max() + beta_->rcut_max());
        const int nr_nl = static_cast<int>(cutoff_nl / lcao_dr) + 5;
        overlap_orb_beta = std::unique_ptr<TwoCenterIntegrator>(new TwoCenterIntegrator);
        if (betap_) {
            overlap_orb_beta->tabulate(*orb_, *beta_, *betap_, *betam_, nr_nl, cutoff_nl);
        } else {
            overlap_orb_beta->tabulate(*orb_, *beta_, 'S', nr_nl, cutoff_nl);
        }
        ModuleBase::Memory::record("TwoCenterTable: Nonlocal", overlap_orb_beta->table_memory());
    }
    sbt.clear();
}

// void TwoCenterBundle::build_beta(int ntype, Numerical_Nonlocal* nl)
// {
//     beta_ = std::unique_ptr<RadialCollection>(new RadialCollection);
//     beta_->build(ntype, nl);
    
//     betap_ = std::unique_ptr<RadialCollection>(new RadialCollection);
//     betam_ = std::unique_ptr<RadialCollection>(new RadialCollection);
//     // p = beta_->p() - 1, pm = 1
//     betap_->build(beta_.get(), beta_->p() - 1, 1);
//     // p = beta_->p() - 1, pm = -1
//     betam_->build(beta_.get(), beta_->p() - 1, -1);
// }

void TwoCenterBundle::build_beta(int ntype, BetaRadials* nl)
{
    beta_ = std::unique_ptr<RadialCollection>(new RadialCollection);
    beta_->build(ntype, nl);
    
    betap_ = std::unique_ptr<RadialCollection>(new RadialCollection);
    betam_ = std::unique_ptr<RadialCollection>(new RadialCollection);
    // p = beta_->p() - 1, pm = 1
    betap_->build(beta_.get(), beta_->p() - 1, 1);
    // p = beta_->p() - 1, pm = -1
    betam_->build(beta_.get(), beta_->p() - 1, -1);
}

// void TwoCenterBundle::to_LCAO_Orbitals(LCAO_Orbitals& ORB,
//                                        const double lcao_ecut,
//                                        const double lcao_dk,
//                                        const double lcao_dr,
//                                        const double lcao_rmax) const
// {
//     ORB.ntype = orb_->ntype();
//     ORB.lmax = orb_->lmax();
//     ORB.nchimax = orb_->nzeta_max();
//     ORB.rcutmax_Phi = orb_->rcut_max();
//     ORB.dR = lcao_dr;
//     ORB.Rmax = lcao_rmax;
//     ORB.dr_uniform = 0.001;

//     // Due to algorithmic difference in the spherical Bessel transform
//     // (FFT vs. Simpson's integration), k grid of FFT is not appropriate
//     // for Simpson's integration. The k grid for Simpson's integration is
//     // specifically set by the two variables below to mimick the original
//     // behavior.
//     ORB.ecutwfc = lcao_ecut;
//     ORB.dk = lcao_dk;

//     if (ORB.ecutwfc < 20)
//     {
//         ORB.kmesh = static_cast<int>(2 * sqrt(ORB.ecutwfc) / ORB.dk) + 4;
//     }
//     else
//     {
//         ORB.kmesh = static_cast<int>(sqrt(ORB.ecutwfc) / ORB.dk) + 4;
//     }
//     ORB.kmesh += 1 - ORB.kmesh % 2;

//     delete[] ORB.Phi;
//     ORB.Phi = new Numerical_Orbital[orb_->ntype()];
//     for (int itype = 0; itype < orb_->ntype(); ++itype)
//     {
//         (*orb_)(itype).to_numerical_orbital(ORB.Phi[itype], ORB.kmesh, ORB.dk);
//     }
// }
