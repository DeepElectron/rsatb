#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_BASIS_MODULE_NAO_TWO_CENTER_BUNDLE_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_BASIS_MODULE_NAO_TWO_CENTER_BUNDLE_H

// #include "source_basis/module_ao/ORB_read.h"
#include "source_basis/module_nao/two_center_integrator.h"

#include <memory>
#include <string>

class TwoCenterBundle
{
  public:
    TwoCenterBundle() = default;
    ~TwoCenterBundle() = default;
    TwoCenterBundle& operator=(TwoCenterBundle&&) = default;

    // NOTE: some variables might be set only on RANK-0
    void build_orb(int ntype, const std::string* file_orb0, const std::string& orbital_dir, MPI_Comm comm=MPI_COMM_WORLD);
    // void build_beta(int ntype, Numerical_Nonlocal* nl);
    void build_beta(int ntype, BetaRadials* nl);

    void tabulate();

    // Unlike the tabulate() above, this overload function computes
    // two-center integration table by direct integration with Simpson's
    // rule, which was the algorithm used prior to v3.3.4.
    void tabulate(const double lcao_ecut, const double lcao_dk, const double lcao_dr, const double lcao_rmax);

    std::unique_ptr<TwoCenterIntegrator> kinetic_orb;
    std::unique_ptr<TwoCenterIntegrator> overlap_orb;
    std::unique_ptr<TwoCenterIntegrator> overlap_orb_beta;
    // std::unique_ptr<TwoCenterIntegrator> overlap_orb_alpha;

    std::unique_ptr<RadialCollection> orb_, orbp_, orbm_;
    // std::unique_ptr<RadialCollection> beta_;
    // std::unique_ptr<RadialCollection> alpha_;
    std::unique_ptr<RadialCollection> beta_;
    std::unique_ptr<RadialCollection> betap_;
    std::unique_ptr<RadialCollection> betam_;
};

#endif
