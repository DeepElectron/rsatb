#include <gtest/gtest.h>
#ifdef __MPI
#include <mpi.h>
#endif
#include "source_basis/module_nao/two_center_bundle.h"
#include <iostream>

class TwoCenterBundleTest : public ::testing::Test
{
  protected:
    void SetUp();
    void TearDown();
    int rank=0;

    TwoCenterBundle bundle;
};

void TwoCenterBundleTest::SetUp()
{
#ifdef __MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    std::string dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";

    int nfile_orb = 3;
    std::string* file_orb = new std::string[nfile_orb];
    file_orb[0] = "C_gga_7au_100Ry_2s2p1d.orb";
    file_orb[1] = "Fe_gga_7au_100Ry_4s2p2d1f.orb";
    file_orb[2] = "O_gga_7au_100Ry_2s2p1d.orb";

    int nfile_desc = 0;

    bundle.build_orb(nfile_orb, file_orb, dir);
    bundle.tabulate();

    delete[] file_orb;
}

void TwoCenterBundleTest::TearDown()
{
}

TEST_F(TwoCenterBundleTest, Build)
{
    TwoCenterIntegrator* ovl = bundle.overlap_orb.release();

    ModuleBase::Vector3<double> vR0 = {0.0, 0.0, 0.0};
    double out;
    double grad[3];

    ovl->calculate(0, 0, 0, 0, 0, 0, 0, 0, vR0, &out, grad);

    std::cout << "Succ calculate!" << std::endl;
    std::cout << "out = " << out << " from rank: " << rank << std::endl;
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