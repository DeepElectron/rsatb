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

    int nfile_orb = 1;
    std::string* file_orb = new std::string[nfile_orb];
    file_orb[0] = "C_gga_7au_100Ry_2s2p1d.orb";

    int nfile_desc = 0;

    bundle.build_orb(nfile_orb, file_orb, dir);
    bundle.tabulate();

    delete[] file_orb;
}

void TwoCenterBundleTest::TearDown()
{
}

TEST_F(TwoCenterBundleTest, Hermiticity)
{
    TwoCenterIntegrator* ovl = bundle.overlap_orb.get();

    ModuleBase::Vector3<double> R1 = {0.0, 0.0, 0.0};
    ModuleBase::Vector3<double> R2 = {1.5, 0.5, 0.5};
    ModuleBase::Vector3<double> vR12 = R2 - R1;
    ModuleBase::Vector3<double> vR21 = R1 - R2;

    int itype = 0;
    int l = 1; 
    int izeta = 0;
    int m = 0;

    double S12, Rx12, Ry12, Rz12;
    double S21, Rx21, Ry21, Rz21;

    // Calculate <1|r|2>
    for(m=-l; m<=l; ++m){
        ovl->calculate(itype, l, izeta, m, itype, l, izeta, m, vR12, R2, &S12, &Rx12, &Ry12, &Rz12);

        // Calculate <2|r|1>
        ovl->calculate(itype, l, izeta, m, itype, l, izeta, m, vR21, R1, &S21, &Rx21, &Ry21, &Rz21);

        std::cout << "S12: " << S12 << ", S21: " << S21 << std::endl;
        std::cout << "Rx12: " << Rx12 << ", Rx21: " << Rx21 << std::endl;
        std::cout << "Ry12: " << Ry12 << ", Ry21: " << Ry21 << std::endl;
        std::cout << "Rz12: " << Rz12 << ", Rz21: " << Rz21 << std::endl;

        EXPECT_NEAR(S12, S21, 1e-8);
        EXPECT_NEAR(Rx12, Rx21, 1e-8);
        EXPECT_NEAR(Ry12, Ry21, 1e-8);
        EXPECT_NEAR(Rz12, Rz21, 1e-8);
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