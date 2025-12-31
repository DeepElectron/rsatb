#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>

#include "source_estate/op2c.hpp"
#include "source_base/matrix.h"

// Test Op2c functionality
TEST(Op2cTest, BasicFunctionality)
{
#ifdef TEST_DATA_DIR
    std::string data_dir = TEST_DATA_DIR;
#else
    std::string data_dir = "./test/pporb/";
#endif
    std::string orb_dir = data_dir;
    std::string psd_dir = data_dir;
    std::vector<std::string> orb_name = {"C_gga_7au_100Ry_2s2p1d.orb"};
    std::vector<std::string> psd_name = {"C_ONCV_PBE-1.0.upf"};
    std::string log_file = "test_op2c.log";

    int ntype = 1;
    int nspin = 1;
    bool lspinorb = false;

#ifdef __MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    int comm = 0;
#endif

    // 1. Initialize Op2c
    Op2c op2c(ntype, nspin, lspinorb, orb_dir, orb_name, psd_dir, psd_name, comm, log_file);

    // 2. Test overlap
    size_t itype = 0;
    size_t jtype = 0;
    ModuleBase::Vector3<double> Rij(1.0, 0.0, 0.0);
    bool is_transpose = false;
    std::vector<double> v;
    std::vector<double> dvx, dvy, dvz;

    op2c.overlap(itype, jtype, Rij, is_transpose, v, &dvx, &dvy, &dvz);
    
    EXPECT_GT(v.size(), 0);
    EXPECT_GT(dvx.size(), 0);
    EXPECT_GT(dvy.size(), 0);
    EXPECT_GT(dvz.size(), 0);

    // 3. Test overlap_position
    ModuleBase::Vector3<double> Ri(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> Rj(1.0, 0.0, 0.0);
    std::vector<double> v_pos, vx, vy, vz;
    
    op2c.overlap_position(itype, jtype, Ri, Rj, is_transpose, v_pos, vx, vy, vz);

    EXPECT_GT(v_pos.size(), 0);
    EXPECT_GT(vx.size(), 0);
    EXPECT_GT(vy.size(), 0);
    EXPECT_GT(vz.size(), 0);

    // 4. Test orb_r_beta
    std::vector<size_t> itypes = {0};
    size_t ktype = 0;
    std::vector<ModuleBase::Vector3<double>> Ris = {Ri};
    ModuleBase::Vector3<double> Rk(0.5, 0.0, 0.0);
    std::vector<ModuleBase::matrix> ob(1), oxb(1), oyb(1), ozb(1);

    op2c.orb_r_beta(itypes, ktype, Ris, Rk, is_transpose, ob, oxb, oyb, ozb);

    EXPECT_GT(ob[0].nr * ob[0].nc, 0);

    // 5. Test ncomm_IKJ
    // 5. Test ncomm_IKJ
    std::vector<std::complex<double>> ncomm_vx(v.size()), ncomm_vy(v.size()), ncomm_vz(v.size());
    
    op2c.ncomm_IKJ(itype, 0, ktype, jtype, 0, ob, oxb, oyb, ozb, 1, is_transpose, ncomm_vx, ncomm_vy, ncomm_vz);
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    std::cout << "Exiting main..." << std::endl;
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
