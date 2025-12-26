#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include "source_estate/read_pseudo.h"
// #include "source_cell/atom_spec.h"
#include "source_basis/module_nao/numerical_radial.h"
#include "source_base/log.h"
// #include "source_base/global_variable.h"

namespace GlobalV {
    std::ofstream ofs_running;
}

#ifdef __MPI
#include <mpi.h>
#endif

class PseudoTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup Atom
        ntype = 1;
        // atoms_vec[0].ncpp.ps_file = "../test/pporb/C_ONCV_PBE-1.0.upf";
        
        // Redirect GlobalV::ofs_running to avoid clutter
        GlobalV::ofs_running.open("test_pseudo_global.log");
    }

    void TearDown() override
    {
        GlobalV::ofs_running.close();
    }

    int ntype;
    std::vector<Atom_pseudo> pseudos;
};

TEST_F(PseudoTest, ReadPseudoAndVerify)
{
    std::ofstream log_file("test_pseudo.log");
    ModuleBase::Logger logger(log_file);
    
    pseudos.resize(ntype);

    // 1. Read Pseudo
#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/";
#endif
    std::vector<std::string> pp_file = {"C_ONCV_PBE-1.0.upf"};
#ifdef __MPI
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, MPI_COMM_WORLD);
#else
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, 0);
#endif
    
    ASSERT_GT(pseudos[0].nbeta, 0);
    
    // 2. Verify BetaRadials
    // Since read_pseudo now calls setup_nonlocal, beta_radials should be populated.
    const BetaRadials& br = pseudos[0].beta_radials;
    
    EXPECT_EQ(br.lmax(), pseudos[0].lmax);
    // EXPECT_EQ(br.nchi(), pseudos[0].nbeta); // nchi in BetaRadials corresponds to number of projectors?
    // Let's check BetaRadials::nchi(). It is total number of radial functions.
    // For BetaRadials, it should be nbeta.
    
    // Verify some values if possible.
    // We can check if rcut is reasonable.
    EXPECT_GT(br.rcut_max(), 0.0);
    
    // Check first projector
    const NumericalRadial& proj = br.cbegin()[0];
    EXPECT_EQ(proj.l(), pseudos[0].lll[0]);
    
    // Check radial grid
    int nr = proj.nr();
    EXPECT_GT(nr, 0);
    EXPECT_NEAR(proj.rgrid(nr-1), pseudos[0].r[nr-1], 1e-4); // Approximate check
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
