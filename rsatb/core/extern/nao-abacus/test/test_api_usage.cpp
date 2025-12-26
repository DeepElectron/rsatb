#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>

#include "source_estate/read_pseudo.h"
#include "source_cell/atom_pseudo.h"
#include "source_base/log.h"

// Simple test to demonstrate the new API usage
TEST(ApiUsageTest, LoadPseudoAndCheckBetaRadials)
{
    std::ofstream log_file("test_api_usage.log");
    ModuleBase::Logger logger(log_file);

#ifdef TEST_DATA_DIR
    std::string pp_dir = TEST_DATA_DIR;
#else
    std::string pp_dir = "./test/pporb/";
#endif
    std::vector<std::string> pp_file = {"C_ONCV_PBE-1.0.upf"};

    // 1. Create a vector of Atom_pseudo
    std::vector<Atom_pseudo> pseudos;
    int ntype = 1;
    pseudos.resize(ntype);

    // 2. Read pseudopotentials
    // This function now handles reading and setting up non-local parts (BetaRadials)
#ifdef __MPI
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, MPI_COMM_WORLD);
#else
    read_pseudo(pp_dir, pp_file, "auto", 10.0, false, 0.0, logger, pseudos, 0);
#endif

    // 3. Verify that BetaRadials are initialized
    const auto& pp = pseudos[0];
    const auto& beta_radials = pp.beta_radials;

    // Check basic properties
    EXPECT_GT(beta_radials.lmax(), -1);
    EXPECT_GT(beta_radials.nchi(), 0);
    
    // Check that we can access radial functions
    for (int i = 0; i < beta_radials.nchi(); ++i) {
        const auto& radial = beta_radials.cbegin()[i];
        EXPECT_GT(radial.nr(), 0);
        EXPECT_GT(radial.rmax(), 0.0);
    }

    std::cout << "Successfully loaded pseudopotential and initialized BetaRadials." << std::endl;
    std::cout << "Lmax: " << beta_radials.lmax() << std::endl;
    std::cout << "Number of projectors: " << beta_radials.nchi() << std::endl;
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
