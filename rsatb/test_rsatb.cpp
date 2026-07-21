#include "rsatb.hpp"
#include "test_data_path.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <mpi.h>

using namespace rsatb;

TEST(RSATBTest, TestSetupAndCompute) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Use npar=1 if size=1, npar=2 if size=2
     // std::vector<std::vector<double>> kpoints = {
    // {0.5 , 0.5 , 0.5 },
    // {0.45, 0.45, 0.45},
    // {0.4 , 0.4 , 0.4 },
    // {0.35, 0.35, 0.35},
    // {0.3 , 0.3 , 0.3 },
    // {0.25, 0.25, 0.25},
    // {0.2 , 0.2 , 0.2 },
    // {0.15, 0.15, 0.15},
    // {0.1 , 0.1 , 0.1 },
    // {0.05, 0.05, 0.05},
    // {0.  , 0.  , 0.  },
    // {0.05, 0.  , 0.05},
    // {0.1 , 0.  , 0.1 },
    // {0.15, 0.  , 0.15},
    // {0.2 , 0.  , 0.2 },
    // {0.25, 0.  , 0.25},
    // {0.3 , 0.  , 0.3 },
    // {0.35, 0.  , 0.35},
    // {0.4 , 0.  , 0.4 },
    // {0.45, 0.  , 0.45},
    // {0.5 , 0.  , 0.5 }
    // };

    std::vector<std::vector<double>> kpoints = {
        {0.  , 0.  , 0.  }
    };

    int nk = kpoints.size();
    int npar = (size >= 2) ? nk : 1;
    
    // 1. Instantiate RSATB
    std::unique_ptr<RSATB> rsatb_inst;

    const std::string vasp_file = rsatb::test::data_path("diamond/diamond_sc111.vasp");
    const std::vector<double> r_max_per_type = {3.7};
    const std::vector<int> type_norb = {13};
    
    rsatb_inst = std::make_unique<RSATB>(
        npar, vasp_file,
        r_max_per_type, type_norb, true, MPI_COMM_WORLD
    );
    
    if (rank == 0) {
        std::cout << "Selected Kernel: " << rsatb_inst->get_kernel_info() << std::endl;
    }

    // Buffers for merged data (must persist until RSATB is destroyed or at least constructed)
    
    // 2. Initialize Basis (needed for build_overlap)
    std::string orb_dir = rsatb::test::data_path("diamond/");
    std::vector<std::string> orb_names = {"C_gga_7au_100Ry_2s2p1d.orb"};
    std::string psd_dir = rsatb::test::data_path("diamond/");
    std::vector<std::string> psd_names = {"C_ONCV_PBE-1.0.upf"};

    std::string Hpath = rsatb::test::data_path("diamond/C2.0/hamiltonians.h5");

    model::SetupOptions setup_options;
    setup_options.kpoints = kpoints;
    setup_options.preconditioner.mode = "H";

    model::LCAOModelConfig model_config;
    model_config.input = model::Hdf5Input{};
    model_config.input->filename = Hpath;
    model_config.integrals.orb_names = orb_names;
    model_config.integrals.orb_dir = orb_dir;
    model_config.integrals.psd_names = psd_names;
    model_config.integrals.psd_dir = psd_dir;

    rsatb_inst->setup(model_config, setup_options);
    
    // 5. Verify Setup
    // Check if H(k) is created
    // K-points are distributed.
    // Rank 0 should have some K-points (if npar=1, all; if npar=2, half).
    
    // // 6. Compute DOS
    std::vector<double> dos, omega;
    rsatb_inst->compute_dos(500, dos, omega);
    
    if (rank == 0) {
        std::cout << "DOS computed. Size: " << dos.size() << std::endl;
        // 2 K-points, 100 steps -> 2 * 200 = 400
        // EXPECT_EQ(dos.size(), 2 * 200);
        double sum_dos = 0;
        for(auto d : dos) sum_dos += d;
        std::cout << "Sum DOS: " << sum_dos << std::endl;
        // EXPECT_GT(sum_dos, 0.0);

        // Output DOS to file for checking
        std::ofstream dos_file("dos.dat");
        if (dos_file.is_open()) {
            dos_file << "# Omega DOS" << std::endl;
            // Assuming omega and dos have same size or omega is shared
            // If omega is per-kpoint or global?
            // Let's assume 1-to-1 mapping for now or print what we have.
            size_t n = std::min(dos.size(), omega.size());
            for (size_t i = 0; i < n; ++i) {
                dos_file << omega[i] << " ";
                for (size_t j=0; j<nk; ++j) {
                    dos_file << dos[j*n+i] << " ";
                }
                dos_file << std::endl;
            }
            dos_file.close();
            std::cout << "DOS written to dos.dat" << std::endl;
        } else {
            std::cerr << "Unable to open dos.dat for writing" << std::endl;
        }
    }
    
    // 7. Compute AC
    std::vector<double> ac;
    rsatb_inst->compute_ac(500, 1000, 1000.0, 16.225300864, ac, omega);
    
    if (rank == 0) {
        std::cout << "AC computed. Size: " << ac.size() << std::endl;
        // 2 K-points, 101 steps, 6 components -> 2 * 101 * 6 = 1212
        std::ofstream ac_file("ac.dat");
        if (ac_file.is_open()) {
            ac_file << "# AC" << std::endl;
            // Assuming omega and dos have same size or omega is shared
            // If omega is per-kpoint or global?
            // Let's assume 1-to-1 mapping for now or print what we have.
            size_t n = ac.size();
            for (size_t i = 0; i < n; ++i) {
                ac_file << ac[i] << " ";
            }
            ac_file << std::endl;
            ac_file.close();
            std::cout << "DOS written to dos.dat" << std::endl;
        } else {
            std::cerr << "Unable to open dos.dat for writing" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
