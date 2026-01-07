#include <iostream>
// #include <int2c/source_cell/atom_spec.h>
#include <int2c/source_estate/read_pseudo.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "int2c installation verification successful!" << std::endl;
        Atom_pseudo atom;
        atom.psd = "C";
        std::cout << "Atom label: " << atom.psd<< std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
