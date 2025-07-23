#ifndef RSATB_SOLVER
#define RSATB_SOLVER

#include "hr2hk.hpp"
#include <slepc.h>
using namespace rsatb::backend;
using namespace rsatb::io;

namespace rsatb{

class Solver
{
private:
    /* data */
    std::string _mode_, _HFile_, _SFile_;
    DM _dm_;
    HDF5Handler _Hhd_, _Shd_;
    Mat _H_, _S_, _Samples_;
    Vec _Sample_;
    KSP _ksp_;
    PC _pc_;
    PetscReal _scale_=1.0, _shift_=0.0;
    MPI_Comm _comm_;
    int *_atom_type_, *_edge_index_, *_type_norb_, *_edge_shift_vec_, _n_atom_, _n_edge_, _n_sample_;
    double *_pos_;
public:
    Solver(
        std::string &HFile, std::string &SFile, 
        int n_atom, int n_edge, 
        int *atom_type, int *edge_index, int *type_norb, int *edge_shift_vec,
        double *pos,
        MPI_Comm comm
        );

    ~Solver();

    void setup_graph(); /* Setup the DM, we assume here the graph information is already initialized in the class constructor method */

    void setup_HS(double Kx, double Ky, double Kz);
    /* 
        Steps:
            1. create HS from DM, assembled them according to the class initialization method (h5ijR or atomicdata or others);
            2. Extract the spectrum range using Slepc, scale the H/S;
            3. Setup the KSP class from S.
    */

    void setup_sample(int n, std::string method);
    /*
        initialize the random samples according to the number of samples n and the method used.
    */

    void compute_dos(int nt, std::string proj_mode, double *dos, double *pdos);
    /* dE defines the energe resolution, which should be scale/nt*/

    void get_nt(double dE, int *nt) {
        *nt = ceil(_scale_ / dE);
    };
};
}

#endif