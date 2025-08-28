#ifndef RSATB_SOLVER
#define RSATB_SOLVER

#include "hr2hk.hpp"
#include <slepc.h>
#include <memory>
using namespace rsatb::backend;
using namespace rsatb::io;

namespace rsatb{

class Solver
{
private:
    /* data */
    DM _dm_;
    Mat _H_, _S_, _Rx_, _Ry_, _Rz_, _Samples_;
    Vec _Sample_;
    KSP _ksp_;
    PC _pc_;
    
    PetscReal _scale_=-1.0, _shift_=0.0, _Kx_, _Ky_, _Kz_;
    MPI_Comm _comm_;
    bool _kpoint_ready_=false, _overlap_ready_=false, _hamiltonian_ready_=false, _position_ready_=false, _samples_ready_=false;

    const int *_atom_type_, *_edge_index_, *_type_norb_, *_edge_shift_vec_, _n_atom_, _n_edge_, _ntype_;
    int _n_sample_=-1;
    const double *_pos_, *_cell_;

    void set_mat_from_h5(Mat *mat, const std::string &MatFile){
        HDF5Handler _hd_(MatFile);
        DMCreateMatrix(_dm_, mat);
        setMatrixR2K(*mat, _dm_, &_hd_, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
    }

public:
    int rank;
    Solver(
        const int n_atom, const int n_edge, 
        const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
        const double *cell, const double *pos,
        MPI_Comm comm, std::string options=" "
        );

    ~Solver();

    void setup_graph(); /* Setup the DM, we assume here the graph information is already initialized in the class constructor method */

    void setup_K(double Kx, double Ky, double Kz){
        _Kx_ = static_cast<PetscReal>(Kx);
        _Ky_ = static_cast<PetscReal>(Ky);
        _Kz_ = static_cast<PetscReal>(Kz);

        _kpoint_ready_ = true;
    }

    void setup_S(const std::string &SFile) { 
        this->set_mat_from_h5(&_S_, SFile); 

        _overlap_ready_ = true;
    }

    void setup_S(const std::string* file_orb0, const std::string& orbital_dir){
        DMCreateMatrix(_dm_, &_S_);
        setMatrixR2K(_S_, _dm_, _cell_, _ntype_, file_orb0, orbital_dir, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
        // Mat ST;
        // MatTranspose(_S_, MAT_INITIAL_MATRIX, &ST);
        // MatAXPY(ST, -1., _S_, SAME_NONZERO_PATTERN);
        // PetscReal nST, nS;
        // MatNorm(ST, NORM_FROBENIUS, &nST);
        // MatNorm(_S_, NORM_FROBENIUS, &nS);
        // std::cout << "The norm of the overlap: " << nS << std::endl;
        // std::cout << "The norm of the overlap deriv: " << nST << std::endl;

        _overlap_ready_ = true;
    }

    void setup_R(const std::string &RxFile, const std::string &RyFile, const std::string &RzFile){
        if (RxFile.size() > 0) this->set_mat_from_h5(&_Rx_, RxFile);
        if (RyFile.size() > 0) this->set_mat_from_h5(&_Ry_, RyFile);
        if (RzFile.size() > 0) this->set_mat_from_h5(&_Rz_, RzFile);

        if ((RxFile.size() > 0) || (RyFile.size() > 0) || (RyFile.size() > 0))
            _position_ready_ = true;
    }

    void setup_R(const std::string* file_orb0, const std::string& orbital_dir){
        Mat s_check;
        DMCreateMatrix(_dm_, &s_check);
        DMCreateMatrix(_dm_, &_Rx_);
        DMCreateMatrix(_dm_, &_Ry_);
        DMCreateMatrix(_dm_, &_Rz_);
        setMatrixR2K(s_check, _dm_, _cell_, _ntype_, file_orb0, orbital_dir, _Kx_, _Ky_, _Kz_, _Rx_, _Ry_, _Rz_);

        if (_overlap_ready_){ // check if the s_check equals or close to the overlap prepared.
            MatAXPY(s_check, -1.0, _S_, SAME_NONZERO_PATTERN);
            PetscReal tol, nS, ns;

            MatNorm(_S_, NORM_FROBENIUS, &nS);
            MatNorm(s_check, NORM_FROBENIUS, &ns);

            tol = 1e-4 + 1e-3 * nS;
            PetscBool close = (ns <= tol) ? PETSC_TRUE : PETSC_FALSE;

            if (!close) {throw std::runtime_error("The Overlap is not accurate comparing to the 2 center integral's"); }
        }
        _position_ready_ = true;
    }

    void setup_H(const std::string &HFile, double emin=0.0, double emax=0.0);
    /* 
        Steps:
            1. create HS from DM, assembled them according to the class initialization method (h5ijR or atomicdata or others);
            2. Extract the spectrum range using Slepc, scale the H/S;
            3. Setup the KSP class from S.
    */
    /*
        This should be called after setup_HS;
    */

    void setup_sample(int n, std::string method);
    /*
        initialize the random samples according to the number of samples n and the method used.
    */

    void compute_dos(int nt, std::string proj_mode, const int *orbtype, int *n_proj, double *dos, std::unique_ptr<double[]>& pdos);
    /* dE defines the energe resolution, which should be scale/nt*/

    void compute_ac();

    void get_nt(double dE, int *nt) {
        *nt = ceil(_scale_ / dE);
    };

    /* Getters */
    int ntype() { return _ntype_; }
    Mat S() { return _S_; }
    Mat H() { return _H_; }
    Mat Rx() { return _Rx_; }
    Mat Ry() { return _Ry_; }
    Mat Rz() { return _Rz_; }

};
}

#endif