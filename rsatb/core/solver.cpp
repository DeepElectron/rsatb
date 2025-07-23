#include "hr2hk.hpp"
#include "solver.hpp"
#include <slepc.h>
#include <iostream>
using namespace rsatb::backend;
using namespace rsatb::io;
using namespace rsatb::calc

namespace rsatb{

Solver::Solver(
    std::string &HFile, std::string &SFile, 
    int n_atom, int n_edge, 
    int *atom_type, int *edge_index, int *type_norb, int *edge_shift_vec,
    double *pos,
    MPI_Comm comm
    ): _HFile_(HFile), _SFile_(SFile), _mode_("H5ijR"), _n_atom_(n_atom),
       _n_edge_(n_edge), _atom_type_(atom_type), _edge_index_(edge_index),
       _type_norb_(type_norb), _edge_shift_vec_(edge_shift_vec), _pos_(pos), _comm_(comm),
       _Hhd_(_HFile_), _Shd_(_SFile_),
       _dm_(nullptr),
        _H_(nullptr), _S_(nullptr),
        _Samples_(nullptr), _Sample_(nullptr),
        _ksp_(nullptr), _pc_(nullptr)
    {};

Solver::~Solver() { /* destructor */
    if(_H_) MatDestroy(&_H_);
    if(_S_) MatDestroy(&_S_);
    if(_dm_) DMDestroy(&_dm_);
    if(_ksp_) KSPDestroy(&_ksp_);
    if(_n_sample_==1){
        if(_Sample_) VecDestroy(&_Sample_);
    }else{
        if(_Samples_) MatDestroy(&_Samples_);
    }
}

void Solver::setup_graph(){ /* Setup the DM, we assume here the graph information is already initialized in the class constructor method */
    rsatb::backend::get_dm_atomic(
        _n_atom_, _n_edge_, _edge_index_, _atom_type_, _type_norb_, _edge_shift_vec_, _pos_, &_dm_, _comm_
    );

    rsatb::backend::ViewDMNetworkPartition(_dm_, false);
};

void Solver::setup_HS(double Kx, double Ky, double Kz){ /* Setup H and S, how to assembled H/S is determined by the initialization mode*/
    DMCreateMatrix(_dm_, &_H_);
    setMatrixR2K(_H_, _dm_, &_Hhd_, Kx, Ky, Kz);

    DMCreateMatrix(_dm_, &_S_);
    setMatrixR2K(_S_, _dm_, &_Shd_, Kx, Ky, Kz);

    /* Use Slepc to get the spectrum amplitude, and scale+shift H*/
    EPS eps;
    PetscScalar emax, emin;
    PetscInt nconv=1;

    EPSCreate(_comm_, &eps);
    EPSSetOperators(eps, _H_, _S_);
    EPSSetProblemType(eps, EPS_GHEP);
    // EPSSetThreshold(eps,1e-4,PETSC_FALSE); // abs, rel
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
    // EPSSetType(eps,EPSKRYLOVSCHUR);
    EPSSetUp(eps);
    EPSSolve(eps);
    EPSGetConverged(eps, &nconv);
    EPSGetEigenpair(eps,0,&emax, NULL, NULL, NULL);

    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    EPSSolve(eps);
    EPSGetConverged(eps, &nconv);
    EPSGetEigenpair(eps,0,&emin, NULL, NULL, NULL);

    std::cout<<"Emax, Emin: "<< emax << " " << emin <<"\n";

    _scale_ = (emax - emin).real() / 2;
    _scale_ += 2; // allow for a bondary distance of 2 eV
    _shift_ = (emax + emin).real() / 2;

    // scale and shift the H/S
    MatAXPY(_H_, -_shift_, _S_, SAME_NONZERO_PATTERN);
    MatScale(_H_, _scale_);

    EPSDestroy(&eps);

    /* Setup the KSP solver also, need test*/
    KSPCreate(_comm_,&_ksp_);
    KSPSetDM(_ksp_, _dm_);
    KSPSetDMActive(_ksp_,PETSC_FALSE);
    KSPSetOperators(_ksp_,_S_,_S_);
    KSPGetPC(_ksp_,&_pc_);
    KSPSetReusePreconditioner(_ksp_, PETSC_TRUE);
    if(_n_sample_==1){
        PCSetType(_pc_,PCGAMG);
        PCGAMGSetType(_pc_, PCGAMGAGG);
        KSPSetType(_ksp_, KSPCG);
    }else{
        PCSetType(_pc_,PCGAMG);
        PCGAMGSetType(_pc_, PCGAMGAGG);
        KSPSetType(_ksp_, KSPHPDDM);
        KSPHPDDMSetType(_ksp_, KSP_HPDDM_TYPE_BCG);
    }
    PCSetUp(_pc_);
    KSPSetUp(_ksp_);
};

void Solver::setup_sample(int n, std::string method){
    if(n==1){
        DMCreateGlobalVector(_dm_, &_Sample_);
    }else{
        Mat temp;
        DMCreateMatrix(_dm_, &temp);
        MatConvert(temp,MATDENSE,MAT_INITIAL_MATRIX,&_Samples_);
        MatDestroy(&temp);
    }
    _n_sample_ = n;

    if(method=="gauss"){
        PetscRandom r;
        PetscRandomCreate(_comm_, &r);
        if(_n_sample_==1){
            VecSetRandom(_Sample_, r);
            VecNormalize(_Sample_, NULL);
        }else{
            PetscReal norms[n];
            MatSetRandom(_Samples_, r);
            Vec col;
            for (PetscInt j = 0; j < n; ++j) {
                MatDenseGetColumnVec(_Samples_,j,&col);
                VecNormalize(col,NULL);
                MatDenseRestoreColumnVec(_Samples_,j,&col);
            }
        }
    // }
    // else if (method=="hutch"){
    //     /* code */
    //     continue;
    // } else if (method=="hutch++"){
    //     continue;
    }else {
        throw std::invalid_argument("sampling method inappropriate, need to choose from gauss/hutch/hutch++");
    }
    
};

void Solver::compute_dos(int nt, std::string proj_mode, int *orbtype, double *dos, double *pdos){
    /*
        1. 
    */
    bool proj=true;
    int n_proj_type=1;
    IS isSrc=nullptr, isDst=nullptr;
    if(proj_mode=="element"){
        rsatb::backend::setup_is_atomic(_dm_, &isSrc, &isDst);
        n_proj_type = sizeof(_atom_type_) / sizeof(_atom_type_[0]);
    }else if(proj_mode=="orbital"){
        rsatb::backend::setup_is_orbital(_dm_, orbtype, _type_norb_, &isSrc, &isDst);
        n_proj_type = *std::max_element(orbtype, orbtype + sizeof(orbtype) / sizeof(orbtype[0]));
    }else{
        proj=false;
    }

    PetscComplex cdos[2*nt+1], pcdos[(2*nt+1)*n_proj_type];

    if(_n_sample_==1){
        rsatb::calc::corr_dos<Vec>(_H_, _ksp_, _Sample_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos);
    } else {
        rsatb::calc::corr_dos<Mat>(_H_, _ksp_, _Samples_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos);
    }

    // Processing cdos and pcdos to dos and pdos
};

};