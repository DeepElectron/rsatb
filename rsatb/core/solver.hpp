#ifndef RSATB_SOLVER
#define RSATB_SOLVER

#include "calc/isqrtm.hpp"
#include "hr2hk.hpp"
#include <slepc.h>
#include <memory>

using namespace rsatb::backend;
using namespace rsatb::io;
using namespace rsatb::calc;

namespace rsatb{

class Solver
{
private:
    /* data */
    DM _dm_, _auxdm_;
    Mat _H_, _S_, _Sr_, _Rx_, _Ry_, _Rz_, _Samples_;
    Vec _Sample_;
    KSP _ksp_;
    PC _pc_;
    
    PetscReal _scale_=-1.0, _shift_=0.0, _Kx_, _Ky_, _Kz_;
    MPI_Comm _comm_;
    bool _kpoint_ready_=false, _overlap_ready_=false, _hamiltonian_ready_=false, _position_ready_=false, _samples_ready_=false;

    const int *_atom_type_, *_edge_index_, *_type_norb_, *_edge_shift_vec_, _n_atom_, _n_edge_, _ntype_;
    int _n_sample_=-1;
    const double *_pos_, *_cell_;
    double _cdn_;

    void set_mat_from_h5(Mat *mat, const std::string &MatFile){
        HDF5Handler _hd_(MatFile);
        DMCreateMatrix(_dm_, mat);
        MatConvert(*mat, "aijmkl", MAT_INPLACE_MATRIX, mat);
        setMatrixR2K(*mat, _dm_, &_hd_, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
        // MatSetType(*mat, MATMPIAIJMKL);
    }

    void check_density(Mat mat){
        
        PetscInt  M,N;
        MatInfo   mi;
        MatGetSize(mat,&M,&N);                         // global rows/cols
        MatGetInfo(mat, MAT_GLOBAL_SUM, &mi);          // after assembly, collective when MAT_GLOBAL_SUM
        PetscReal density = (M>0 && N>0)
                ? (PetscReal)mi.nz_used / ((PetscReal)M*(PetscReal)N)
                : 0.0;
        if (rank == 0){
            std::cout << "Mat Summary: nnz - " << mi.nz_used << " with total allocation - " << mi.nz_allocated << " density - " << density << std::endl;
        }
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

    void setup_auxgraph(int n_edge, const int *edge_index, const int *edge_shift_vec); /* Setup auxilliary DM, incase sometimes we need a enlarged matrix structure with a different rcut - therefore update edge_index and edge_shift_vec*/

    void setup_K(double Kx, double Ky, double Kz){
        _Kx_ = static_cast<PetscReal>(Kx);
        _Ky_ = static_cast<PetscReal>(Ky);
        _Kz_ = static_cast<PetscReal>(Kz);

        _kpoint_ready_ = true;
    }

    void setup_S(const std::string &SFile) { 
        this->set_mat_from_h5(&_S_, SFile); 
        // MatView(_S_, PETSC_VIEWER_STDOUT_WORLD);
        MatSetOption(_S_, MAT_HERMITIAN, PETSC_TRUE);
        MatSetOption(_S_, MAT_SPD, PETSC_TRUE);
        _overlap_ready_ = true;
    }

    void setup_S(const std::string* file_orb0, const std::string& orbital_dir){
        DMCreateMatrix(_dm_, &_S_);
        MatConvert(_S_, "aijmkl", MAT_INPLACE_MATRIX, &_S_);
        setMatrixR2K(_S_, _dm_, _cell_, _ntype_, file_orb0, orbital_dir, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
        MatSetOption(_S_, MAT_HERMITIAN, PETSC_TRUE);
        MatSetOption(_S_, MAT_SPD, PETSC_TRUE);

        // Mat ST;
        // MatTranspose(_S_, MAT_INITIAL_MATRIX, &ST);
        // MatView(ST, PETSC_VIEWER_STDOUT_WORLD);
        // MatAXPY(ST, -1., _S_, SAME_NONZERO_PATTERN);
        // PetscReal nST, nS;
        // MatNorm(ST, NORM_FROBENIUS, &nST);
        // MatNorm(_S_, NORM_FROBENIUS, &nS);
        // std::cout << "The norm of the overlap: " << nS << std::endl;
        // std::cout << "The norm of the overlap deriv: " << nST << std::endl;

        // check the smallest eigenvalue of S
        // EPS eps;
        // PetscInt nconv=1;
        // PetscScalar eigmin, eigmax;
        // ST st; KSP ksp; PC pc;
        // Mat SH;

        // // MatShift(_S_, 100.0);

        // EPSCreate(_comm_, &eps);
        // EPSSetOperators(eps, _S_, NULL);
        // EPSSetProblemType(eps, EPS_HEP);
        // EPSSetTolerances(eps,1e-6,50);
        // // EPSSetDimensions(eps,1,PETSC_DEFAULT,PETSC_DEFAULT);
        // EPSSetDimensions(eps,1,100,100);
        // EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);

        // EPSSolve(eps);
        // EPSGetConverged(eps, &nconv);
        // EPSGetEigenpair(eps,0,&eigmax, NULL, NULL, NULL);

        // EPSGetST(eps, &st);
        // STGetKSP(st,&ksp);
        // KSPSetType(ksp,KSPCG);
        // KSPSetDM(ksp, _dm_);
        // KSPSetDMActive(ksp,PETSC_FALSE);
        // KSPGetPC(ksp,&pc);
        // KSPSetReusePreconditioner(ksp, PETSC_TRUE);
        // KSPSetFromOptions(ksp);

        // // PCSetType(pc, PCASM);
        // // PCASMSetOverlap(pc, 0);
        // // PCASMSetTotalSubdomains(pc, 8, NULL, NULL);
        // // PetscOptionsInsertString(NULL,"-sub_pc_type cholesky ");
        // // PCASMSetSubMatType(pc, MATSEQAIJMKL);
        // // PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
        // // PCFactorSetShiftAmount(pc, 0.1);

        // PCSetType(pc,PCGAMG);
        // PCGAMGSetType(pc, PCGAMGAGG);
        // PCGAMGSetGraphSymmetrize(pc, PETSC_TRUE);
        // // PCGAMGSetSquareGraph(pc, 1);
        // // PCGAMGSetAggressiveLevels(pc,0);
        // // PetscReal thr = 0.15;                 /* try 0.12 first; 0.08–0.15 is a good band */
        // // PCGAMGSetThreshold(pc, &thr, 1);
        // // PCGAMGSetLowMemoryFilter(pc, PETSC_TRUE);
        // // PCGAMGSetCoarseEqLim(pc, 2200);
        // // PCGAMGSetNSmooths(pc, 1);


        // STSetType(st, STSINVERT);
        // EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL);
        // EPSSetTarget(eps, 0.0);

        // EPSSolve(eps);
        // EPSGetConverged(eps, &nconv);
        // EPSGetEigenpair(eps,0,&eigmin, NULL, NULL, NULL);

        // EPSDestroy(&eps);

        // _cdn_ = eigmax.real() / eigmin.real();

        // std::cout << "The conditional number: " << eigmax.real() << "/" << eigmin.real() << " = " << _cdn_ << std::endl;
        // if (eigmin.real() < 0 || std::abs(eigmin.imag()) > 0) {
        //     throw std::runtime_error("The overlap matrix is not positive definite, wrong input!");
        // }

        // MatShift(_S_, -100.0);

        // MatScale(_S_, 0.5);
        // MatCreateHermitianTranspose(_S_, &SH);
        // MatAXPY(_S_, 1.0, SH, SAME_NONZERO_PATTERN);
        _overlap_ready_ = true;
    }

    void test_Sinv(){
        if ( !_samples_ready_ ) throw std::runtime_error("Please setup random states (samples) before testing the S inverse.");
        if ( !_kpoint_ready_ )  throw std::runtime_error("Please setup kpoint before testing the S inverse.");
        if ( !_overlap_ready_ ) throw std::runtime_error("Please setup overlap before testing the S inverse.");

        // setup KSP
        KSPCreate(_comm_,&_ksp_);
        KSPSetDM(_ksp_, _dm_);
        KSPSetDMActive(_ksp_,PETSC_FALSE);
        KSPSetOperators(_ksp_,_S_,_S_);
        KSPGetPC(_ksp_,&_pc_);
        KSPSetReusePreconditioner(_ksp_, PETSC_TRUE);
        KSPSetTolerances(_ksp_, 1e-8, 1e-9, PETSC_CURRENT, PETSC_CURRENT);

        if(_n_sample_== 1){
            KSPSetType(_ksp_, KSPCG);
            KSPCGSetType(_ksp_, KSP_CG_HERMITIAN); // or -ksp_cg_type hermitian via options
        }else{
            KSPSetType(_ksp_, KSPHPDDM);
            KSPHPDDMSetType(_ksp_, KSP_HPDDM_TYPE_BCG);
        }

        //setup pc
        // PCSetType(_pc_, PCASM);
        // PCASMSetOverlap(_pc_, 0);
        // // PCASMSetTotalSubdomains(pc, 8, NULL, NULL);
        // PetscOptionsInsertString(NULL,"-sub_pc_type icc -pc_factor_shift_positive_definite -sub_ksp_type preonly -sub_pc_factor_shift_type positive_definite -sub_pc_factor_levels 1");
        // PCASMSetSubMatType(_pc_, MATSEQAIJMKL);

        // PCSetType(_pc_,PCGAMG);
        // PCGAMGSetType(_pc_, PCGAMGAGG);
        // PCGAMGSetGraphSymmetrize(_pc_, PETSC_TRUE);
        // PCGAMGSetAggressiveSquareGraph(_pc_, PETSC_FALSE);
        // // PCGAMGSetAggressiveLevels(_pc_, 0);
        // PCGAMGMISkSetAggressive(_pc_, 1);
        // PetscOptionsInsertString(NULL, "-pc_mg_type multiplicative -pc_gamg_agg_nsmooths 1 -pc_mg_nsmooths 1 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_ksp_type preonly -mg_coarse_pc_type cholesky");
        // PetscReal thres[1]={0.02};
        // PCGAMGSetThreshold(_pc_, thres, 1);
        // // PCGAMGSetAggressiveLevels(_pc_,2);
        // PCGAMGSetCoarseEqLim(_pc_, 50);
        // PCGAMGSetReuseInterpolation(_pc_, PETSC_TRUE);

        // PC pc0, pc1;
        Mat Sinv;

        MatCreateNormalHermitian(_Sr_, &Sinv);
        // MatHermitianTranspose(_Sr_, MAT_INITIAL_MATRIX, &SrH);
        PCSetType(_pc_, PCMAT);
        // MatCompositeSetType(Sinv, MAT_COMPOSITE_MULTIPLICATIVE);
        // MatCompositeAddMat(Sinv, _Sr_);
        // MatCompositeAddMat(Sinv, SrH);
        PCSetOperators(_pc_, _S_, Sinv);
        // KSPGetPC(_ksp_,&_pc_); 
        // PCSetType(_pc_, PCNONE); // or your preconditioner
        KSPSetFromOptions(_ksp_);
        KSPSetUp(_ksp_);

        
        if(_n_sample_==1){
            Vec test_vec;
            VecDuplicate(_Sample_, &test_vec);
            KSPSolve(_ksp_, _Sample_, test_vec);
            VecDestroy(&test_vec);
        }else{
            Mat test_vec;
            MatDuplicate(_Samples_, MAT_COPY_VALUES, &test_vec);
            KSPMatSolve(_ksp_, _Samples_, test_vec);
            MatDestroy(&test_vec);
        }
    }

    void Ssqrtm(double tol=1e-7, double sptol=1e-6, int maxit = 20, double factor=0.31){
        DMCreateMatrix(_dm_, &_Sr_);
        MatAssemblyBegin(_Sr_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(_Sr_, MAT_FINAL_ASSEMBLY);
        rsatb::calc::isqrtm(_S_, _dm_, tol, sptol, maxit, factor, &_Sr_);
        MatConvert(_Sr_, "aijmkl", MAT_INPLACE_MATRIX, &_Sr_);

        // check whether _Sr_ works well
        // Mat tmp, tmp2;
        // PetscReal norm;
        // MatMatMult(_Sr_, _Sr_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp);
        // MatMatMult(tmp, _S_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp2);
        // MatShift(tmp2, -1.0);
        // MatNorm(tmp2, NORM_INFINITY, &norm);
        // std::cout << "ZSZ-I norm: " << norm << std::endl;


        // check
        // PetscReal error;
        // Mat tmp1=nullptr, tmp2=nullptr;
        // MatMatMult(_Sr_, _S_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp1);
        // MatMatMult(tmp1, _Sr_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp2);
        // MatShift(tmp2, -1.0);
        // MatNorm(tmp2, NORM_INFINITY, &error);
        // MatShift(tmp2, 1.0);

        // std::cout << "output ZSZ-I 's norm " << error << std::endl;
        // Mat Sc=nullptr;
        // dbcsr_matrix tmp=nullptr;
        // DMat2DBCSR(_S_, _dm_, &tmp, 'N');
        // DBCSR2Mat(_dm_, tmp, &Sc);

        // PetscViewer viewer;
        // PetscViewerASCIIOpen(PETSC_COMM_WORLD, "/nessa/users/zhanghao/dev/rsatb/test/data/diamond/Sr.mtx", &viewer);
        // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        // MatView(_Sr_, viewer);
        // PetscViewerDestroy(&viewer);
    }

    void Ssqrtm_sparse(double tol=1e-7, double sptol=1e-6, int maxit = 20, double factor=0.31){
        PetscReal error = 999.0;
        // PetscReal thres = filter;
        int nnorb = this->norb();
        int count = 0;
        Mat ZT=NULL, tmp1=NULL, tmp2=NULL; // D=ZSZ, Z=_Sr_
        
        if (!_Sr_) DMCreateMatrix(_dm_, &_Sr_);
        MatAssemblyBegin(_Sr_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(_Sr_, MAT_FINAL_ASSEMBLY);
        MatSetOption(_Sr_, MAT_HERMITIAN, PETSC_TRUE);
        MatConvert(_Sr_, "aijkokkos", MAT_INPLACE_MATRIX, &_Sr_);
        MatConvert(_S_, "aijkokkos", MAT_INPLACE_MATRIX, &_S_); // Convert S temperarily
        // init _Sr_ & ZT
        MatZeroEntries(_Sr_);
        MatShift(_Sr_, factor);

        this->check_density(_S_);

        while (error > tol && count < maxit)
        {
            /* code */
            // PetscInt ars, are, brs, bre;
            // MatGetOwnershipRange(_Sr_, &ars, &are);
            // MatGetOwnershipRange(ZT, &brs, &bre);
            // std::cout << ars << " " << brs << " " << are << " " << bre << std::endl; 
            // assert(ars==brs && are==bre);
            
            // MatCopy(_Sr_, ZT, DIFFERENT_NONZERO_PATTERN);
            // MatHermitianTranspose(ZT, MAT_INPLACE_MATRIX, &ZT);
            if (ZT) MatDestroy(&ZT);
            MatHermitianTranspose(_Sr_, MAT_INITIAL_MATRIX, &ZT);
            if (tmp1) MatDestroy(&tmp1);
            MatMatMult(ZT, _S_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp1);
            MatConvert(tmp1, MATAIJ, MAT_INPLACE_MATRIX, &tmp1);
            MatFilter(tmp1, sptol, PETSC_TRUE, PETSC_TRUE);
            MatConvert(tmp1, "aijkokkos", MAT_INPLACE_MATRIX, &tmp1);
            if (tmp2) MatDestroy(&tmp2);
            MatMatMult(tmp1, _Sr_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp2);
            MatConvert(tmp2, MATAIJ, MAT_INPLACE_MATRIX, &tmp2);
            MatFilter(tmp2, sptol, PETSC_TRUE, PETSC_TRUE);
            MatConvert(tmp2, "aijkokkos", MAT_INPLACE_MATRIX, &tmp2);
            

            // check error with nZ
            MatShift(tmp2, -1.0);
            MatNorm(tmp2, NORM_INFINITY, &error);
            MatShift(tmp2, 1.0);

            if (rank == 0){
                std::cout << "Recursive Factorization of S: current error - " << error  << std::endl;
            }

            // compute DD
            if (tmp1) MatDestroy(&tmp1);
            MatMatMult(tmp2, tmp2, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp1);
            MatConvert(tmp1, MATAIJ, MAT_INPLACE_MATRIX, &tmp1);
            MatFilter(tmp1, sptol, PETSC_TRUE, PETSC_TRUE);
            MatConvert(tmp1, "aijkokkos", MAT_INPLACE_MATRIX, &tmp1);

            // compute DD = 0.375 * DD - 1.25 * ZSZ + 1.875 * I
            MatScale(tmp1, 0.375);
            MatAXPY(tmp1, -1.25, tmp2, DIFFERENT_NONZERO_PATTERN);
            MatShift(tmp1, 1.875);
            MatConvert(tmp1, MATAIJ, MAT_INPLACE_MATRIX, &tmp1);
            MatFilter(tmp1, sptol, PETSC_TRUE, PETSC_TRUE);
            MatConvert(tmp1, "aijkokkos", MAT_INPLACE_MATRIX, &tmp1);
            
            // compute _Sr_ = _Sr_ @ DD
            if (tmp2) MatDestroy(&tmp2);
            MatMatMult(_Sr_, tmp1, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp2);
            MatConvert(tmp2, MATAIJ, MAT_INPLACE_MATRIX, &tmp2);
            MatFilter(tmp2, sptol, PETSC_TRUE, PETSC_TRUE);
            MatConvert(tmp2, "aijkokkos", MAT_INPLACE_MATRIX, &tmp2);
            MatCopy(tmp2, _Sr_, DIFFERENT_NONZERO_PATTERN);

            this->check_density(_Sr_);
            
            count ++;
        }

        MatDestroy(&ZT); MatDestroy(&tmp1); MatDestroy(&tmp2);
        MatConvert(_S_, "aijmkl", MAT_INPLACE_MATRIX, &_S_);
    }

    void setup_R(const std::string &RxFile, const std::string &RyFile, const std::string &RzFile){
        if (RxFile.size() > 0) this->set_mat_from_h5(&_Rx_, RxFile);
        if (RyFile.size() > 0) this->set_mat_from_h5(&_Ry_, RyFile);
        if (RzFile.size() > 0) this->set_mat_from_h5(&_Rz_, RzFile);

        // MatSetType(_Rx_, MATMPIAIJMKL);
        // MatSetType(_Ry_, MATMPIAIJMKL);
        // MatSetType(_Rz_, MATMPIAIJMKL);

        if ((RxFile.size() > 0) || (RyFile.size() > 0) || (RyFile.size() > 0))
            _position_ready_ = true;
    }

    void setup_R(const std::string* file_orb0, const std::string& orbital_dir){
        Mat s_check;
        DMCreateMatrix(_dm_, &s_check);
        DMCreateMatrix(_dm_, &_Rx_);
        DMCreateMatrix(_dm_, &_Ry_);
        DMCreateMatrix(_dm_, &_Rz_);

        MatConvert(s_check, "aijmkl", MAT_INPLACE_MATRIX, &s_check);
        MatConvert(_Rx_, "aijmkl", MAT_INPLACE_MATRIX, &_Rx_);
        MatConvert(_Ry_, "aijmkl", MAT_INPLACE_MATRIX, &_Ry_);
        MatConvert(_Rz_, "aijmkl", MAT_INPLACE_MATRIX, &_Rz_);

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

    void export_S(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/S.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_S_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void compute_dos(int nt, std::string proj_mode, const int *orbtype, int *n_proj, double *dos, std::unique_ptr<double[]>& pdos);
    /* dE defines the energe resolution, which should be scale/nt*/

    void compute_ac(int nt, int n_kpm, double temp, double mu, double *ac, double *omegas);

    // void compute_fermi_dirac(int n_kpm, double temp, double mu, double *occ);

    void get_nt(double dE, int *nt) {
        *nt = ceil(_scale_ / dE);
    };

    /* Getters */
    int ntype() { return _ntype_; }
    int norb() {
        int matdim = 0;
        for(int i=0; i<_n_atom_; ++i) matdim += _type_norb_[_atom_type_[i]];
        return matdim;
    }

    double volume() {
        return _cell_[0]*(_cell_[4]*_cell_[8] - _cell_[5]*_cell_[7]) \
             - _cell_[1]*(_cell_[3]*_cell_[8] - _cell_[5]*_cell_[6]) \
             + _cell_[2]*(_cell_[3]*_cell_[7] - _cell_[4]*_cell_[6]);
    }

    DM dm() { return _dm_; }
    
    Mat S() { return _S_; }
    Mat H() { return _H_; }
    Mat Rx() { return _Rx_; }
    Mat Ry() { return _Ry_; }
    Mat Rz() { return _Rz_; }

};
}

#endif