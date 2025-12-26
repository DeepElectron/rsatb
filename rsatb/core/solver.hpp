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
    std::unique_ptr<DMAtomic> _dm_atomic_;
    Mat _H_, _S_, _Sr_, _Rx_, _Ry_, _Rz_, _Samples_;
    Vec _Sample_;
    KSP _ksp_=nullptr;
    PC _pc_;
    
    PetscReal _scale_=-1.0, _shift_=0.0, _Kx_, _Ky_, _Kz_;
    MPI_Comm _comm_;
    bool _kpoint_ready_=false, _overlap_ready_=false, _hamiltonian_ready_=false, _position_ready_=false, _samples_ready_=false, _sqrtm_ready_=false;
    bool _explicit_orth_ = false, _explicit_form_ = false;
    int _n_sample_=-1;
    double _cdn_;

    void set_mat_from_h5(Mat *mat, const std::string &MatFile){
        HDF5Handler _hd_(MatFile);
        DMCreateMatrix(_dm_atomic_->dm(), mat);
        MatConvert(*mat, "aijmkl", MAT_INPLACE_MATRIX, mat);
        setMatrixR2K(*mat, *_dm_atomic_, &_hd_, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
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
        const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
        const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
        const double *cell, const double *pos,
        MPI_Comm comm, std::string options=" "
        );

    ~Solver();

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
        DMCreateMatrix(_dm_atomic_->dm(), &_S_);
        MatConvert(_S_, "aijmkl", MAT_INPLACE_MATRIX, &_S_);
        setMatrixR2K(_S_, *_dm_atomic_, _dm_atomic_->cell(), _dm_atomic_->n_type(), file_orb0, orbital_dir, 
            static_cast<double>(_Kx_), static_cast<double>(_Ky_), static_cast<double>(_Kz_));
        MatSetOption(_S_, MAT_HERMITIAN, PETSC_TRUE);
        MatSetOption(_S_, MAT_SPD, PETSC_TRUE);

        _overlap_ready_ = true;
    }

    void test_Sinv(){
        if ( !_samples_ready_ ) throw std::runtime_error("Please setup random states (samples) before testing the S inverse.");
        if ( !_kpoint_ready_ )  throw std::runtime_error("Please setup kpoint before testing the S inverse.");
        if ( !_overlap_ready_ ) throw std::runtime_error("Please setup overlap before testing the S inverse.");
        if ( !_sqrtm_ready_ ) throw std::runtime_error("Please setup S^1/2 before configuring the Hamiltonian.");

        // setup KSP
        KSPCreate(_comm_,&_ksp_);
        KSPSetDM(_ksp_, _dm_atomic_->dm());
        KSPSetDMActive(_ksp_,PETSC_FALSE);
        KSPSetOperators(_ksp_,_S_,_S_);
        KSPGetPC(_ksp_,&_pc_);
        KSPSetReusePreconditioner(_ksp_, PETSC_TRUE);
        KSPSetTolerances(_ksp_, 1e-8, 1e-9, PETSC_DEFAULT, PETSC_DEFAULT);

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
        // PCASMSetSubMatType(_pc_, MATSEQaij);

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
        PCSetType(_pc_, PCMAT);
        PCSetOperators(_pc_, _S_, Sinv);
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

        KSPDestroy(&_ksp_);
    }

    void Ssqrtm(double tol=1e-7, double sptol=1e-6, int maxit = 20, bool verbose=false, bool orthogonalize=false){
        DMCreateMatrix(_dm_atomic_->dm(), &_Sr_);
        MatAssemblyBegin(_Sr_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(_Sr_, MAT_FINAL_ASSEMBLY);

        // get the largest eigenvalue of S
        EPS eps;
        PetscInt nconv=1;
        PetscScalar eigmax;

        EPSCreate(_comm_, &eps);
        EPSSetOperators(eps, _S_, NULL);
        EPSSetProblemType(eps, EPS_HEP);
        EPSSetTolerances(eps,1e-3,50);
        // EPSSetDimensions(eps,1,PETSC_DEFAULT,PETSC_DEFAULT);
        EPSSetDimensions(eps,1,100,100);
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);

        EPSSolve(eps);
        EPSGetConverged(eps, &nconv);
        EPSGetEigenpair(eps,0,&eigmax, NULL, NULL, NULL);

        EPSDestroy(&eps);

        if (_dm_atomic_->rank()==0 && verbose){
            std::cout << "The largest eigenvalue of overlap S: " << eigmax.real() << std::endl;
        }

        double factor = 1.0 / std::sqrt(eigmax.real() + 0.1);

        if (orthogonalize) {
             if (!_hamiltonian_ready_) throw std::runtime_error("Hamiltonian must be ready before orthogonalization in Ssqrtm.");
             rsatb::calc::isqrtm(_S_, *_dm_atomic_, tol, sptol, maxit, factor, &_Sr_, verbose, &_H_, &_H_);
             _explicit_orth_ = true;
        } else {
             rsatb::calc::isqrtm(_S_, *_dm_atomic_, tol, sptol, maxit, factor, &_Sr_, verbose);
        }
        
        MatConvert(_Sr_, "aijmkl", MAT_INPLACE_MATRIX, &_Sr_);

        // check whether _Sr_ works well
        // Mat tmp, tmp2;
        // PetscReal norm;
        // MatMatMult(_Sr_, _Sr_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp);
        // MatMatMult(tmp, _S_, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp2);
        // MatShift(tmp2, -1.0);
        // MatNorm(tmp2, NORM_INFINITY, &norm);
        // if (_dm_atomic_->rank()==0 && verbose){
        //     std::cout << "ZSZ-I norm: " << norm << std::endl;
        // }

        // MatDestroy(&tmp); MatDestroy(&tmp2);


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

        _sqrtm_ready_ = true;
    }

    void setup_R(const std::string &RxFile, const std::string &RyFile, const std::string &RzFile){
        if (RxFile.size() > 0) this->set_mat_from_h5(&_Rx_, RxFile);
        if (RyFile.size() > 0) this->set_mat_from_h5(&_Ry_, RyFile);
        if (RzFile.size() > 0) this->set_mat_from_h5(&_Rz_, RzFile);

        MatSetType(_Rx_, "aijmkl");
        MatSetType(_Ry_, "aijmkl");
        MatSetType(_Rz_, "aijmkl");

        if ((RxFile.size() > 0) || (RyFile.size() > 0) || (RyFile.size() > 0))
            _position_ready_ = true;
    }

    void setup_R(const std::string* file_orb0, const std::string& orbital_dir){
        DMCreateMatrix(_dm_atomic_->dm(), &_Rx_);
        DMCreateMatrix(_dm_atomic_->dm(), &_Ry_);
        DMCreateMatrix(_dm_atomic_->dm(), &_Rz_);

        MatConvert(_Rx_, "aijmkl", MAT_INPLACE_MATRIX, &_Rx_);
        MatConvert(_Ry_, "aijmkl", MAT_INPLACE_MATRIX, &_Ry_);
        MatConvert(_Rz_, "aijmkl", MAT_INPLACE_MATRIX, &_Rz_);

        if (!_overlap_ready_) {
            DMCreateMatrix(_dm_atomic_->dm(), &_S_);
            MatConvert(_S_, "aijmkl", MAT_INPLACE_MATRIX, &_S_);
            
            setMatrixR2K(_S_, *_dm_atomic_, _dm_atomic_->cell(),  _dm_atomic_->n_type(), file_orb0, orbital_dir, _Kx_, _Ky_, _Kz_, _Rx_, _Ry_, _Rz_);
            
            MatSetOption(_S_, MAT_HERMITIAN, PETSC_TRUE);
            MatSetOption(_S_, MAT_SPD, PETSC_TRUE);
            _overlap_ready_ = true;
        } else {
            Mat s_check;
            DMCreateMatrix(_dm_atomic_->dm(), &s_check);
            MatConvert(s_check, "aijmkl", MAT_INPLACE_MATRIX, &s_check);
            
            setMatrixR2K(s_check, *_dm_atomic_, _dm_atomic_->cell(),  _dm_atomic_->n_type(), file_orb0, orbital_dir, _Kx_, _Ky_, _Kz_, _Rx_, _Ry_, _Rz_);

            // check if the s_check equals or close to the overlap prepared.
            MatAXPY(s_check, -1.0, _S_, SAME_NONZERO_PATTERN);
            PetscReal tol, nS, ns;

            MatNorm(_S_, NORM_FROBENIUS, &nS);
            MatNorm(s_check, NORM_FROBENIUS, &ns);

            tol = 1e-4 + 1e-3 * nS;
            PetscBool close = (ns <= tol) ? PETSC_TRUE : PETSC_FALSE;
            
            MatDestroy(&s_check);

            if (!close) {throw std::runtime_error("The Overlap is not accurate comparing to the 2 center integral's"); }
        }
        _position_ready_ = true;
    }

    void setup_H_h5(const std::string &HFile);
    /* 
        Steps:
            1. create HS from DM, assembled them according to the class initialization method (h5ijR or atomicdata or others);
            2. Extract the spectrum range using Slepc, scale the H/S;
            3. Setup the KSP class from S.
    */
    /*
        This should be called after setup_HS;
    */

    void orthogonalize_H(double tol, double sptol, int maxit, bool explicit_formation=true, bool verbose=false);

    void scale_shift_H(double emin=0.0, double emax=0.0, bool verbose=false);

    void setup_sample(int n, std::string method);
    /*
        initialize the random samples according to the number of samples n and the method used.
    */

    void setup_KSP();
    /*
        Setup the KSP class from S.
    */

    void export_S(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/S.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_S_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void export_Rx(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/Rx.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_Rx_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void export_Ry(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/Ry.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_Ry_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void export_Rz(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/Rz.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_Rz_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void export_H(std::string root){
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, (root + "/H.mtx").c_str(), &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
        MatView(_H_, viewer);
        PetscViewerDestroy(&viewer);
    }

    void compute_dos(int nt, std::string proj_mode, const int *orbtype, int *n_proj, double *dos, std::unique_ptr<double[]>& pdos, double *omegas, bool verbose=false);
    /* dE defines the energe resolution, which should be scale/nt*/

    void compute_ac(int nt, int n_kpm, double temp, double mu, double *ac, double *omegas);

    // void compute_fermi_dirac(int n_kpm, double temp, double mu, double *occ);

    void get_nt(double dE, int *nt) {
        *nt = ceil(_scale_ / dE);
    };

    double get_scale() { return _scale_; }

    /* Getters */
    int ntype() { return  _dm_atomic_->n_type(); }
    int norb() {
        int matdim = 0;
        for(int i=0; i<_dm_atomic_->n_atom(); ++i) matdim += _dm_atomic_->type_norb()[_dm_atomic_->atom_type()[i]];

        MPI_Allreduce(MPI_IN_PLACE, &matdim, 1, MPI_INT, MPIU_SUM, _comm_);
        return matdim;
    }

    double volume() {
        const double *cl = _dm_atomic_->cell();
        return cl[0]*(cl[4]*cl[8] - cl[5]*cl[7]) \
             - cl[1]*(cl[3]*cl[8] - cl[5]*cl[6]) \
             + cl[2]*(cl[3]*cl[7] - cl[4]*cl[6]);
    }

    DMAtomic& dm_atomic() { return *_dm_atomic_; }
    
    Mat S() { return _S_; }
    Mat H() { return _H_; }
    Mat Rx() { return _Rx_; }
    Mat Ry() { return _Ry_; }
    Mat Rz() { return _Rz_; }

};
}

#endif