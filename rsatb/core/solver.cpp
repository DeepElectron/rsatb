#include "hr2hk.hpp"
#include "calc/dos.hpp"
#include "calc/ac.hpp"
#include "calc/fft.hpp"
#include "calc/sample.hpp"
#include "solver.hpp"
#include <math.h>
#include <iostream>
#include <algorithm>
using namespace rsatb::backend;
using namespace rsatb::io;
using namespace rsatb::calc;

/*
1. call setup_S
2. call setup_H
3. if explicit_orth, call orthogonalize_H
        if explicit_form, call setup_sqrtm with H
        if not, call setup_sqrtm without H, form H = S^{-1/2}HS^{-1/2} as MatShell
   if not, call setup_sqrtm, call setup_ksp
4. call scaleshift_H
5. call setup_samples

*/

typedef struct {
  Mat H, Sr;
  Vec w1;  // work vectors of size m
} MContext;

PetscErrorCode SrHSrMult(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;
  MContext      *ctx;

  ierr = MatShellGetContext(M, (void**)&ctx);CHKERRQ(ierr);

  // w1 = A * x
  ierr = MatMult(ctx->Sr, x, y);CHKERRQ(ierr);

  // w2 = B * w1
  ierr = MatMult(ctx->H, y, ctx->w1);CHKERRQ(ierr);

  // y = A^H * w2 (Hermitian transpose; for real case you can use MatMultTranspose)
  ierr = MatMultHermitianTranspose(ctx->Sr, ctx->w1, y);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode SrHSrDestroy(Mat M)
{
    PetscErrorCode ierr;
    MContext      *ctx;

    ierr = MatShellGetContext(M, (void**)&ctx);CHKERRQ(ierr);

    // Destroy the work vector
    if (ctx->w1) { ierr = VecDestroy(&ctx->w1);CHKERRQ(ierr); }

    // Release references to the underlying matrices
    if (ctx->H)  { ierr = MatDestroy(&ctx->H);CHKERRQ(ierr); }
    if (ctx->Sr) { ierr = MatDestroy(&ctx->Sr);CHKERRQ(ierr); }

    // Free the context structure
    ierr = PetscFree(ctx);CHKERRQ(ierr);

    return 0;
}


namespace rsatb{

Solver::Solver(
    const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge, 
    const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
    const double *cell, const double *pos,
    MPI_Comm comm,
    std::string options
    ): 
        _comm_(comm),
        _H_(nullptr), _S_(nullptr), _Rx_(nullptr), _Ry_(nullptr), _Rz_(nullptr), _Sr_(nullptr),
        _Samples_(nullptr), _Sample_(nullptr),
        _ksp_(nullptr), _pc_(nullptr)
    {
        // PetscOptionsSetValue(NULL, "-info", "stacking.log:ksp,pc");
        // PetscOptionsSetValue(NULL, "-log_view", ":profiling.log");
        PetscOptionsInsertString(NULL, options.c_str());
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
        c_dbcsr_init_lib(_comm_, nullptr);
        MPI_Comm_rank(_comm_, &rank);
        
        _dm_atomic_ = std::make_unique<DMAtomic>(
            n_atom, N_atom, atom_offset, n_edge, N_edge,
            atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
            cell, pos,
            comm
        );

        DMSetMatType(_dm_atomic_->dm(), "aij");
    };

Solver::~Solver() { /* destructor */
    if(_H_) MatDestroy(&_H_);
    if(_S_) MatDestroy(&_S_);
    if(_Sr_) MatDestroy(&_Sr_);
    if(_Rx_) MatDestroy(&_Rx_);
    if(_Ry_) MatDestroy(&_Ry_);
    if(_Rz_) MatDestroy(&_Rz_);
    if(_ksp_) KSPDestroy(&_ksp_);
    if(_n_sample_==1){
        if(_Sample_) VecDestroy(&_Sample_);
    }else if (_n_sample_ > 1) {
        if(_Samples_) MatDestroy(&_Samples_);
    }
    c_dbcsr_finalize_lib();
    SlepcFinalize();
    PetscFinalize();
}

void Solver::setup_H_h5(const std::string &HFile){ /* Setup H and S, how to assembled H/S is determined by the initialization mode*/
    if ( !_samples_ready_ ) throw std::runtime_error("Please setup random states (samples) before configuring the Hamiltonian.");
    if ( !_kpoint_ready_ )  throw std::runtime_error("Please setup kpoint before configuring the Hamiltonian.");
    if ( !_overlap_ready_ ) throw std::runtime_error("Please setup overlap before configuring the Hamiltonian.");
    
    // std::cout << "Here? Setup H!" << std::endl;
    this->set_mat_from_h5(&_H_, HFile);

    _hamiltonian_ready_ = true;
};

void Solver::setup_KSP(){
    if ( !_hamiltonian_ready_ ) { throw std::runtime_error("Please setup Hamiltonian before configuring the KSP."); }
    if ( !_overlap_ready_ ) { throw std::runtime_error("Please setup overlap before configuring the KSP."); }
    if ( !_sqrtm_ready_ ) { throw std::runtime_error("Please setup S^1/2 before configuring the KSP."); }
    if ( !_kpoint_ready_ ) { throw std::runtime_error("Please setup kpoint before configuring the KSP."); }
    
    KSPCreate(_comm_,&_ksp_);
    KSPSetDM(_ksp_, _dm_atomic_->dm());
    KSPSetDMActive(_ksp_,PETSC_FALSE);
    KSPSetOperators(_ksp_,_S_,_S_);
    KSPGetPC(_ksp_,&_pc_);
    KSPSetReusePreconditioner(_ksp_, PETSC_TRUE);
    // KSPSetInitialGuessNonzero(_ksp_, PETSC_TRUE);

    if(_n_sample_== 1){
        KSPSetType(_ksp_, KSPCG);
    }else{
        KSPSetType(_ksp_, KSPHPDDM);
        KSPHPDDMSetType(_ksp_, KSP_HPDDM_TYPE_BCG);
    }
    
    // approximated invert square root

    Mat Sinv;

    MatCreateNormalHermitian(_Sr_, &Sinv);
    PCSetType(_pc_, PCMAT);
    PCSetOperators(_pc_, _S_, Sinv);

    KSPSetTolerances(_ksp_, 1e-8, 1e-9, PETSC_DEFAULT, PETSC_DEFAULT);
    // PCSetUp(_pc_);
    KSPSetFromOptions(_ksp_);
    KSPSetUp(_ksp_);
}

void Solver::scale_shift_H(double emin, double emax, bool verbose){
    PetscScalar _emax_, _emin_;
    if (emax <= emin) { // in this case we think the input emin, emax is not meaningful
        EPS eps;
        PetscInt nconv=1;
        EPSCreate(_comm_, &eps);

        if (_explicit_orth_){
            EPSSetOperators(eps, _H_, NULL);
            EPSSetProblemType(eps, EPS_HEP);
        } else {
            EPSSetOperators(eps, _H_, _S_);
            EPSSetProblemType(eps, EPS_GHEP);

            ST st;  KSP ksp;  PC pc;
            EPSGetST(eps,&st);                 // grab the ST
            STGetKSP(st,&ksp);                 // grab its KSP        :contentReference[oaicite:3]{index=3}
            KSPSetType(ksp,KSPCG);
            KSPGetPC(ksp,&pc);

            KSPSetReusePreconditioner(ksp, PETSC_TRUE);
            
            Mat Sinv;
            MatCreateNormalHermitian(_Sr_, &Sinv);
            PCSetType(pc, PCMAT);
            PCSetOperators(pc, _S_, Sinv);
            STSetPreconditionerMat(st, Sinv);
            KSPSetFromOptions(ksp);
            KSPSetTolerances(ksp,1e-8, 1e-9, PETSC_DEFAULT, PETSC_DEFAULT);

            KSPSetUp(ksp);
        }

        // config the KSP internel of SLEPC

        EPSSetTolerances(eps,1e-2,50);
        EPSSetDimensions(eps,1,100,100);
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);

        // temp
        BV V;
        EPSGetBV(eps, &V);
        // BVSetMatrix(V,NULL,PETSC_FALSE);
        BVSetDefiniteTolerance(V, 1e-9);

        EPSSetUp(eps);
        // EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);

        EPSSolve(eps);

        EPSGetConverged(eps, &nconv);
        EPSGetEigenpair(eps,0,&_emax_, NULL, NULL, NULL);

        EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
        EPSSolve(eps);
        EPSGetConverged(eps, &nconv);
        EPSGetEigenpair(eps,0,&_emin_, NULL, NULL, NULL);
        EPSDestroy(&eps);

        if (_dm_atomic_->rank() == 0 && verbose){
            std::cout<<"Estimated Emax : "<< _emax_ << "  Emin : " << _emin_ <<"\n";
        }
    } else {
        _emax_ = emax;
        _emin_ = emin; 
    }
    /* Use Slepc to get the spectrum amplitude, and scale+shift H*/

    _scale_ = (_emax_ - _emin_).real() / 2;
    _scale_ += 4; // allow for a bondary distance of 10 eV
    _shift_ = (_emax_ + _emin_).real() / 2;

    // scale and shift the H/S
    if (_explicit_orth_){
        if (!_explicit_form_){
            MContext *ctx;
            MatShellGetContext(_H_, (void**)&ctx);
            MatAXPY(ctx->H, -(PetscScalar)_shift_, _S_, SAME_NONZERO_PATTERN);
            MatScale(ctx->H, (PetscScalar)(1/_scale_));
        }else{
            MatShift(_H_, -(PetscScalar)_shift_);
            MatScale(_H_, (PetscScalar)(1/_scale_));
        }
    }else{
        MatAXPY(_H_, -(PetscScalar)_shift_, _S_, SAME_NONZERO_PATTERN);
        MatScale(_H_, (PetscScalar)(1/_scale_));
    }
}

void Solver::orthogonalize_H(double tol, double sptol, int maxit, bool explicit_formation, bool verbose){
    if (!_hamiltonian_ready_) throw std::runtime_error("Hamiltonian must be ready before orthogonalization.");
    if (_explicit_orth_) return; // Already orthogonalized

    if (explicit_formation) {
        Ssqrtm(tol, sptol, maxit, verbose, true);
    } else {
        // Setup Htmp as MatShell Htmp where the action is Htmp v = S^-1/2 H S^-1/2 v
        Ssqrtm(tol, sptol, maxit, verbose, false);
        Mat Htmp;

        MContext *ctx;
        PetscNew(&ctx);
        ctx->H = _H_; ctx->Sr = _Sr_;
        MatCreateVecs(_H_, &ctx->w1, NULL);
        PetscObjectReference((PetscObject)ctx->H);
        PetscObjectReference((PetscObject)ctx->Sr);

        PetscInt local_rows, local_cols, global_rows, global_cols;
        MatGetSize(_H_, &global_rows, &global_cols);
        MatGetLocalSize(_H_, &local_rows, &local_cols);

        MatCreateShell(_comm_,
                       local_rows, local_cols, global_rows, global_cols,
                       (void*)ctx, &Htmp);
        MatShellSetOperation(Htmp, MATOP_MULT, (void(*)(void))SrHSrMult);
        MatShellSetOperation(Htmp, MATOP_DESTROY, (void(*)(void))SrHSrDestroy);
        MatSetUp(Htmp);
        MatDestroy(&_H_);
        _H_ = Htmp;
    }
    _explicit_orth_ = true;
    _explicit_form_ = explicit_formation;
}

void Solver::setup_sample(int n, std::string method){
    if(n==1){
        DMCreateGlobalVector(_dm_atomic_->dm(), &_Sample_);
    }else{
        PetscInt global_rows, local_rows;
        Vec tmp;

        DMGetGlobalVector(_dm_atomic_->dm(), &tmp);               // Get temporary vector from DM
        VecGetSize(tmp, &global_rows);
        VecGetLocalSize(tmp, &local_rows);
        DMRestoreGlobalVector(_dm_atomic_->dm(), &tmp);

        MatCreate(PetscObjectComm((PetscObject)(_dm_atomic_->dm())), &_Samples_);
        MatSetSizes(_Samples_, local_rows, PETSC_DECIDE, global_rows, n);
        MatSetType(_Samples_, MATDENSE);
        MatSetDM(_Samples_, _dm_atomic_->dm());
        MatSetUp(_Samples_);
    }
    _n_sample_ = n;

    if(method=="gauss"){
        PetscRandom r;
        PetscRandomCreate(_comm_, &r);
        PetscRandomSetType(r, PETSCRAND48);
        PetscRandomSetInterval(r, 0.0, 1.0);
        if(_n_sample_==1){
            // VecSetRandom(_Sample_, r);
            VecSetGaussianRandom(_Sample_, r);
            VecNormalize(_Sample_, NULL);
        }else{
            // MatSetRandom(_Samples_, r);
            Vec col;
            for (PetscInt j = 0; j < n; ++j) {
                MatDenseGetColumnVec(_Samples_,j,&col);
                VecSetGaussianRandom(col, r);
                VecNormalize(col,NULL);
                MatDenseRestoreColumnVec(_Samples_,j,&col);
            }
        }
        PetscRandomDestroy(&r);
    // }
    // else if (method=="hutch"){
    //     /* code */
    //     continue;
    // } else if (method=="hutch++"){
    //     continue;
    }else {
        throw std::invalid_argument("sampling method inappropriate, need to choose from gauss/hutch/hutch++");
    }

    _samples_ready_ = true;
    
};

void Solver::compute_dos(int nt, std::string proj_mode, const int *orbtype, int *n_proj, double *dos, std::unique_ptr<double[]>& pdos, double *omegas, bool verbose){
    /*
        We should have dos allocated before this function's call, but instead have pdos unallocated as a unique_ptr, since we do not know the size of the projection before.
    */

    if ( !(_kpoint_ready_ && _overlap_ready_ && _hamiltonian_ready_ && _samples_ready_ && _sqrtm_ready_) ) 
    { 
        throw std::runtime_error("Not ready for computing dos, setup: 1. samples, 2. kpoint 3. Overlap 4. S^-1/2 5. Hamiltonian first!"); 
    }

    bool proj=true;
    int n_proj_type=1;
    std::complex<double> imI(0.0, 1.0); 

    IS isSrc, isDst;
    int n_atom_type = _dm_atomic_->n_type();

    if(proj_mode=="element"){
        rsatb::backend::setup_is_atomic(*_dm_atomic_, &isSrc, &isDst);
        n_proj_type = n_atom_type;
    }else if(proj_mode=="orbital"){
        int norb=0;
        for(int i=0; i<n_atom_type;i++) norb += _dm_atomic_->type_norb()[i];
        n_proj_type = (*std::max_element(orbtype, orbtype + norb))+1;
        rsatb::backend::setup_is_orbital(*_dm_atomic_, orbtype, n_atom_type, _dm_atomic_->type_norb(), &isSrc, &isDst);
    }else{
        proj=false;
    }

    if(n_proj!=nullptr) *n_proj = n_proj_type;

    PetscComplex cdos[2*nt+1], pcdos[(2*nt+1)*n_proj_type];

    if(_n_sample_==1){
        rsatb::calc::corr_dos<Vec>(_H_, _ksp_, _Sample_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos, verbose, _explicit_orth_);
    } else {
        rsatb::calc::corr_dos<Mat>(_H_, _ksp_, _Samples_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos, verbose, _explicit_orth_);
    }


    // Processing cdos and pcdos to dos and pdos
    if(rank==0){
        /* Computing the smearing window */
        PetscComplex factor, poscorr, negcorr;
        // cdos[nt] *= std::exp(-imI*_shift_*M_PI*nt/_scale_);
        // if(proj){
        //     for(int ip=0; ip < n_proj_type; ++ip){ // pcdos have shape [nt, n_proj_type]
        //         pcdos[nt+n_proj_type*ip] *= std::exp(-imI*_shift_*M_PI*nt/_scale_);
        //     }
        // }

        for(int i=1;i<nt+1;++i){
            poscorr = std::exp(imI*_shift_*M_PI*i/_scale_);
            negcorr = std::exp(-imI*_shift_*M_PI*i/_scale_);
            factor = 0.5 * (1 + std::cos(M_PI * (i-1) / nt));
            cdos[nt+i] *= factor * negcorr;
            cdos[nt-i] *= factor * poscorr;

            if(proj){
                for(int ip=0; ip < n_proj_type; ++ip){ // pcdos have shape [nt, n_proj_type]
                    pcdos[(nt+i)*n_proj_type+ip] *= factor * negcorr;
                    pcdos[(nt-i)*n_proj_type+ip] *= factor * poscorr;
                }
            }
        }

        // for(int i=0;i<(2*nt)*n_proj_type;++i) std::cout << pcdos[i+n_proj_type] << " ";
        // std::cout << "\n";

        std::complex<double> temp_dos[2*nt], temp_pdos[(2*nt)*n_proj_type];
        ifft_1d(2*nt, reinterpret_cast<cd*>(cdos+1), temp_dos);
        if(proj) {
            bifft_1d(2*nt, n_proj_type, reinterpret_cast<cd*>(pcdos+n_proj_type), temp_pdos);
            pdos = std::make_unique<double[]>(2*nt*n_proj_type);
        }

        for(int i=0;i<nt;++i){
            dos[i] = std::abs(temp_dos[nt+i]);
            dos[nt+i] = std::abs(temp_dos[i]);
            omegas[i] = (i-nt) * _scale_ / nt + _shift_;
            omegas[nt+i] = i * _scale_ / nt + _shift_;
            if(proj){
                for(int ip=0; ip<n_proj_type; ++ip) {
                    pdos[i*n_proj_type+ip] = std::abs(temp_pdos[(i+nt)*n_proj_type+ip]);
                    pdos[(i+nt)*n_proj_type+ip] = std::abs(temp_pdos[i*n_proj_type+ip]);
                }
            }
        }
    }
};

void Solver::compute_ac(int nt, int n_kpm, double temp, double mu, double *ac, double *omegas){
    if ( !(_kpoint_ready_ && _overlap_ready_ && _hamiltonian_ready_ && _samples_ready_ && _position_ready_ && _sqrtm_ready_) ) 
    { 
        throw std::runtime_error("Not ready for computing ac, setup: 1. samples, 2. kpoint 3. Overlap 4. S^-1/2 5. Hamiltonian 6. position first!"); 
    }

    double beta = 1.0 / (temp * 8.617333262e-5);
    double ac_prefactor = this->norb() / this->volume();

    double t_step = 3.14159265358979323846 / _scale_;

    for( int i=0; i<nt+1; ++i ){
        omegas[i] = i * _scale_ / (nt+1);
    }

    PetscComplex cac[(nt+1)*9];
    if(_n_sample_==1){
        rsatb::calc::corr_ac<Vec>(
            _H_, _Rx_, _Ry_, _Rz_, _ksp_,
            _Sample_, nt, temp, mu, _scale_, _shift_,
            n_kpm, 1e-14, cac, _explicit_orth_
        );
    } else {
        rsatb::calc::corr_ac<Mat>(
            _H_, _Rx_, _Ry_, _Rz_, _ksp_,
            _Samples_, nt, temp, mu, _scale_, _shift_,
            n_kpm, 1e-14, cac, _explicit_orth_
        );
    }

    int idx;
    for( int iac=0; iac<9; ++iac ){
        for( int iw=0; iw<=nt; ++iw ){
            idx = iw*9+iac;
            ac[idx] = 0.0;
            if (omegas[iw] != 0){
                for( int it=0; it<=nt; ++it ){
                    // idx = iw*9+iac
                    ac[idx] += 2.0 * std::sin(omegas[iw]*it*t_step) * exp(-2.0*std::pow((it+1) / (nt+1), 2)) * cac[it*9+iac].imag();
                }
                ac[idx] = ac_prefactor * t_step * ac[idx] * (exp(-beta * omegas[iw]) - 1) / omegas[iw];
            }
            

        }
    }
}


// void Solver::compute_fermi_dirac(int n_kpm, double temp, double mu, double *occ){
//     if ( !(_kpoint_ready_ && _overlap_ready_ && _hamiltonian_ready_ && _samples_ready_) ) 
//     { 
//         throw std::runtime_error("Not ready for computing dos, setup: 1. samples, 2. kpoint 3. Overlap 4. Hamiltonian first!"); 
//     }

    
//     if (_n_sample_>1){
//         rsatb::calc::fermi_dirac_kpm<Mat>(_H_, _ksp_, _Sample_, );
//     }
//     else{
//         Vec ovec;
//         VecDuplicate(_Sample_, ovec);
//         rsatb::calc::fermi_dirac_kpm<Vec>(_H_, _ksp_, _Samples_, ovec, temp, mu, _scale_, _shift_, n_kpm);
//     }


// }
};