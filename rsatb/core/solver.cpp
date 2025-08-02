#include "hr2hk.hpp"
#include "calc/dos.hpp"
#include "calc/fft.hpp"
#include "calc/sample.hpp"
#include "solver.hpp"
#include <math.h>
#include <iostream>
#include <algorithm>
using namespace rsatb::backend;
using namespace rsatb::io;
using namespace rsatb::calc;

namespace rsatb{

Solver::Solver(
    std::string &HFile, std::string &SFile, 
    const int n_atom, const int n_edge, 
    const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
    const double *pos,
    MPI_Comm comm,
    std::string options
    ): _HFile_(HFile), _SFile_(SFile), _mode_("H5ijR"), _n_atom_(n_atom),
       _n_edge_(n_edge), _atom_type_(atom_type), _edge_index_(edge_index),
       _type_norb_(type_norb), _edge_shift_vec_(edge_shift_vec), _pos_(pos), _comm_(comm),
       _Hhd_(_HFile_), _Shd_(_SFile_),
       _dm_(nullptr),
        _H_(nullptr), _S_(nullptr),
        _Samples_(nullptr), _Sample_(nullptr),
        _ksp_(nullptr), _pc_(nullptr)
    {
        // PetscOptionsSetValue(NULL, "-info", "stacking.log:ksp,pc");
        // PetscOptionsSetValue(NULL, "-log_view", ":profiling.log");
        PetscOptionsInsertString(NULL, options.c_str());
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
        MPI_Comm_rank(_comm_, &rank);
    };

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
    SlepcFinalize();
    PetscFinalize();
}

void Solver::setup_graph(){ /* Setup the DM, we assume here the graph information is already initialized in the class constructor method */
    
    rsatb::backend::get_dm_atomic(
        _n_atom_, _n_edge_, _edge_index_, _atom_type_, _type_norb_, _edge_shift_vec_, _pos_, &_dm_, _comm_
    );
    DMSetMatType(_dm_, MATMPIAIJMKL);

    rsatb::backend::ViewDMNetworkPartition(_dm_, false);
};

void Solver::setup_HS(double Kx, double Ky, double Kz, double emin, double emax){ /* Setup H and S, how to assembled H/S is determined by the initialization mode*/
    if (_n_sample_==-1) throw std::runtime_error("Please setup random states (samples) before configuring the Hamiltonian and Overlap Operator.");
    
    DMCreateMatrix(_dm_, &_H_);
    setMatrixR2K(_H_, _dm_, &_Hhd_, Kx, Ky, Kz);

    DMCreateMatrix(_dm_, &_S_);
    setMatrixR2K(_S_, _dm_, &_Shd_, Kx, Ky, Kz);
    PetscScalar _emax_, _emin_;

    if (emax <= emin) { // in this case we think the input emin, emax is not meaningful
        EPS eps;
        PetscInt nconv=1;

        EPSCreate(_comm_, &eps);
        EPSSetOperators(eps, _H_, _S_);
        EPSSetProblemType(eps, EPS_GHEP);
        // EPSSetType(eps,EPSLANCZOS);
        // EPSLanczosSetReorthog(eps,EPS_LANCZOS_REORTHOG_LOCAL);
        EPSSetTolerances(eps,1e-3,50);
        EPSSetDimensions(eps,1,PETSC_DEFAULT,PETSC_DEFAULT);
        // EPSSetThreshold(eps,1e-4,PETSC_FALSE); // abs, rel
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
        // EPSSetType(eps,EPSKRYLOVSCHUR);

        // config the KSP internel of SLEPC
        ST st;  KSP ksp;  PC pc;
        EPSGetST(eps,&st);                 // grab the ST
        STGetKSP(st,&ksp);                 // grab its KSP        :contentReference[oaicite:3]{index=3}
        KSPSetType(ksp,KSPCG);
        KSPGetPC(ksp,&pc);
        KSPSetReusePreconditioner(ksp, PETSC_TRUE);

        // Set for pc
        PCSetType(pc, PCASM);
        PCASMSetSubMatType(pc, MATSEQAIJMKL);
        PCASMSetOverlap(pc, 0);
        PetscOptionsInsertString(NULL,"-sub_pc_type icc");

        KSPSetTolerances(ksp,1e-8, 1e-9, PETSC_CURRENT, PETSC_CURRENT);

        EPSSetUp(eps);

        EPSSolve(eps);
        EPSGetConverged(eps, &nconv);
        EPSGetEigenpair(eps,0,&_emax_, NULL, NULL, NULL);

        EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
        EPSSolve(eps);
        EPSGetConverged(eps, &nconv);
        EPSGetEigenpair(eps,0,&_emin_, NULL, NULL, NULL);
        EPSDestroy(&eps);

        std::cout<<"Estimated Emax : "<< _emax_ << "  Emin : " << _emin_ <<"\n";
    } else {
        _emax_ = emax;
        _emin_ = emin; 
    }
    /* Use Slepc to get the spectrum amplitude, and scale+shift H*/

    _scale_ = (_emax_ - _emin_).real() / 2;
    _scale_ += 4; // allow for a bondary distance of 10 eV
    _shift_ = (_emax_ + _emin_).real() / 2;

    // scale and shift the H/S
    MatAXPY(_H_, -_shift_, _S_, SAME_NONZERO_PATTERN);
    MatScale(_H_, 1/_scale_);
    

    /* Setup the KSP solver also, need test*/
    KSPCreate(_comm_,&_ksp_);
    KSPSetDM(_ksp_, _dm_);
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
    // ICC(1), Fast in serial
    // PCSetType(_pc_, PCICC);
    // PCFactorSetLevels(_pc_, 1);

    // ASM + ICC(1)
    PCSetType(_pc_, PCASM);
    PCASMSetSubMatType(_pc_, MATSEQAIJMKL);
    PCASMSetOverlap(_pc_, 0);
    PetscOptionsInsertString(NULL,"-sub_pc_type icc");


    // GAMG performance profile
    // PCSetType(_pc_,PCGAMG);
    // PCGAMGSetType(_pc_, PCGAMGAGG);
    // PetscReal thres[1]={0.10};
    // PCGAMGSetThreshold(_pc_, thres, 1);
    // PCGAMGSetAggressiveLevels(_pc_,2);
    // PCGAMGSetCoarseEqLim(_pc_, 200);
    // PCGAMGSetReuseInterpolation(_pc_, PETSC_TRUE);

    KSPSetTolerances(_ksp_, 1e-8, 1e-9, PETSC_CURRENT, PETSC_CURRENT);
    PCSetUp(_pc_);
    KSPSetFromOptions(_ksp_);
    KSPSetUp(_ksp_);
};

void Solver::setup_velocityOp(){};

void Solver::setup_sample(int n, std::string method){
    if(n==1){
        DMCreateGlobalVector(_dm_, &_Sample_);
    }else{
        PetscInt global_rows, local_rows;
        Vec tmp;

        DMGetGlobalVector(_dm_, &tmp);               // Get temporary vector from DM
        VecGetSize(tmp, &global_rows);
        VecGetLocalSize(tmp, &local_rows);
        DMRestoreGlobalVector(_dm_, &tmp);

        MatCreate(PetscObjectComm((PetscObject)_dm_), &_Samples_);
        MatSetSizes(_Samples_, local_rows, PETSC_DECIDE, global_rows, n);
        MatSetType(_Samples_, MATDENSE);
        MatSetDM(_Samples_, _dm_);
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
    
};

void Solver::compute_dos(int nt, std::string proj_mode, const int *orbtype, int *n_proj, double *dos, std::unique_ptr<double[]>& pdos){
    /*
        We should have dos allocated before this function's call, but instead have pdos unallocated as a unique_ptr, since we do not know the size of the projection before.
    */
    bool proj=true;
    int n_proj_type=1;
    std::complex<double> imI(0.0, 1.0); 

    IS isSrc, isDst;
    int n_atom_type = (*std::max_element(_atom_type_, _atom_type_ + _n_atom_))+1;
    if(proj_mode=="element"){
        rsatb::backend::setup_is_atomic(_dm_, &isSrc, &isDst);
        n_proj_type = n_atom_type;
    }else if(proj_mode=="orbital"){
        int norb=0;
        for(int i=0; i<n_atom_type;i++) norb += _type_norb_[i];
        n_proj_type = (*std::max_element(orbtype, orbtype + norb))+1;
        rsatb::backend::setup_is_orbital(_dm_, orbtype, n_atom_type, _type_norb_, &isSrc, &isDst);
    }else{
        proj=false;
    }
    *n_proj = n_proj_type;

    PetscComplex cdos[2*nt+1], pcdos[(2*nt+1)*n_proj_type];

    if(_n_sample_==1){
        rsatb::calc::corr_dos<Vec>(_H_, _ksp_, _Sample_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos);
    } else {
        rsatb::calc::corr_dos<Mat>(_H_, _ksp_, _Samples_, nt, proj, n_proj_type, &isSrc, &isDst, cdos, pcdos);
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
            if(proj){
                for(int ip=0; ip<n_proj_type; ++ip) {
                    pdos[i*n_proj_type+ip] = std::abs(temp_pdos[(i+nt)*n_proj_type+ip]);
                    pdos[(i+nt)*n_proj_type+ip] = std::abs(temp_pdos[i*n_proj_type+ip]);
                }
            }
        }
    }
};

};