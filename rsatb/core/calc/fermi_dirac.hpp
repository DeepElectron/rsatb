#ifndef FERMI_DIRAC
#define FERMI_DIRAC

#include <petsc.h>
#include <slepc.h>
#include "cheby.hpp"
#include "fft.hpp"
#include <math.h>
#include "kpm.hpp"


using cd = std::complex<double>;

typedef struct {
  Mat  H;          /* the Hamiltonian   */
  KSP  kspS;       /* pre-factored S    */
  Vec  tmp;        /* workspace vector  */
  bool orthogonal; /* orthogonal flag */
} Ctx;

PetscErrorCode MatMult_SinvH(Mat A, Vec x, Vec y)
{
  Ctx *ctx;  MatShellGetContext(A,&ctx);

  /* tmp = H * x */
  MatMult(ctx->H,x,ctx->tmp);

  /* y  = S^{-1} * tmp */
  if(!ctx->orthogonal) {
      KSPSolve(ctx->kspS,ctx->tmp,y);
  } else {
      VecCopy(ctx->tmp, y);
  }

  return 0;
}

namespace rsatb::calc{

inline void fermi_dirac_krylov(MPI_Comm comm, Mat H, KSP ksp, Vec Psi_in, Vec Psi_out, double T, double e_fermi, double tol, bool orthogonal=false){
    // Use SLEPc's MFN krylov method to approximate f(H)|v>
    Ctx ctx;
    Vec tmp;
    Mat A;
    PetscInt matdim;

    // for MFN
    MFN mfn;
    // FN f, p, q, r, l;
    FN f;
    PetscScalar beta = 1.0 / (T * 8.617333262e-5);

    VecDuplicate(Psi_in, &tmp);
    VecGetSize(Psi_in, &matdim);
    ctx.H = H;
    ctx.kspS = ksp;
    ctx.tmp = tmp;
    ctx.orthogonal = orthogonal;
    
    MatCreateShell(comm, PETSC_DECIDE,PETSC_DECIDE, matdim, matdim, &ctx, &A);
    MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_SinvH);

    /* Optional—tell SLEPc the operator is (non-)Hermitian */
    MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE);
    // 1 / (1+e^{a (x-mu)}), r = 1/(1+q), q = e^{beta p}, p = x-mu

    // MFN op
    MFNCreate(comm, &mfn);
    MFNSetOperator(mfn, A);

    // FNCreate(comm, &p);
    // FNCreate(comm, &q);
    // FNCreate(comm, &r);
    // FNCreate(comm, &l);
    MFNGetFN(mfn, &f);
    MFNSetType(mfn, MFNKRYLOV);
    // FNSetType(f,FNCOMBINE);
    // FNSetType(l,FNCOMBINE);

    // p = x-ef
    // FNSetType(p, FNRATIONAL);
    // PetscScalar ncoeff[2] = {1, -e_fermi};
    // PetscScalar dcoeff[1] = {1};
    // FNRationalSetNumerator(p, 2, ncoeff);
    // FNRationalSetDenominator(p, 1, dcoeff);
    // // q = exp(x/kbT)
    // FNSetType(q, FNEXP);
    // FNSetScale(q, beta, 1.0);

    // // l = q(p)(x) = exp((x-ef)/kbT)
    // FNCombineSetChildren(l, FN_COMBINE_COMPOSE, p, q);

    // // r = 1 / (x + 1)
    // FNSetType(r, FNRATIONAL);
    // PetscScalar rncoeff[1] = {1};
    // PetscScalar rdcoeff[2] = {1,1};
    // FNRationalSetNumerator(r, 1, rncoeff);
    // FNRationalSetDenominator(r, 2, rdcoeff);

    // // f = r(l)(x) = 1 / (1 + exp((x-ef)/kbT))
    // FNCombineSetChildren(f, FN_COMBINE_COMPOSE, l, r);

    /* -------------------------------------------------------------------
   constants and helper FNs
--------------------------------------------------------------------*/
    PetscScalar theta = 0.5 * beta;                     /* θ = β/2            */
    PetscScalar one   = 1.0,  plus_half  =  0.5,  minus_half = -0.5;

    /* constant 1 --------------------------------------------------------- */
    FN c_one;
    FNCreate(comm,&c_one);  FNSetType(c_one,FNRATIONAL);
    FNRationalSetNumerator  (c_one,1,&one);
    FNRationalSetDenominator(c_one,1,&one);            /* 1 / 1 = 1          */

    /* linear map  x ↦ x-μ  ---------------------------------------------- */
    FN xmum;
    PetscScalar num_xmum[2] = { 1.0, -e_fermi };       /* x - μ              */
    FNCreate(comm,&xmum);  FNSetType(xmum,FNRATIONAL);
    FNRationalSetNumerator  (xmum,2,num_xmum);
    FNRationalSetDenominator(xmum,1,&one);             /* /(1)               */

    /* exp(x)  ------------------------------------------------------------ */
    FN exp_fn;  FNCreate(comm,&exp_fn);  FNSetType(exp_fn,FNEXP);

    /* θ(x-μ) ------------------------------------------------------------- */
    FN theta_xmum, mtheta_xmum;
    FNCreate(comm,&theta_xmum);   FNSetType(theta_xmum,FNCOMBINE);
    FNCombineSetChildren(theta_xmum,FN_COMBINE_MULTIPLY,xmum,c_one);
    FNSetScale(theta_xmum,1.0,theta);          /* multiply argument by θ   */

    FNCreate(comm,&mtheta_xmum);  FNSetType(mtheta_xmum,FNCOMBINE);
    FNCombineSetChildren(mtheta_xmum,FN_COMBINE_MULTIPLY,xmum,c_one);
    FNSetScale(mtheta_xmum,1.0,-theta);         /* argument by −θ          */

    /* q = exp( θ(x-μ) )  ;  r = exp( −θ(x-μ) ) --------------------------- */
    FN q,r;
    FNCreate(comm,&q); FNSetType(q,FNCOMBINE);
    FNCombineSetChildren(q,FN_COMBINE_COMPOSE,theta_xmum,exp_fn);

    FNCreate(comm,&r); FNSetType(r,FNCOMBINE);
    FNCombineSetChildren(r,FN_COMBINE_COMPOSE,mtheta_xmum,exp_fn);

    /* -r(x) -------------------------------------------------------------- */
    FN minus_one_fn, rneg;
    PetscScalar minus_one = -1.0;
    FNCreate(comm,&minus_one_fn);  FNSetType(minus_one_fn,FNRATIONAL);
    FNRationalSetNumerator  (minus_one_fn,1,&minus_one);
    FNRationalSetDenominator(minus_one_fn,1,&one);

    FNCreate(comm,&rneg); FNSetType(rneg,FNCOMBINE);
    FNCombineSetChildren(rneg,FN_COMBINE_MULTIPLY,minus_one_fn,r);

    /* diff = q + (-r) ;  sum = q + r ------------------------------------ */
    FN diff,sum;
    FNCreate(comm,&diff); FNSetType(diff,FNCOMBINE);
    FNCombineSetChildren(diff,FN_COMBINE_ADD,q,rneg);

    FNCreate(comm,&sum ); FNSetType(sum ,FNCOMBINE);
    FNCombineSetChildren(sum ,FN_COMBINE_ADD,q,r);

    /* tanh = diff / sum -------------------------------------------------- */
    FN tanhfn; FNCreate(comm,&tanhfn); FNSetType(tanhfn,FNCOMBINE);
    FNCombineSetChildren(tanhfn,FN_COMBINE_DIVIDE,diff,sum);

    /* constants +½ and -½ ---------------------------------------------- */
    FN c_half, c_mhalf;
    FNCreate(comm,&c_half);  FNSetType(c_half ,FNRATIONAL);
    FNRationalSetNumerator  (c_half ,1,&plus_half);
    FNRationalSetDenominator(c_half ,1,&one);

    FNCreate(comm,&c_mhalf); FNSetType(c_mhalf,FNRATIONAL);
    FNRationalSetNumerator  (c_mhalf,1,&minus_half);
    FNRationalSetDenominator(c_mhalf,1,&one);

    /* (-½)·tanh --------------------------------------------------------- */
    FN prod;
    FNCreate(comm,&prod); FNSetType(prod,FNCOMBINE);
    FNCombineSetChildren(prod,FN_COMBINE_MULTIPLY,c_mhalf,tanhfn);

    /* f(x) = ½ + prod ---------------------------------------------------- */
    FN fermi_fn;
    FNCreate(comm,&fermi_fn); FNSetType(fermi_fn,FNCOMBINE);
    FNCombineSetChildren(fermi_fn,FN_COMBINE_ADD,c_half,prod);

    /* robust evaluation for tiny Hessenberg ----------------------------- */
    // FNSetMethod(fermi_fn,FN_METHOD_EIG);



    MFNSetFN(mfn,fermi_fn);


    // MFNSetFN(mfn, f);

    MFNSetFromOptions(mfn);
    MFNSetUp(mfn);
    MFNSolve(mfn, Psi_in, Psi_out);
    // FNDestroy(&p); FNDestroy(&q); FNDestroy(&r); FNDestroy(&l);
    MatDestroy(&A);
    VecDestroy(&tmp);
    MFNDestroy(&mfn);
}


inline void fermi_dirac(double E, double beta, double mu, double *out){
    double tmp = 0.0;
    if (E >= mu) {
        tmp = std::exp(beta * (mu - E));
        *out = tmp / (1.0 + tmp);
    } else {
        tmp = std::exp(beta * (E - mu));
        *out = 1.0 / (1.0 + tmp);
    }
}

inline void shifted_fermi_dirac(double E, double beta, double mu, double shift, double *out){
    double tmp = 0.0;
    double factor = std::exp(shift);
    if (E >= mu) {
        tmp = std::exp(beta * (mu - E)) / factor;
        *out = tmp / (1.0 + tmp);
    } else {
        tmp = std::exp(beta * (E - mu)) * factor;
        *out = 1.0 / (1.0 + tmp);
    }
}

// 1/(1+x)=a/(1+a)/, a= 1/x
// 1/(1+e^{beta*sc (H-sh-mu+sh)/sc})=1/(1+e^{beta*(H-mu)})
// = 1/(1+e^{beta*sc [(H-sh)/sc-(mu-sh)/sc]}

template<typename T>
inline void fermi_dirac_kpm(Mat H, KSP ksp, T Psi_in, T Psi_out, double temp, double mu, double scale, double shift, int n_kpm, 
  double eps=1e-14, ChebyMode mode=ChebyMode::NORMAL, bool orthogonal=false){
    double dE = M_PI / n_kpm;
    double mu_re = (mu - shift) / scale;
    double beta_re = scale / (temp * 8.617333262e-5);
    double shift_re = shift / (temp * 8.617333262e-5);

    PetscComplex *coeff;
    PetscInt matdim, nrhs;
    if constexpr (std::is_same_v<T, Vec>){
        VecGetSize(Psi_in, &matdim);
    } else {
        MatGetSize(Psi_in, &matdim, &nrhs);
    }

    // get coeff
    PetscComplex tmp_coeff[2*n_kpm];
    double kernel[n_kpm], fermi_tmp;
    for (int i=0; i<2*n_kpm; ++i){
      fermi_dirac(std::cos(dE*i), beta_re, mu_re, &fermi_tmp);
      tmp_coeff[i] = fermi_tmp;
    }

    jackson_kernel(n_kpm, kernel);

    fft_1d(2*n_kpm, tmp_coeff, tmp_coeff); // only n_kpt moment is useful
    for (int i=0; i<n_kpm; ++i){
      tmp_coeff[i] = tmp_coeff[i].real() * kernel[i] / n_kpm;
    }

    tmp_coeff[0] *= 0.5;

    int n_kpm_ = n_kpm;
    // Truncate fd an fd_omf according to significance
    for (int i = 0; i < n_kpm - 1; ++i) {
      if ((std::abs(static_cast<cd>(tmp_coeff[i])) <= eps) && (std::abs(static_cast<cd>(tmp_coeff[i + 1])) <= eps)) {
        n_kpm_ = i;
        break; // The truncation method need to be verify
      }
    }

    // for(int i=0; i<n_kpm; ++i) std::cout << tmp_coeff[i];
    // std::cout << "\n";

    std::cout << "fermi-dirac calculation: #KPM moments - " << n_kpm_ << "\n";

    if constexpr (std::is_same_v<T, Vec>){
      Vec  buff[3];
      VecDuplicate(Psi_in, buff);
      VecDuplicate(Psi_in, buff+1);
      VecDuplicate(Psi_in, buff+2);
      // VecDuplicate(Psi_in, &Psi_out);
      
      cheby_recur(
          H, ksp, Psi_in,
          buff, // length 3 vec
          tmp_coeff, n_kpm_, 
          Psi_out, mode, orthogonal
      );

      VecDestroy(buff); VecDestroy(buff+1); VecDestroy(buff+2);
    }else{
      Mat buff[4];
      PetscScalar temp=0;
      MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff);
      MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+1);
      MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+2);
      MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+3);
      // MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_out);
    
      cheby_recur_mult(
          H, ksp, Psi_in,
          buff, // length 4 Mat
          tmp_coeff, n_kpm_,
          Psi_out, mode, orthogonal
      );

      MatDestroy(buff); MatDestroy(buff+1); MatDestroy(buff+2); MatDestroy(buff+3);
    }

};

template void fermi_dirac_kpm<Vec>(Mat, KSP, Vec, Vec, double, double, double, double, int, double, ChebyMode, bool);
template void fermi_dirac_kpm<Mat>(Mat, KSP, Mat, Mat, double, double, double, double, int, double, ChebyMode, bool);

}

#endif