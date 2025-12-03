#include <petsc.h>
#include "cheby.hpp"
#include <math.h>

namespace rsatb::calc{

template<typename T>
void corr_dos(Mat H, KSP ksp, T Psi_in, int nt, bool proj, int n_proj_type, IS *isSrc, IS *isDst, PetscComplex *cdos, PetscComplex *pcdos, bool verbose=false, bool orthogonal=false){
    /*
        1. Create Bessel grid and time grid;
        2. Do orbital selection according if computing pdos
        3. perform propergation, record the cdos
        4. proj: whether to compute the pdos
    */
    // *nt = ceil(Escale / dE);
    double dt = M_PI;
    PetscComplex *coeff;
    PetscInt matdim, nrhs;
    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)H, &comm);
    int rank;
    MPI_Comm_rank(comm, &rank);

    if constexpr (std::is_same_v<T, Vec>){
        VecGetSize(Psi_in, &matdim);
    } else {
        MatGetSize(Psi_in, &matdim, &nrhs);
    }
    
    VecScatter scat;                               /* reusable for the whole run   */
    Vec        sum, v_tmp;
    const PetscScalar *pv;
    int n_bes=0;
    if(proj){
        VecCreateSeq(MPI_COMM_SELF, n_proj_type, &sum);
        if constexpr (std::is_same_v<T, Vec>){
            VecScatterCreate(Psi_in,*isSrc,sum,*isDst,&scat);
        }else{
            MatDenseGetColumnVecRead(Psi_in, 0, &v_tmp);
            VecScatterCreate(v_tmp,*isSrc,sum,*isDst,&scat);
            MatDenseRestoreColumnVecRead(Psi_in, 0, &v_tmp);
        }
    }

    cdos[nt] = 1;
    if(proj){
        Vec init_tmp;
        if constexpr (std::is_same_v<T, Vec>) {
            VecDuplicate(Psi_in, &init_tmp);
        } else {
            MatDenseGetColumnVecRead(Psi_in, 0, &v_tmp);
            VecDuplicate(v_tmp, &init_tmp);
            MatDenseRestoreColumnVecRead(Psi_in, 0, &v_tmp);
        }
        // initialize the pdos
        VecZeroEntries(sum);
        VecSet(init_tmp, 1/matdim);
        VecScatterBegin(scat,init_tmp,sum,ADD_VALUES,SCATTER_FORWARD);
        VecScatterEnd  (scat,init_tmp,sum,ADD_VALUES,SCATTER_FORWARD);
        VecGetArrayRead(sum, &pv);
        PetscArraycpy(pcdos+nt*n_proj_type,pv,n_proj_type);
        VecRestoreArrayRead(sum, &pv);

        VecDestroy(&init_tmp);
    }
    timeprop_coeff(dt, 1e-14, true, &coeff, &n_bes);

    if constexpr (std::is_same_v<T, Vec>){
        
        Vec  buff[3], Psi_out, Psi_curr;
        VecDuplicate(Psi_in, buff);
        VecDuplicate(Psi_in, buff+1);
        VecDuplicate(Psi_in, buff+2);
        VecDuplicate(Psi_in, &Psi_out);
        VecDuplicate(Psi_in, &Psi_curr);
        VecCopy(Psi_in, Psi_curr);

        for(int it=0; it<nt; ++it){
            if ( rank == 0 && verbose){
                std::cout << "corr_dos calculation: " << it << " / " << nt << "\n";
            }

            cheby_recur(
                H, ksp, Psi_curr,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out, ChebyMode::NORMAL, orthogonal
                );

            if(proj){
                cdos[nt+it+1] = 0; // give 0 initial value, since proj cdos is computed from accumulating the pdos

                VecPointwiseMult(Psi_curr, Psi_out, Psi_in); // TODO: This work because Psi_in have no imaginary part, if not, need more consideration
                VecZeroEntries(sum);
                VecScatterBegin(scat,Psi_curr,sum,ADD_VALUES,SCATTER_FORWARD);
                VecScatterEnd  (scat,Psi_curr,sum,ADD_VALUES,SCATTER_FORWARD);
                // copy the value from sum to _pcdos_
                VecGetArrayRead(sum, &pv);
                PetscArraycpy(pcdos+(nt+it+1)*n_proj_type,pv,n_proj_type);
                VecRestoreArrayRead(sum, &pv);

                for (PetscInt i=0; i<n_proj_type; ++i) {
                    cdos[nt+it+1] += pcdos[(nt+it+1)*n_proj_type+i];
                    // copy pdos to negative time
                    pcdos[(nt-it-1)*n_proj_type+i] = PetscConj(pcdos[(nt+it+1)*n_proj_type+i]);
                }
            }else{
                VecDot(Psi_out, Psi_in, cdos + (nt+it+1)); // notice the order of out and in
            }
            // copy dos to negative time
            cdos[nt-it-1] = PetscConj(cdos[nt+it+1]);
            VecCopy(Psi_out, Psi_curr);
        }

        VecDestroy(buff); VecDestroy(buff+1); VecDestroy(buff+2);
        VecDestroy(&Psi_out); VecDestroy(&Psi_curr);
    }else{
        Mat buff[4], Psi_out, Psi_curr;
        Vec v_psi_in, v_psi_out, v_psi_curr;
        PetscScalar temp=0;
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+2);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+3);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_out);
        MatDuplicate(Psi_in, MAT_COPY_VALUES, &Psi_curr);

        for(int it=0; it<nt; ++it){
            if (rank == 0 && verbose){
                std::cout << "corr_dos calculation: " << it << " / " << nt << "\n";
            }
            cdos[nt+it+1] = 0;
            if(proj){
                for (PetscInt i=0; i<n_proj_type; ++i) pcdos[(nt+it+1)*n_proj_type+i] = 0;
            }

            cheby_recur_mult(
                H, ksp, Psi_curr,
                buff, // length 4 vec
                coeff, n_bes,
                Psi_out, ChebyMode::NORMAL, orthogonal
                );

            for(int in=0; in<nrhs; ++in){
                temp = 0;
                MatDenseGetColumnVecRead(Psi_in, in, &v_psi_in);
                MatDenseGetColumnVecRead(Psi_out, in, &v_psi_out);
                if(proj){
                    // initialize pcdos with zero;
                    MatDenseGetColumnVec(Psi_curr, in, &v_psi_curr);
                    VecPointwiseMult(v_psi_curr, v_psi_in, v_psi_out);
                    VecZeroEntries(sum);
                    VecScatterBegin(scat,v_psi_curr,sum,ADD_VALUES,SCATTER_FORWARD);
                    VecScatterEnd  (scat,v_psi_curr,sum,ADD_VALUES,SCATTER_FORWARD);
                    MatDenseRestoreColumnVec(Psi_curr, in, &v_psi_curr);
                    // copy the value from sum to _pcdos_
                    VecGetArrayRead(sum, &pv);
                    for (PetscInt i=0; i<n_proj_type; ++i) {
                        pcdos[(nt+it+1)*n_proj_type+i] += pv[i];
                        temp += pv[i];
                    }
                    VecRestoreArrayRead(sum, &pv);
                }else{
                    VecDot(v_psi_out, v_psi_in, &temp);
                }
                cdos[nt+it+1] += temp;
                MatDenseRestoreColumnVecRead(Psi_in, in, &v_psi_in);
                MatDenseRestoreColumnVecRead(Psi_out, in, &v_psi_out);
            }

            cdos[nt+it+1] /= nrhs;
            cdos[nt-it-1] = PetscConj(cdos[nt+it+1]);
            if(proj){
                for (PetscInt i=0; i<n_proj_type; ++i) {
                    pcdos[(nt+it+1)*n_proj_type+i] /= nrhs;
                    pcdos[(nt-it-1)*n_proj_type+i] = PetscConj(pcdos[(nt+it+1)*n_proj_type+i]);    
                }
            }
            MatCopy(Psi_out, Psi_curr, SAME_NONZERO_PATTERN);
        }

        MatDestroy(buff); MatDestroy(buff+1); MatDestroy(buff+2); MatDestroy(buff+3);
        MatDestroy(&Psi_out); MatDestroy(&Psi_curr);
    }
    delete coeff;
};

template void corr_dos<Vec>(Mat, KSP, Vec, int, bool, int, IS*, IS*, PetscComplex*, PetscComplex*, bool, bool);
template void corr_dos<Mat>(Mat, KSP, Mat, int, bool, int, IS*, IS*, PetscComplex*, PetscComplex*, bool, bool);
}