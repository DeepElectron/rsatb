#include <gtest/gtest.h>
#include "cheby.hpp"
#include <mpi.h>
#include <iostream>


using namespace rsatb::calc;

TEST(calcTest, TestChebyIter) {
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar prev[]={-1.16125422*PETSC_i, -0.15076596*PETSC_i};
    const PetscScalar curr[]={-0.79143568,  0.59208448};
    const PetscScalar next[]={0.0,0.0};
    const PetscScalar next_tgt[]={1.13431303*PETSC_i, 0.14726822*PETSC_i};

    Mat  H,S;
    Vec  v_prev,v_curr,v_next;
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
        }
    }
    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);

    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,prev,    &v_prev);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,curr,    &v_curr);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,next,    &v_next);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    /*Do the test*/
    rsatb::calc::cheby_iter(H, ksp, v_prev, v_curr, v_next, -2*PETSC_i);
    VecView(v_curr,PETSC_VIEWER_STDOUT_SELF);

    VecDestroy(&v_prev);   VecDestroy(&v_curr);  VecDestroy(&v_next);
    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp);
    
}

TEST(calcTest, TestChebyIterMulti) {
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar prev[]={-1.16125422*PETSC_i, -0.15076596*PETSC_i};
    const PetscScalar curr[]={-0.79143568,  0.59208448};
    const PetscScalar next[]={0.0,0.0};
    const PetscScalar next_tgt[]={1.13431303*PETSC_i, 0.14726822*PETSC_i};

    Mat  H,S,v_prev,v_curr,v_next,v_tmp;
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,&v_prev);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,&v_curr);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,&v_next);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
        }
        MatSetValue(v_prev,i,0,prev[i],INSERT_VALUES);
        MatSetValue(v_curr,i,0,curr[i],INSERT_VALUES);
        MatSetValue(v_next,i,0,next[i],INSERT_VALUES);
    }
    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(v_prev,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (v_prev,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(v_curr,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (v_curr,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(v_next,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (v_next,MAT_FINAL_ASSEMBLY);

    MatDuplicate(v_next, MAT_SHARE_NONZERO_PATTERN, &v_tmp);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    /*Do the test*/
    rsatb::calc::cheby_iter_mult(H, ksp, &v_prev, &v_curr, &v_next, &v_tmp, -2*PETSC_i);
    MatView(v_curr,PETSC_VIEWER_STDOUT_SELF);

    MatDestroy(&v_prev);   MatDestroy(&v_curr);  MatDestroy(&v_next);
    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp);
    
}


TEST(calcTest, TestChebyRecur) {
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar Psi_in[] = {0.80072413, -0.59903328};
    PetscScalar prev[]={-1.16125422*PETSC_i, -0.15076596*PETSC_i};
    PetscScalar curr[]={-0.79143568,  0.59208448};
    PetscScalar next[]={0.0,0.0};
    const PetscScalar next_tgt[]={1.13431303*PETSC_i, 0.14726822*PETSC_i};
    PetscReal coeff[]={1.0,1.0,1.0,1.0};

    Mat  H,S;
    Vec  v_psi_in, v_psi_out;
    Vec  buff[3];
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
        }
    }
    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);

    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,Psi_in,    &v_psi_in);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,Psi_in,    &v_psi_out);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,prev,    buff);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,curr,    buff+1);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,next,    buff+2);
    // VecView(buff[0],PETSC_VIEWER_STDOUT_SELF);

    

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    /*Do the test*/
    rsatb::calc::cheby_recur(
        H, ksp, v_psi_in,
        buff, // length 3 vec
        coeff, 4, 
        v_psi_out, true
    );
    VecView(v_psi_out,PETSC_VIEWER_STDOUT_SELF);

    VecDestroy(buff);   VecDestroy(buff+1);  VecDestroy(buff+2);  VecDestroy(&v_psi_in);  VecDestroy(&v_psi_out);
    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp);
    
}

TEST(calcTest, TestChebyRecurMulti) {
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar Psi_in[] = {0.80072413, -0.59903328};
    PetscScalar prev[]={-1.16125422*PETSC_i, -0.15076596*PETSC_i};
    PetscScalar curr[]={-0.79143568,  0.59208448};
    PetscScalar next[]={0.0,0.0};
    const PetscScalar next_tgt[]={1.13431303*PETSC_i, 0.14726822*PETSC_i};
    PetscReal coeff[]={1.0,1.0,1.0,1.0};

    Mat  H,S,v_psi_in,v_psi_out;
    Mat buff[4];
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,buff);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,buff+1);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,buff+2);
    MatCreateSeqDense(PETSC_COMM_SELF,2,1,NULL,&v_psi_in);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
        }
        MatSetValue(buff[0],i,0,prev[i],INSERT_VALUES);
        MatSetValue(buff[1],i,0,curr[i],INSERT_VALUES);
        MatSetValue(buff[2],i,0,next[i],INSERT_VALUES);
        MatSetValue(v_psi_in,i,0,Psi_in[i],INSERT_VALUES);
    }
    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(buff[0],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (buff[0],MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(buff[1],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (buff[1],MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(buff[2],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (buff[2],MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(v_psi_in,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (v_psi_in,MAT_FINAL_ASSEMBLY);

    MatDuplicate(buff[0], MAT_SHARE_NONZERO_PATTERN, buff+3);
    MatDuplicate(buff[0], MAT_SHARE_NONZERO_PATTERN, &v_psi_out);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    /*Do the test*/
    rsatb::calc::cheby_recur_mult(
        H, ksp, v_psi_in,
        buff, // length 3 vec
        coeff, 4, 
        v_psi_out, true
    );
    MatView(v_psi_out,PETSC_VIEWER_STDOUT_SELF);

    MatDestroy(buff);   MatDestroy(buff+1);  MatDestroy(buff+2); MatDestroy(buff+3);  MatDestroy(&v_psi_in);  MatDestroy(&v_psi_out);
    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp);
    
}


// TEST(calcTest, TestChebyRecur);

// TEST(calcTest, TestChebyRecurMult);

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}