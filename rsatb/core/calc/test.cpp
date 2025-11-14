#include <gtest/gtest.h>
#include "cheby.hpp"
#include "dos.hpp"
#include "ac.hpp"
#include "fft.hpp"
#include "fermi_dirac.hpp"
#include <mpi.h>
#include <iostream>

using namespace rsatb::calc;
// using namespace rsatb::backend;

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
    rsatb::calc::cheby_iter(H, ksp, v_prev, v_curr, v_next);
    // VecView(v_curr,PETSC_VIEWER_STDOUT_SELF);

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
    rsatb::calc::cheby_iter_mult(H, ksp, &v_prev, &v_curr, &v_next, &v_tmp);
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
    PetscComplex coeff[]={1.0,1.0,1.0,1.0};

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
        v_psi_out
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
    PetscComplex coeff[]={1.0,1.0,1.0,1.0};

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
        v_psi_out
    );
    MatView(v_psi_out,PETSC_VIEWER_STDOUT_SELF);

    MatDestroy(buff);   MatDestroy(buff+1);  MatDestroy(buff+2); MatDestroy(buff+3);  MatDestroy(&v_psi_in);  MatDestroy(&v_psi_out);
    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp);
    
}

TEST(calcTest, TestTimePropCoeff) {
    PetscComplex *coeff;
    int n_bes;
    double precision = 1.0e-14;
    timeprop_coeff(3.14, precision, true,
            &coeff, &n_bes);
    for(int i=0;i<n_bes;i++) std::cout << coeff[i];
    std::cout<<"\n";

    delete coeff; // since the mem of coeff is newed in timeprop_coeff
}


TEST(calcTest, TestCDOS) {
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar Psi_in[] = {0.80072413, -0.59903328};

    Mat  H,S,m_psi_in;
    Vec  v_psi_in;
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&m_psi_in);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
            MatSetValue(m_psi_in,j,i,Psi_in[j],INSERT_VALUES);
        }   
    }

    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(m_psi_in,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (m_psi_in,MAT_FINAL_ASSEMBLY);

    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,Psi_in,&v_psi_in);
    

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    PetscComplex cdos[21]={0};

    std::cout << "Hello I am here!\n";

    rsatb::calc::corr_dos<Vec>(H, ksp, v_psi_in, 10, false, 1, nullptr, nullptr, cdos, nullptr);

    std::cout << "The function works!\n";
    for(int i=0;i<21;++i) std::cout << cdos[i] << " ";
    std::cout << "\n";

    PetscComplex cdos_m[21]={0};
    rsatb::calc::corr_dos<Mat>(H, ksp, m_psi_in, 10, false, 1, nullptr, nullptr, cdos_m, nullptr);
    for(int i=0;i<21;++i) std::cout << cdos_m[i] << " ";
    std::cout << "\n";

    IS isSrc, isDst;
    PetscInt src[2]={0,1};
    PetscInt dst[2]={1,0};

    ISCreateGeneral(MPI_COMM_WORLD,2,src,PETSC_COPY_VALUES,&isSrc);
    ISCreateGeneral(MPI_COMM_WORLD,2,dst,PETSC_COPY_VALUES,&isDst);

    std::cout<<"IS Create Success!\n";

    PetscComplex cdos_mp[21]={0};
    PetscComplex pcdos_mp[42]={0};
    rsatb::calc::corr_dos<Mat>(H, ksp, m_psi_in, 10, true, 2, &isSrc, &isDst, cdos_mp, pcdos_mp);
    std::cout<<"Function run success!\n";
    for(int i=0;i<21;++i) std::cout << cdos_mp[i] << " ";
    std::cout << "\n";
    for(int i=0;i<42;++i) std::cout << pcdos_mp[i] << " ";
    std::cout << "\n";

    rsatb::calc::corr_dos<Vec>(H, ksp, v_psi_in, 10, true, 2, &isSrc, &isDst, cdos_mp, pcdos_mp);
    std::cout<<"Function run success!\n";
    for(int i=0;i<21;++i) std::cout << cdos_mp[i] << " ";
    std::cout << "\n";
    for(int i=0;i<42;++i) std::cout << pcdos_mp[i] << " ";
    std::cout << "\n";

    MatDestroy(&H);        MatDestroy(&S);
    KSPDestroy(&ksp); VecDestroy(&v_psi_in);
    MatDestroy(&m_psi_in);
}


TEST(calcTest, Testfft) {
    std::complex<double> in[6] = {std::complex<double>(2.0, 0.0),std::complex<double>(3.0, 0.0),std::complex<double>(4.0, 0.0),std::complex<double>(3.0, 0.0),std::complex<double>(2.0, 0.0),std::complex<double>(1.0, 0.0)};
    std::complex<double> out[6];
    std::complex<double> bin[12], bout[12];

    for(int i=0;i<6;i++){
        std::cout << in[i] << " ";
    }
    std::cout << "\n";

    rsatb::calc::fft_1d(6, in, out);

    for(int i=0;i<6;i++){
        std::cout << out[i] << " ";
    }
    std::cout << "\n";

    rsatb::calc::ifft_1d(6, in, out);

    for(int i=0;i<6;i++){
        bin[2*i] = in[i];
        bin[2*i+1] = in[i];
        std::cout << out[i] << " ";
    }
    std::cout << "\n";

    rsatb::calc::bfft_1d(6, 2, bin, bout);

    for(int i=0;i<12;i++){
        std::cout << bout[i] << " ";
    }
    std::cout << "\n";

    // GTEST_ASSERT_EQ(out[0], std::complex<double>(0.0, 0.0));
}

TEST(calcTest, TestMFN){
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

    rsatb::calc::fermi_dirac_krylov(MPI_COMM_WORLD, H, ksp, v_curr, v_next, 300.0, -0.9, 1e-14);

    VecView(v_next,PETSC_VIEWER_STDOUT_SELF);

}


TEST(calcTest, TestFermiKPM){
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar curr[]={0.80072413, -0.59903328};
    const PetscScalar next[]={0.0,0.0};
    const PetscScalar next_tgt[]={0.17984554, 0.3707678};

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

    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,curr,    &v_curr);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,next,    &v_next);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    rsatb::calc::fermi_dirac_kpm(
        H, ksp, v_curr, v_next, 3000.0, 0.0, 1.0, 0.0, 500, 1e-14
    ); // 10K 1996 moment for 1e-14 acc

    VecView(v_next,PETSC_VIEWER_STDOUT_SELF);

    rsatb::calc::fermi_dirac_kpm(
        H, ksp, v_curr, v_next, 3000.0, 0.0, 1.0, 0.0, 500, 1e-14, true
    ); // 10K 1996 moment for 1e-14 acc

    VecView(v_next,PETSC_VIEWER_STDOUT_SELF);

}

TEST(calcTest, TestCurrent){
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar Rx_in[]={-0.50633759,  2.8521911, 2.8521911 , -3.1169422};
    const PetscScalar Ry_in[]={-1.72592812, -1.15052508, -1.15052508,  3.24730585};
    const PetscScalar Rz_in[]={-2.56982708,  1.332095  ,  1.332095  , -0.10829134};
    const PetscScalar Psi_in[] = {0.80072413, -0.59903328};


    Mat  H,S,Rx,Ry,Rz,m_psi_in, m_psix, m_psiy, m_psiz, m_buff1, m_buff2, m_buff3;
    Vec  v_psi_in, psix, psiy, psiz, buff1, buff2, buff3, temp;
    KSP  ksp;

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Rx);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Ry);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Rz);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&m_psi_in);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
            MatSetValue(Rx,i,j,Rx_in[i*2+j],INSERT_VALUES);
            MatSetValue(Ry,i,j,Ry_in[i*2+j],INSERT_VALUES);
            MatSetValue(Rz,i,j,Rz_in[i*2+j],INSERT_VALUES);
            MatSetValue(m_psi_in,j,i,Psi_in[j],INSERT_VALUES);
        }   
    }

    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Rx,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Rx,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Ry,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Ry,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Rz,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Rz,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(m_psi_in,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (m_psi_in,MAT_FINAL_ASSEMBLY);


    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,Psi_in,&v_psi_in);
    VecDuplicate(v_psi_in, &psix);
    VecDuplicate(v_psi_in, &temp);
    VecDuplicate(v_psi_in, &psiy);
    VecDuplicate(v_psi_in, &psiz);
    VecDuplicate(v_psi_in, &buff1);
    VecDuplicate(v_psi_in, &buff2);
    VecDuplicate(v_psi_in, &buff3);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    PetscScalar *out;
    PetscScalar out_label[2];

    rsatb::calc::current(H, ksp, Rx, Ry, Rz, v_psi_in, psix, psiy, psiz, buff1, buff2);

    out_label[0] = PetscComplex(0.0, 7.20774155);
    out_label[1] = PetscComplex(0.0, 6.52346567);
    VecGetArray(psix, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -6.24458614);
    out_label[1] = PetscComplex(0.0, -5.65174861);
    VecGetArray(psiy, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -0.08882071);
    out_label[1] = PetscComplex(0.0, -0.08038841);
    VecGetArray(psiz, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    rsatb::calc::current_adjoint(H, ksp, Rx, Ry, Rz, v_psi_in, psix, psiy, psiz, buff1, buff2, buff3);

    out_label[0] = PetscComplex(0.0, 3.2857697);
    out_label[1] = PetscComplex(0.0, 7.50314678);
    VecGetArray(psix, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -2.84669918);
    out_label[1] = PetscComplex(0.0, -6.50051699);
    VecGetArray(psiy, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -0.04049041);
    out_label[1] = PetscComplex(0.0, -0.09246098);
    VecGetArray(psiz, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_psix);
    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_psiy);
    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_psiz);
    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_buff1);
    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_buff2);
    MatDuplicate(m_psi_in, MAT_SHARE_NONZERO_PATTERN, &m_buff3);

    // current mult
    rsatb::calc::current_mult(H, ksp, Rx, Ry, Rz, &m_psi_in, &m_psix, &m_psiy, &m_psiz, &m_buff1, &m_buff2);

    out_label[0] = PetscComplex(0.0, 7.20774155);
    out_label[1] = PetscComplex(0.0, 6.52346567);
    MatGetColumnVector(m_psix, temp, 0);
    VecGetArray(temp, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -6.24458614);
    out_label[1] = PetscComplex(0.0, -5.65174861);
    MatGetColumnVector(m_psiy, temp, 0);
    VecGetArray(temp, &out);


    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -0.08882071);
    out_label[1] = PetscComplex(0.0, -0.08038841);
    MatGetColumnVector(m_psiz, temp, 0);
    VecGetArray(temp, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    rsatb::calc::current_mult_adjoint(H, ksp, Rx, Ry, Rz, &m_psi_in, &m_psix, &m_psiy, &m_psiz, &m_buff1, &m_buff2, &m_buff3);

    out_label[0] = PetscComplex(0.0, 3.2857697);
    out_label[1] = PetscComplex(0.0, 7.50314678);
    MatGetColumnVector(m_psix, temp, 0);
    VecGetArray(temp, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -2.84669918);
    out_label[1] = PetscComplex(0.0, -6.50051699);
    MatGetColumnVector(m_psiy, temp, 0);
    VecGetArray(temp, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    out_label[0] = PetscComplex(0.0, -0.04049041);
    out_label[1] = PetscComplex(0.0, -0.09246098);
    MatGetColumnVector(m_psiz, temp, 0);
    VecGetArray(temp, &out);

    ASSERT_NEAR(out_label[0].real(), out[0].real(), 1e-7);
    ASSERT_NEAR(out_label[0].imag(), out[0].imag(), 1e-7);
    ASSERT_NEAR(out_label[1].real(), out[1].real(), 1e-7);
    ASSERT_NEAR(out_label[1].imag(), out[1].imag(), 1e-7);

    MatDestroy(&H); MatDestroy(&S); MatDestroy(&Rx); MatDestroy(&Ry); MatDestroy(&Rz); MatDestroy(&m_psix); MatDestroy(&m_psiy); MatDestroy(&m_psiz);
    MatDestroy(&m_buff1); MatDestroy(&m_buff2); MatDestroy(&m_buff3); MatDestroy(&m_psi_in);
    VecDestroy(&v_psi_in); VecDestroy(&psix); VecDestroy(&psiy); VecDestroy(&psiz); VecDestroy(&buff1); VecDestroy(&buff2); VecDestroy(&buff3);
    VecDestroy(&temp);
}
// TEST(calcTest, TestChebyRecur);

// TEST(calcTest, TestChebyRecurMult);

TEST(calcTest, TestCORRAC){
    const PetscScalar H_in[]={0.46500671, -0.38921396, -0.38921396, -0.52943857};
    const PetscScalar S_in[]={0.53371569, -0.0947521 , -0.0947521 ,  0.76628431};
    const PetscScalar Rx_in[]={-0.50633759,  2.8521911, 2.8521911 , -3.1169422};
    const PetscScalar Ry_in[]={-1.72592812, -1.15052508, -1.15052508,  3.24730585};
    const PetscScalar Rz_in[]={-2.56982708,  1.332095  ,  1.332095  , -0.10829134};
    const PetscScalar Psi_in[] = {0.80072413, -0.59903328};


    Mat  H,S,Rx,Ry,Rz,m_psi_in;
    Vec  v_psi_in;
    KSP  ksp;
    PetscComplex ac[12];

    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&H);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&S);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Rx);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Ry);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&Rz);
    MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&m_psi_in);

    for (PetscInt i=0; i<2; ++i){
        for (PetscInt j=0; j<2; ++j){
            MatSetValue(H,i,j,H_in[i*2+j],INSERT_VALUES);
            MatSetValue(S,i,j,S_in[i*2+j],INSERT_VALUES);
            MatSetValue(Rx,i,j,Rx_in[i*2+j],INSERT_VALUES);
            MatSetValue(Ry,i,j,Ry_in[i*2+j],INSERT_VALUES);
            MatSetValue(Rz,i,j,Rz_in[i*2+j],INSERT_VALUES);
            MatSetValue(m_psi_in,j,i,Psi_in[j],INSERT_VALUES);
        }   
    }

    MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (H,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (S,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Rx,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Rx,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Ry,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Ry,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Rz,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Rz,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(m_psi_in,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (m_psi_in,MAT_FINAL_ASSEMBLY);
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,2,Psi_in,&v_psi_in);

    /* Set the ksp */
    KSPCreate(PETSC_COMM_SELF,&ksp);
    // KSPSetDM(ksp, dm);
    KSPSetOperators(ksp,S,S);
    KSPSetType(ksp, KSPPREONLY);
    KSPSetUp(ksp);

    rsatb::calc::corr_ac<Vec>(H, Rx, Ry, Rz, ksp, v_psi_in, 1, 300.0, 0.0, 1.0, 0.0, 1000, 1e-14, ac);

    ASSERT_NEAR(5.7386113981, ac[0].real(), 1e-7);
    ASSERT_NEAR(-4.97177277359, ac[1].real(), 1e-7);
    ASSERT_NEAR(-0.07071667868, ac[2].real(), 1e-7);
    ASSERT_NEAR(4.3074051887, ac[3].real(), 1e-7);
    ASSERT_NEAR(0.0612669569, ac[4].real(), 1e-7);
    ASSERT_NEAR(0.0008714388, ac[5].real(), 1e-7);

    ASSERT_NEAR(5.73765599, ac[6].real(), 1e-7);
    ASSERT_NEAR(0.10471105, ac[6].imag(), 1e-7);
    ASSERT_NEAR(-4.97094504, ac[7].real(), 1e-7);
    ASSERT_NEAR(-0.09071873, ac[7].imag(), 1e-7);
    ASSERT_NEAR(-0.070704905, ac[8].real(), 1e-7);
    ASSERT_NEAR(-0.001290350, ac[8].imag(), 1e-7);
    ASSERT_NEAR(4.30668806, ac[9].real(), 1e-7);
    ASSERT_NEAR(0.078596181, ac[9].imag(), 1e-7);
    ASSERT_NEAR(0.0612567569, ac[10].real(), 1e-7);
    ASSERT_NEAR(0.0011179233, ac[10].imag(), 1e-7);
    ASSERT_NEAR(0.000871293, ac[11].real(), 1e-7);
    ASSERT_NEAR(0.00001590093, ac[11].imag(), 1e-7);

    // test multiple RHS
    rsatb::calc::corr_ac<Mat>(H, Rx, Ry, Rz, ksp, m_psi_in, 1, 300.0, 0.0, 1.0, 0.0, 1000, 1e-14, ac);
    // for( int i=0; i<12; ++i) {
    //     std::cout << ac[i] << std::endl;
    // }
    ASSERT_NEAR(5.7386113981, ac[0].real(), 1e-7);
    ASSERT_NEAR(-4.97177277359, ac[1].real(), 1e-7);
    ASSERT_NEAR(-0.07071667868, ac[2].real(), 1e-7);
    ASSERT_NEAR(4.3074051887, ac[3].real(), 1e-7);
    ASSERT_NEAR(0.0612669569, ac[4].real(), 1e-7);
    ASSERT_NEAR(0.0008714388, ac[5].real(), 1e-7);

    ASSERT_NEAR(5.73765599, ac[6].real(), 1e-7);
    ASSERT_NEAR(0.10471105, ac[6].imag(), 1e-7);
    ASSERT_NEAR(-4.97094504, ac[7].real(), 1e-7);
    ASSERT_NEAR(-0.09071873, ac[7].imag(), 1e-7);
    ASSERT_NEAR(-0.070704905, ac[8].real(), 1e-7);
    ASSERT_NEAR(-0.001290350, ac[8].imag(), 1e-7);
    ASSERT_NEAR(4.30668806, ac[9].real(), 1e-7);
    ASSERT_NEAR(0.078596181, ac[9].imag(), 1e-7);
    ASSERT_NEAR(0.0612567569, ac[10].real(), 1e-7);
    ASSERT_NEAR(0.0011179233, ac[10].imag(), 1e-7);
    ASSERT_NEAR(0.000871293, ac[11].real(), 1e-7);
    ASSERT_NEAR(0.00001590093, ac[11].imag(), 1e-7);

    MatDestroy(&H); MatDestroy(&S); MatDestroy(&Rx); MatDestroy(&Ry); MatDestroy(&Rz); MatDestroy(&m_psi_in);
    VecDestroy(&v_psi_in);
}

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    SlepcFinalize();
    PetscFinalize();
    return result;
}