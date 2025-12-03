#include <petsc.h>
#include <gtest/gtest.h>
#include "calc/cheby.hpp"

// Mock or minimal setup for testing ChebyMode::CONJUGATE

using namespace rsatb::calc;

class ChebyTest : public ::testing::Test {
protected:
    Mat H;
    KSP ksp;
    Vec psi_in, psi_out;
    PetscInt n = 4;

    void SetUp() override {
        // Initialize PETSc if not already
        PetscInitialize(NULL, NULL, NULL, NULL);

        // Create a simple 4x4 matrix H
        MatCreate(PETSC_COMM_WORLD, &H);
        MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, n, n);
        MatSetFromOptions(H);
        MatSetUp(H);

        // Fill H with some values (non-symmetric to distinguish transpose)
        // H = [ 1 2 0 0 ]
        //     [ 0 2 3 0 ]
        //     [ 0 0 3 4 ]
        //     [ 5 0 0 4 ]
        PetscInt rows[] = {0, 1, 2, 3};
        PetscInt cols[] = {0, 1, 2, 3};
        PetscScalar vals[] = {
            1, 2, 0, 0,
            0, 2, 3, 0,
            0, 0, 3, 4,
            5, 0, 0, 4
        };
        // Set values row by row
        for(int i=0; i<n; ++i) {
            for(int j=0; j<n; ++j) {
                if (vals[i*4+j] != 0)
                    MatSetValue(H, i, j, vals[i*4+j], INSERT_VALUES);
            }
        }
        MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

        // Create KSP (Identity for simplicity, S=I => S^-1=I)
        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, H, H); // Just to set something, we won't solve with H really if we want S=I
        // Actually we want S=I, so let's create an Identity matrix for KSP
        Mat S;
        MatCreate(PETSC_COMM_WORLD, &S);
        MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, n, n);
        MatSetFromOptions(S);
        MatSetUp(S);
        MatZeroEntries(S);
        MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
        MatShift(S, 1.0); // S = I
        KSPSetOperators(ksp, S, S);
        KSPSetType(ksp, KSPPREONLY);
        PC pc;
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCNONE); // S^-1 = I

        VecCreate(PETSC_COMM_WORLD, &psi_in);
        VecSetSizes(psi_in, PETSC_DECIDE, n);
        VecSetFromOptions(psi_in);
        VecSet(psi_in, 1.0); // |x> = [1, 1, 1, 1]^T

        VecDuplicate(psi_in, &psi_out);
    }

    void TearDown() override {
        MatDestroy(&H);
        KSPDestroy(&ksp);
        VecDestroy(&psi_in);
        VecDestroy(&psi_out);
        PetscFinalize();
    }
};

TEST_F(ChebyTest, ConjugateMode) {
    // Test ChebyMode::CONJUGATE
    // Operation: T_next = -T_prev + 2 (S^-1)^T H^T T_curr
    // With S=I, this is -T_prev + 2 H^T T_curr
    
    // Let's manually compute one iteration
    // T_0 = (S^-1)^T H^T |x> = H^T |x>
    // H^T = [ 1 0 0 5 ]
    //       [ 2 2 0 0 ]
    //       [ 0 3 3 0 ]
    //       [ 0 0 4 4 ]
    // |x> = [1, 1, 1, 1]^T
    // H^T |x> = [6, 4, 6, 8]^T

    // Use cheby_recur with n_coeff=3 (minimal)
    // coeff = [c0, c1, c2]
    // result = c0*T0 + c1*T1 + c2*T2 ... wait, cheby_recur formula is:
    // Psi_out = c0*T0 + c1*T1 + c2*T2
    // But cheby_recur implementation:
    // T_0 = (S^-1)^T H^T |Psi_in> (This is actually T_1 in standard notation if T_0=Psi_in)
    // Wait, let's check cheby_recur implementation again.
    
    // cheby_recur:
    // buff[0] (T_prev) = (S^-1)^T H^T Psi_in
    // buff[1] (T_curr) = 2 (S^-1)^T H^T buff[0] - Psi_in
    // Psi_out = c0*Psi_in + c1*buff[0] + c2*buff[1]

    // So:
    // Term 0: Psi_in
    // Term 1: (S^-1)^T H^T Psi_in
    // Term 2: 2 ((S^-1)^T H^T)^2 Psi_in - Psi_in

    // Let's set coeffs: c0=0, c1=1, c2=0 -> Result should be Term 1 = H^T |x>
    PetscComplex coeffs[] = {0.0, 1.0, 0.0};
    
    cheby_recur(H, ksp, psi_in, nullptr, coeffs, 3, psi_out, ChebyMode::CONJUGATE, false);

    // Check result
    const PetscScalar* array;
    VecGetArrayRead(psi_out, &array);
    EXPECT_NEAR(std::real(array[0]), 6.0, 1e-10);
    EXPECT_NEAR(std::real(array[1]), 4.0, 1e-10);
    EXPECT_NEAR(std::real(array[2]), 6.0, 1e-10);
    EXPECT_NEAR(std::real(array[3]), 8.0, 1e-10);
    VecRestoreArrayRead(psi_out, &array);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
