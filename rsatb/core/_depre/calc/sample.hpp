#ifndef SAMPLE
#define SAMPLE

#include <petsc.h>
#include <mpi.h>
#include "bessel.hpp"

namespace rsatb::calc{

PetscErrorCode VecSetGaussianRandom(Vec v, PetscRandom r)
{
    PetscInt       n;
    PetscScalar    *array;
    PetscInt       i;
    PetscFunctionBegin;

    VecGetLocalSize(v, &n);
    VecGetArray(v, &array);

    // Use Box-Muller transform: generate in pairs
    for (i = 0; i < n; i += 2) {
        PetscReal u1, u2;
        PetscReal r1, theta;
        PetscScalar z0, z1;

        PetscRandomGetValueReal(r, &u1);
        PetscRandomGetValueReal(r, &u2);

        if (u1 < 1e-10) u1 = 1e-10;  // avoid log(0)
        r1 = PetscSqrtReal(-2.0 * PetscLogReal(u1));
        theta = 2.0 * PETSC_PI * u2;

        z0 = r1 * PetscCosReal(theta);
        z1 = r1 * PetscSinReal(theta);

        array[i] = z0;
        if (i + 1 < n)
            array[i + 1] = z1;
    }

    VecRestoreArray(v, &array);
    PetscFunctionReturn(0);
}

};

#endif