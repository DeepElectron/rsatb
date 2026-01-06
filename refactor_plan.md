There will be several needs for this refactor. 

1. First, it was doing so for maximize the performance. Current Petsc's sparse MPIAIJ type does not support very efficient sparse MVP, it have good performance on single kernel, but the distributed ones does not utilize the advance kernel such as MKL, it fall back the its own implementation, which is not very good. The connectivity of the LCAO matrix is not that sparse actaully, so the MPi based distribution have large communication overhead. It should be used only when the current process number is not very large, and need multinode distributed computation. Therefore, this is not a standard case that PETSc solves, which is mainly MPI distributed computation that PETSc is designed for. 

2. PETSc/SLEPc is too hard to install, and the dependencies are huge, and not very available on differene env. Actaully, I am using only very basic features of them, so removing them with extra work, in exchange of a simple and easy to install alternative is a good idea. 

3. Last, the PETSc does not really gives me good degree of freedom to control and optimize the computation. The PETSc does not support variable block size, which is a common case in quantum chemistry. And matching different blocks size with RHS would accelerate the computation vastly.

So we need a replacing refactor plan:

1. design graph structure and matrix, vector, multivector structure that hold the connectivity of the atomicgraph. It should be a general implemenetation that is irrelevent with the atoms information.

2. support operations that needed for rsatb algorithm. Such as MVP, AXPY, MatMult, etc..

3. Support constructing the matrix from the connectivity graph distributedly and locally. Support assembling process that accumulate the values that needed for the HR2HK operation.

4. Rewrite the DMAtomic class to support the atomic Hamiltonian, Overlap, Position and Velocity matrix construction. The new class will be named as AtomicData. It should accept the atomic data as input, and pass that to the dist_graph for communication and matrix construction. It should be noticed that the atomic data have i,j,R corresponding to a edge, since the atomic system are periodic, while the dist_graph requires a reduced connectivity graph, so we need to remove the repetitive edges. The atomic info are important for the matrix assembling process in HR2HK.

The AtomicData should also reimplemented the three-body interaction preallocation generation process of DMAtomic.

5. Rewrote the Hr2hk functionalities. This should be extended as a general operators formation code in the backend folder, that use the newly developed data structure. Please adjust the input parameter name to match current data structure also. 

For reading mat from H5File, the process should be the same. For overlap integrals and the position matrix, it should also be similar. Things changed for the velocity matrix, since it requires the three body matrix, you should carefully treat it as it requires the formation of matrix by adding blocks on process that does not owns it. Please carefully read the code and decide the rules for the assembly.

6. Interfacing the with other packages to provide some extra support. 

First, the DBCSR. We should provide the interface for the transformation from the BlockSpMat to the DBCSR matrix. And the transformation back. Beaware that the transformation backward is not that simple, since the DBCSR might gives additional blocks that are not in the original BlockSpMat, so you need to first conting the blocks, preallocate the memory, and then copy the values. Please read the current version for PETSc's Mat and the DBCSR interface I build in @dbcsr.hpp and @dbcsr_capi.h

Second, the PRIMME for eigenvalues. You should read the current code in @solver.hpp and @solver.cpp that use the SLEPc to compute the largest eigenvalue of overlap matrix, and the spectrum bond of the (H,S) pair. This is vital for the function of the solver. So, adding the support for iterative eigenvalue computation is important. You should use the PRIMME lib to do this for simplicity, and do not transform data structure, instead, wrap the current new datastructure to provide the required operations that needed for the PRIMME interfaces. 

These two DBCSR data type transformation should be added in the backend module, while the PRIMME should be added in the calc module.

7. Reimplementing the calc module, creating a new folder calc_, and reimplementing all the calculation function in the calc module using the new data structure. Beaware that the calculation may need some functions of the matrix, vector or multivector to perform computation, but it might not be currently supported, please read and design a detail plan first and review it with me. Please remind that, many of the computation in calc are optimizable by adding a specific matrix or vector operation supported to it, but it is not done so since the old code use PETSc. Currently, you should take that into account and design a plan that can be extended to support such operations to maximize the efficiency.

8. The rsatb class. It is a interface class that allow the c++ user and the python user to use the rsatb code.

8. For the MVP, add a sparsity aware version that check the sparsity pattern of the RHS vector, and if it is sparse, skip the dense kernel corresponding to the 0 rols of the RHS vector.

9. Optimization:
    - The vector and multivector ghost grabing comm is not very efficient curently, it can be optimized use threading, and preallocation of send and recv buffer. The storage can also be optimized to avoid the packing and unpacking. Same as the dot product, and scale computation. Using the LA packages is also possible.
    - The matrix assembly process can be optimized by using threading, and preallocation of send and recv buffer. Or direct calling the kernels of BLAS, MKL, OpenBLAS, etc for adding the matrix and matrix. This need to be taken care if we also want threading for filling multiple blocks at the same time.
    - Mult and Multdense is also optimizable, but not very sure how now, can be done later.

10. It is good to have a HS container like feature class that grap a set of blockcsr class according to the R vector, it is helpful to avoid the io or 2c integrals again when forming matrix at different K point, and easier to implement method like the velocity matrix calculation in real space method Vij=Hij(Rj-Ri).
<!-- Third, the MKL, BLAS, OpenBLAS, etc. for linear algebra support of the dense MVP, MM kernel. -->

RSATB -> Solver:
The solver class should be replaced with RSATB class, which provide all functions that use the implemented new interfaces. The new RSATB class should have these functions:

class construction:

1. comm partition. The RSATB should incharge of the all parrallel structure of the code. Two parralization are needed, one is the K point parrallelization, the other is the atom parrallelization. First, the RSATB class should divide the totoal comm into subcomm, determined by the user's input of npar. Then, each kpoint will be assigned to a subcomm, and the atom parrallelization will be done within each subcomm. Then, each subcomm will incharge of the parallelization of the matrix construction, and the matrix assembly, and the matrix solve, each process in the subcomm will contains a partition of atoms. Each subcomm's rank 0 process will form a new comm, called master comm to gather the result after solve.

The result gathar will also be done at the master comm, once all calculation is done.

2. AtomicData construction:
The atomic data should be constructed at the class constructor stage. It will calls the from point or from file function of the atomic data to perform the construction. The resulting atomicdata are owned by each subcomm, and is partitioned across the subcomm.

3. Matrix allocation.
The overlap, Hamiltonian matrix need to be allocated. The overlap matrix should have a input tag, indicating whether this system is orthogonal or not. If it is, the overlap matrix should be empty. The Hamiltonian and Overlap should be allocated as a image container class.

class function:
1. setup (Hfile, kpoints, orb_names, orb_dirs, psd_names, psd_dirs, precond="H/S"):
This step setup the Hamiltonian and/or Overlap files, the kpoints, the scale/shift factor, and the preconditioner. The image container will first read in the files to construct the Hamiltonian and Overlap matrix for each R, using the orb and psd given. In orthogonal case, the overlap will not be constructed. Then, the kpoint will be looped to compute the preconditioner first. Each preconditioner will be saved in the RSATB class as a map from the kpoint to the preconditioner. The constructed Hamiltonian and/or Overlap matrix at each k will also be saved as maps. in each kpoint loop, the computed preconditioner will be used to compute the largest and lowest eigenvalues of the Hamiltonian operator using the PRIMME interface. The computed eigenvalues will be communicated globally to find the largest and lowest, which determined the final scale/shift factor. If the precond is on H, then after the comptuation of preconditioner, the overlap matrix will be dropped as nullptr, mem will be freeed.


2. Compute_dos
It should loop over all kpoints in each subcomm and compute the does using the setup Hamiltonian, overlap and preconditioner. Then, it should collect the result of the dos from all subcomm to the master comm.

3. compute_ac
It should loop over all kpoints in each subcomm. It should load the current Hamiltonian and overlap, preconditioner matrix. It also need to call the operator construction function to compute the velocity matrix for this. Then, it should perform the AC calculation.