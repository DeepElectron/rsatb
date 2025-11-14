# What is this
There are different ordering system in the high performance package we use. Each of them corresponding to a index system
We need to handle them carefully to avoid bugs.

# PETSc
In petsc, there are: 
1. the input index that is used when you construct the DM object.
2. the index after ordering and distribution
3. the local index

# DBCSR
The DBCSR use consistent ordering once you have defined a distribution object, that is quite good.
But we need to deside:
    Do I want the matrix have the order of the PETSc actual storage?