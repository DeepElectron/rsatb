#ifndef DBCSRHPP
#define DBCSRHPP

#include "petsc.hpp"
#include "dbcsr_capi.h"
#include <dbcsr.h>
#include <complex>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

// TODO: We should allow this function work on different dtype, symmetry, block transpose simulation
using namespace rsatb::backend;

void DMat2DBCSR(Mat mat, DMAtomic &dm_atomic, dbcsr_matrix *m_dbcsr, char mat_type){
    MPI_Comm comm, group, comm_cart;
    PetscObjectGetComm((PetscObject)(dm_atomic.dm()), &comm);
    int gidx, mpi_size, mpi_rank, brd_rank, norb, nLocal, NGlobal, dims[2] = {0,0}, dims_comm[2] = {0,0}, period[2] = {1,1};
    PetscInt vStart, vEnd, eStart, eEnd, gindex;
    bool ghost;
    dbcsr_matrix m_brd;
    dbcsr_distribution dist_brd, dist_dbcsr;
    const bool keep_sparsity = false, summation = false, mutable_work=true;
    std::unordered_set<std::string> esm;
    AtomData atomdata;
    
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);
    dims_comm[0] = mpi_size;
    dims_comm[1] = 1;

    // this function assumes the dbcsr is initialized
    MPI_Dims_create(mpi_size, 2, dims);
    MPI_Cart_create(comm, 2, dims, period, 0, &group);
    MPI_Cart_create(comm, 2, dims_comm, period, 0, &comm_cart);

    MPI_Comm_rank(comm, &brd_rank);

    // std::cout << "The cart dims: " << dims[0] << " " << dims[1] << std::endl;

    // build the graph for DBCSR
    dm_atomic.get_vertex_range_local(&vStart, &vEnd);
    nLocal = 0;
    for(int k=vStart; k<vEnd; ++k){
        ghost = dm_atomic.atom_is_ghost(k);
        if(!ghost){
            nLocal++;
        }
    }
    NGlobal = nLocal;
    MPI_Allreduce(MPI_IN_PLACE, &NGlobal, 1, MPI_INT, MPIU_SUM, comm);

        // block size list & build distribution list
    std::vector<int> blk_size((size_t)NGlobal, 0), dist_row_brd((size_t)NGlobal, 0), dist_col_brd((size_t)NGlobal, 0), dist_row_dbcsr((size_t)NGlobal, 0), dist_col_dbcsr((size_t)NGlobal, 0);

    for(int k=vStart; k<vEnd; ++k){
        ghost = dm_atomic.atom_is_ghost(k);
        if(!ghost){
            dm_atomic.get_atom_norb(k, &norb);
            gindex = dm_atomic.LVI2GAI(k);
            gidx = gindex;
            blk_size[gidx] = norb; // Here the BRD matrix should be ordered just like that of PETSc, instead of the order of atom index
            dist_row_brd[gidx] = brd_rank;
            dist_row_dbcsr[gidx] = gidx % dims[0]; /* TODO: need optimization, CP2K use a very complex 2D partition algorithm 
                                                            for load balancing and communication reduction */
            dist_col_dbcsr[gidx] = gidx % dims[1];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, blk_size.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, dist_row_brd.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, dist_row_dbcsr.data(), NGlobal, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, dist_col_dbcsr.data(), NGlobal, MPI_INT, MPI_MAX, comm);

    c_dbcsr_distribution_new(&dist_brd, comm_cart, dist_row_brd.data(), NGlobal, dist_col_brd.data(), NGlobal);
    c_dbcsr_distribution_new(&dist_dbcsr, group, dist_row_dbcsr.data(), NGlobal, dist_col_dbcsr.data(), NGlobal);
    // std::cout << "NGlobal: " << NGlobal << std::endl;
    // for(int p=0; p<NGlobal; ++p){
    //     std::cout << "dist_row " << p << " " << dist_row_brd[p] << std::endl;
    //     std::cout << "dist_col " << p << " " << dist_col_brd[p] << std::endl;
    //     std::cout << "blk_size " << p << " " << blk_size[p] << std::endl;
    // }

    c_dbcsr_create_new(&m_brd, "BRDtmp", dist_brd, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, &mutable_work, nullptr); // datatype canbe optimized
    c_dbcsr_create_new(m_dbcsr, "MATDBCSR", dist_dbcsr, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, &mutable_work, nullptr); // datatype canbe optimized

        // Fill the blocks locally
    PetscInt i, j, igindex, jgindex, offset_i, offset_j; 
    int inorb, jnorb, eorb, igidx, jgidx;
    dm_atomic.get_edge_range_local(&eStart, &eEnd);
    std::complex<double> scale = 1.0;
    std::string edge_string;
    // bool tr=false; // found=false;

    for (int iv=vStart; iv<vEnd; iv++){
        ghost = dm_atomic.atom_is_ghost(iv);
        if (!ghost){
            gindex = dm_atomic.LVI2GAI(iv);
            gidx = gindex;
            norb = blk_size[gidx];
            edge_string = std::to_string(iv)+"-"+std::to_string(iv);

            dm_atomic.get_offset_local(iv, &offset_i);
            std::vector<PetscInt> idxm((size_t)norb, 0);
            std::vector<PetscScalar> y((size_t)norb*norb, 0);
            for(int im=0;im<norb;im++) idxm[im] = im+offset_i;

            MatGetValuesLocal(mat, (PetscInt)norb, idxm.data(), (PetscInt)norb, idxm.data(), y.data()); // ensure local filling
            c_dbcsr_put_block2d_z_api(m_brd, gidx, gidx, y.data(), norb, norb, &summation, &scale);

            esm.insert(edge_string);
        }
        // std::cout << "diagonal filling at block " << gidx << std::endl;
    }
    
    // int count=0;
    // PetscBool ghost2;
    for (int ie=eStart; ie<eEnd; ie++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        dm_atomic.get_connected_atoms(ie, &i, &j);
        edge_string = std::to_string(i)+"-"+std::to_string(j);
        igindex = dm_atomic.LVI2GAI(i); jgindex = dm_atomic.LVI2GAI(j);
        igidx = igindex; jgidx = jgindex;
        inorb = blk_size[igidx];
        jnorb = blk_size[jgidx];
        
        if (esm.find(edge_string) == esm.end()){ // not exist
            // Not found, fill in the value
            // DMNetworkIsGhostVertex(dm,i,&ghost);
            // DMNetworkIsGhostVertex(dm,j,&ghost2);
            int nvar, idxi, idxj;
            dm_atomic.get_atom_id(i, &idxi); dm_atomic.get_atom_id(j, &idxj);
            // std::cout << igidx << " " << jgidx << " " << i << " " << j << " " << idxi << " " << idxj << " " << mpi_rank << " " << inorb << std::endl;
            // std::cout << i << " " << j << " " << mpi_rank << " " << ie << std::endl;
            dm_atomic.get_offset_local(i, &offset_i); dm_atomic.get_offset_local(j, &offset_j);
            std::vector<PetscInt> idxm((size_t)inorb, 0), idxn((size_t)jnorb, 0);
            std::vector<PetscScalar> y((size_t)inorb*jnorb, 0); 
            

            for(int im=0;im<inorb;im++) idxm[im] = im+offset_i;
            for(int in=0;in<jnorb;in++) idxn[in] = in+offset_j;

            MatGetValuesLocal(mat, (PetscInt)inorb, idxm.data(), (PetscInt)jnorb, idxn.data(), y.data()); // ensure local filling

            c_dbcsr_put_block2d_z_api(m_brd, igidx, jgidx, y.data(), inorb, jnorb, &summation, &scale);

            // std::cout << "Off-diag filling at block " << igidx << " " << jgidx << std::endl;
            esm.insert(edge_string);
            // count ++;

            // if (count > 10){
            //     break;
            // }
        }
        // PetscScalar *kk;
        // std::cout << "here??" << std::endl;
        // std::cout << igidx << " " << jgidx << " " << inorb << " " << jnorb << std::endl;
        // c_dbcsr_get_block_p_z(m_brd, igidx, jgidx, reinterpret_cast<double _Complex**>(&kk), &tr, &found, nullptr, nullptr);
        // std::cout << "here!!" << std::endl;
    }

    // std::complex<double> L(2,1);
    // c_dbcsr_add_on_diag_z_api(m_brd, L);
    c_dbcsr_finalize(m_brd);
    c_dbcsr_complete_redistribute(m_brd, *m_dbcsr, &keep_sparsity, &summation);
    c_dbcsr_finalize(*m_dbcsr);
    c_dbcsr_release(&m_brd);
    c_dbcsr_distribution_release(&dist_brd);
    MPI_Comm_free(&comm_cart);
    // add check, e.g. the trace of matrix of different format, to see whether they match
};

void DBCSR2Mat(
        DMAtomic &dm_atomic, 
        dbcsr_matrix m_dbcsr, 
        Mat *mat,
        bool release=true,
        bool profile=false
){
    /*

    */

    // MPI Commucator related init: the BRD & PETSC
    MPI_Comm comm, group, comm_cart;
    PetscObjectGetComm((PetscObject)(dm_atomic.dm()), &comm);
    c_dbcsr_get_group(m_dbcsr, &group);

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int dims_comm[2]={size,1}, period[2] = {1,1};
    MPI_Cart_create(comm, 2, dims_comm, period, 0, &comm_cart);

    // init local and global information from DM
        // oStart/End the ownership starts ends, mat local indices
    PetscInt vStart, vEnd, oStart=1073741824, oEnd=0, n=0, offset;
    bool ghost;
    PetscInt compkey, nblk;
    void *ptr = nullptr;
    int nLocal, NGlobal;

    dm_atomic.get_vertex_range_local(&vStart, &vEnd);
    nLocal = 0;
    for(int k=vStart; k<vEnd; ++k){
        ghost = dm_atomic.atom_is_ghost(k);
        if(!ghost){
            nLocal++;
        }
    }

    NGlobal = nLocal;
    MPI_Allreduce(MPI_IN_PLACE, &NGlobal, 1, MPI_INT, MPIU_SUM, comm);

    // DBCSR & global offset
    std::vector<int> blk_size((size_t)NGlobal, 0), offsets((size_t)NGlobal, 0);
    PetscInt gindex;
    c_dbcsr_get_col_blk_size(m_dbcsr, blk_size.data(), NGlobal);
    for(int i=1; i<NGlobal; ++i) {
        offsets[i] += offsets[i-1] + blk_size[i-1];
    }

    for(int k=0; k<NGlobal; ++k){
        n += blk_size[k];
    }
    // construct brd info
    std::vector<int> dist_row_brd((size_t)NGlobal, 0), dist_col_brd((size_t)NGlobal, 0);
    int mloc=0;
    for(int k=vStart; k<vEnd; ++k){
        ghost = dm_atomic.atom_is_ghost(k);
        gindex = dm_atomic.LVI2GAI(k);
        if(!ghost){
            dist_row_brd[(int)gindex] = rank;
            mloc += blk_size[(int)gindex];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, dist_row_brd.data(), NGlobal, MPI_INT, MPI_MAX, comm);

    if(*mat!=nullptr) MatDestroy(mat);

    MatCreate(comm, mat);
    MatSetType(*mat, "aij");
    MatSetSizes(*mat, mloc, mloc, n, n);
    PetscInt rstart;
    MatGetOwnershipRange(*mat, &rstart, nullptr);

    // create the brd matrix
    dbcsr_distribution dist_brd;
    const bool keep_sparsity = false, summation = false;
    dbcsr_matrix m_brd;
    char mat_type;

    c_dbcsr_distribution_new(&dist_brd, comm_cart, dist_row_brd.data(), NGlobal, dist_col_brd.data(), NGlobal);
    mat_type = c_dbcsr_get_matrix_type(m_dbcsr);
    c_dbcsr_create_new(&m_brd, "BRDtmp", dist_brd, mat_type, blk_size.data(), NGlobal,
        blk_size.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, nullptr, nullptr); // datatype canbe optimized
    c_dbcsr_complete_redistribute(m_dbcsr, m_brd, &keep_sparsity, &summation);

    // start iteration
    // 1. first allocate the memory
    // 2. second filling the PETSc mat
    
    dbcsr_iterator it;
    int nrow, ncol, i_blk, j_blk, blk_id, blk_p, iloc;
    PetscInt offset_i, offset_j;

    // c_dbcsr_get_proc_row_dist(m_brd, row_dist_brd.data(), NGlobal);
    bool shared=false, dynamic=false, dynamic_byrow=false, continuous_pointer=false, read_only=false;
    c_dbcsr_iterator_start(&it, m_brd, &shared, &dynamic, &dynamic_byrow, &continuous_pointer, &read_only); // the third param control multithreadding, it can be used to accelerate the iteration
    
    std::vector<PetscInt> d_nnz(mloc, 0), o_nnz(mloc, 0);
    while (c_dbcsr_iterator_blocks_left(it)) {
        
        c_dbcsr_iterator_next_block_index(it, &i_blk, &j_blk, &blk_id, &blk_p);
        if (dist_row_brd[i_blk] == rank) {
            // if (!dm_atomic.GAI_is_owned(i_blk)){
            //     throw std::runtime_error("The row ownership have some problems!");
            // }
            nrow = blk_size[i_blk];
            ncol = blk_size[j_blk];
            iloc = dm_atomic.GAI2LVI(i_blk);
            // dm_atomic.get_offset_local(iloc, &offset_i);
            offset_i = offsets[i_blk] - rstart;
            // std::cout << "Check out the offset: " << offset_i << " " << i_blk << " " << rank << std::endl;
            // std::cout << offset_i << " " << mloc << std::endl;
            // if ( dm_atomic.GAI_is_owned(j_blk) ) { // j (GAI) is not owned, off diag block
            // std::cout << "Whether the ownership aligns: " << (dist_row_brd[j_blk] == rank) << " " << dm_atomic.GAI_is_owned(j_blk) << std::endl;
            if ( dist_row_brd[j_blk] == rank ) {
                for(int ir=0; ir<nrow; ++ir){
                    d_nnz[ir+offset_i] += ncol;
                }
            } else {
                for(int ir=0; ir<nrow; ++ir){
                    o_nnz[ir+offset_i] += ncol;
                }
            }
        }
    }
    c_dbcsr_iterator_stop(&it);

    // preallocation
    MatMPIAIJSetPreallocation(*mat, 0, d_nnz.data(), 0, o_nnz.data());

    // MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);
    // MatShift(*mat, 1.0);
    // MatView(*mat, PETSC_VIEWER_STDOUT_WORLD);

    // IMPORTANT: install the DM’s local-to-global mappings if you will call SetValuesLocal
    // ISLocalToGlobalMapping rmap;
    // DMGetLocalToGlobalMapping(dm,&rmap); // or DMGetLocalToGlobalMapping(dm,&rmap); same for columns
    // MatSetLocalToGlobalMapping(*mat, rmap, rmap);

    // MatSetOption(*mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
    MatSetOption(*mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    // MatSetOption(*mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);

    // Second round
    std::complex<double> *v=nullptr;
    PetscScalar *v_petsc=nullptr; 
    bool tr, found;
    PetscBool v_free;
    double blk_max = 0;

    c_dbcsr_iterator_start(&it, m_brd, &shared, &dynamic, &dynamic_byrow, &continuous_pointer, &read_only);
    // std::cout << c_dbcsr_iterator_blocks_left(it) << std::endl;
    while (c_dbcsr_iterator_blocks_left(it)) {
        // std::cout << "start getting the blocks index" << std::endl;
        c_dbcsr_iterator_next_block_index(it, &i_blk, &j_blk, &blk_id, &blk_p); // i_blk, j_blk should follow the index as the global index of PETSc's dm vertices
        // std::cout << "Filling the Mat at block: " << i_blk << " " << j_blk << std::endl; 
        if (dist_row_brd[i_blk] == rank){
            // std::cout << "block: " << i_blk << " " << j_blk << " " << rank << std::endl;
            // std::cout << "Filling the Mat at block: " << i_blk << std::endl; 
            // get the block and put it into the mat
            // std::cout << "start getting the blocks" << std::endl;
            c_dbcsr_get_block_p_z(m_brd, i_blk, j_blk, reinterpret_cast<double _Complex**>(&v), &tr, &found, &nrow, &ncol);
            // we will assert the block has been found here.

            // std::cout << "start transform it to PETSc format" << std::endl;
            rsatb::backend::cdouble2petscscalar(nrow*ncol, v, &v_petsc, &v_free);
            std::vector<PetscInt> idxm((size_t)nrow, 0), idxn((size_t)ncol, 0);

            std::vector<PetscScalar> tmp((size_t)(nrow*ncol), 0);
            blk_max = 0;

            for(int ir=0; ir<nrow; ++ir){
                for(int ic=0; ic<ncol; ++ic){
                    tmp[ir*ncol+ic] = v_petsc[ic*nrow+ir];
                    blk_max = std::max(blk_max, std::norm(v[ic*nrow+ir]));
                }
            }

            // std::cout << "start plugin" << std::endl;

            if (blk_max > 1e-8){
                offset_i = offsets[i_blk];
                offset_j = offsets[j_blk];

                for(int im=0;im<nrow;im++) idxm[im] = im+offset_i;
                for(int in=0;in<ncol;in++) idxn[in] = in+offset_j;
                // std::cout << "filling at: " << idxm[0] << " " << idxn[0] << std::endl;
                MatSetValues(*mat, (PetscInt)nrow, idxm.data(), (PetscInt)ncol, idxn.data(), tmp.data(), INSERT_VALUES);
            }
            if(v_free) { PetscFree(v_petsc); }
        }
    }

    // std::cout << "start assembling" << std::endl;
    MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);

    if (profile){
        MatInfo info;
        MatGetInfo(*mat, MAT_LOCAL, &info);

        // 3. Check the "Mallocs" score
        // info.mallocs count how many times PETSc had to expand the memory arrays.
        if (info.mallocs == 0) {
            PetscPrintf(comm, "PERFORMANCE SUCCESS: 0 mallocs. Preallocation was perfect.\n");
        } else {
            PetscPrintf(comm, "PERFORMANCE WARNING: %g mallocs occurred during assembly.\n", info.mallocs);
            PetscPrintf(comm, "This means the preallocation logic underestimated the non-zeros.\n");
        }

        // Optional: See how much memory was used vs allocated
        PetscPrintf(comm, "[Rank %d] Used: %g, Allocated: %g\n", rank, info.nz_used, info.nz_allocated);
    }

    // MatView(*mat, PETSC_VIEWER_STDOUT_WORLD);

    // std::cout << "end assembling" << std::endl;

    // std::cout << "start releasing" << std::endl;

    c_dbcsr_iterator_stop(&it);
    c_dbcsr_finalize(m_brd);
    c_dbcsr_distribution_release(&dist_brd);
    c_dbcsr_release(&m_brd);
    MPI_Comm_free(&comm_cart);
    // std::cout << "start optional releasing" << std::endl;
    if(release) { // 0.333119, 0.313348, 0.289591
        // std::cout << "here!" << std::endl;
        dbcsr_distribution dist_dbcsr;
        // std::cout << "here!" << std::endl;
        c_dbcsr_get_distribution(m_dbcsr, &dist_dbcsr);
        // std::cout << "here!" << std::endl;
        c_dbcsr_distribution_release(&dist_dbcsr);

        c_dbcsr_release(&m_dbcsr);
        // std::cout << "here!" << std::endl;
        // std::cout << "here!" << std::endl;
        MPI_Comm_free(&group);
        // std::cout << "here!" << std::endl;
    }

    // std::cout << "end releasing" << std::endl;
    // add check, e.g. the trace of matrix of different format, to see whether they match
};

#endif