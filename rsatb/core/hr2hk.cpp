#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"
#include "hr2hk.hpp"
#include <math.h>
#include <ATen/tensor.h>

using namespace rsatb::backend;
using namespace rsatb::io;
namespace rsatb{

// PetscErrorCode SetBlockR2K(Mat mat, DM dm, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v){ // get the local idx-th edge and transform it, and add it on mat. 
//     rsatb::backend::AtomData atom_i, atom_j;
//     std::complex<double> phase;
//     PetscInt offset_i, offset_j;
//     PetscScalar *v_petsc = nullptr;
//     PetscBool v_free;
//     PetscInt idxm[inorb], idxn[jnorb];
//     // rsatb::backend::get_edge_index_from_dm(dm, idx, &i, &j);

//     // rsatb::backend::BondData bond;
//     // rsatb::backend::get_attr_from_dm<rsatb::backend::BondData>(&dm, "edge", idx, &bond);
//     cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
//     // get offset
//     DMNetworkGetGlobalVecOffset(dm, i, 0, &offset_i); /* offset is the first point position of the Dof of vertices i in the local numbering*/
//     DMNetworkGetGlobalVecOffset(dm, j, 0, &offset_j);
//     // std::cout << "Offset i,j " << offset_i << " " << offset_j << "\n";
//     // adjust data format
    
//     rsatb::backend::double2petscscalar(inorb*jnorb, v, &v_petsc, &v_free);
//     // complex phase to PETscScalar
//     PetscScalar phase_p;
//     #if defined(PETSC_USE_COMPLEX)
//     phase_p = PetscCMPLX(phase.real(),phase.imag());
//     #else
//     phase_p = static_cast<PetscScalar>(phase.real());
//     if(phase.imag() != 0) throw std::runtime_error("The PETSc should be compiled with complex type when the system have periodicity.");
//     #endif
//     for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) v_petsc[k] = v_petsc[k]*phase_p;

//     // for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) std::cout<<v_petsc[k];
//     // std::cout<<phase_p<<"\n";

//     // add the vcalue v to the mat.
//     for(int im=0;im<inorb;im++) idxm[im] = im+offset_i;
//     for(int in=0;in<jnorb;in++) idxn[in] = in+offset_j;

//     MatSetValues(mat, inorb, idxm, jnorb, idxn, v_petsc, ADD_VALUES);

//     if(v_free) {PetscCall(PetscFree(v_petsc));}

//     PetscFunctionReturn(PETSC_SUCCESS);
// };

PetscErrorCode SetLBlockR2K(Mat mat, DMAtomic &dm_atomic, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v, std::complex<double> factor){ // get the local idx-th edge and transform it, and add it on mat. 
    rsatb::backend::AtomData atom_i, atom_j;
    std::complex<double> phase;
    PetscInt offset_i, offset_j;
    PetscScalar *v_petsc = nullptr;
    PetscBool v_free;
    PetscInt idxm[inorb], idxn[jnorb];
    // rsatb::backend::get_edge_index_from_dm(dm, idx, &i, &j);

    // rsatb::backend::BondData bond;
    // rsatb::backend::get_attr_from_dm<rsatb::backend::BondData>(&dm, "edge", idx, &bond);
    cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
    phase *= factor;
    // get offset
    dm_atomic.get_offset_local(i, &offset_i);
    dm_atomic.get_offset_local(j, &offset_j);
    // std::cout << "Offset i,j " << offset_i << " " << offset_j << "\n";
    // adjust data format
    
    rsatb::backend::double2petscscalar(inorb*jnorb, v, &v_petsc, &v_free);
    // complex phase to PETscScalar
    PetscScalar phase_p;
    #if defined(PETSC_USE_COMPLEX)
    phase_p = PetscCMPLX(phase.real(),phase.imag());
    #else
    phase_p = static_cast<PetscScalar>(phase.real());
    if (phase.imag() != 0) throw std::runtime_error("The PETSc should be compiled with complex type when the system have periodicity.");
    #endif
    for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) v_petsc[k] = v_petsc[k]*phase_p;

    // for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) std::cout<<v_petsc[k];
    // std::cout<<phase_p<<"\n";

    // add the vcalue v to the mat.
    for (int im=0;im<inorb;im++) idxm[im] = im+offset_i;
    for (int in=0;in<jnorb;in++) idxn[in] = in+offset_j;

    MatSetValuesLocal(mat, inorb, idxm, jnorb, idxn, v_petsc, ADD_VALUES);

    if (v_free) {PetscCall(PetscFree(v_petsc));}

    PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode SetGBlockR2K(Mat mat, DMAtomic &dm_atomic, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v, std::complex<double> factor){ // get the local idx-th edge and transform it, and add it on mat. 
    rsatb::backend::AtomData atom_i, atom_j;
    std::complex<double> phase;
    PetscInt offset_i, offset_j;
    PetscScalar *v_petsc = nullptr;
    PetscBool v_free;
    PetscInt idxm[inorb], idxn[jnorb];

    cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
    phase *= factor;
    // get offset
    dm_atomic.get_offset_global(i, &offset_i);
    dm_atomic.get_offset_global(j, &offset_j);
    // std::cout << "Offset i,j " << offset_i << " " << offset_j << "\n";
    // adjust data format
    
    rsatb::backend::double2petscscalar(inorb*jnorb, v, &v_petsc, &v_free);
    // complex phase to PETscScalar
    PetscScalar phase_p;
    #if defined(PETSC_USE_COMPLEX)
    phase_p = PetscCMPLX(phase.real(),phase.imag());
    #else
    phase_p = static_cast<PetscScalar>(phase.real());
    if (phase.imag() != 0) throw std::runtime_error("The PETSc should be compiled with complex type when the system have periodicity.");
    #endif
    for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) v_petsc[k] = v_petsc[k]*phase_p;

    // for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) std::cout<<v_petsc[k];
    // std::cout<<phase_p<<"\n";

    // add the vcalue v to the mat.
    for (int im=0;im<inorb;im++) idxm[im] = im+offset_i;
    for (int in=0;in<jnorb;in++) idxn[in] = in+offset_j;

    MatSetValues(mat, inorb, idxm, jnorb, idxn, v_petsc, ADD_VALUES);

    if (v_free) {PetscCall(PetscFree(v_petsc));}

    PetscFunctionReturn(PETSC_SUCCESS);
};

// void setMatrixR2K(Mat mat, DM dm, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz){
//     /* Iterate over edge, node, and them:
//         1. Get the atom original idx, and the R vector;
//         2. Get the H(R)/S(R) block value from io interface;
//         3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
//         4. Assemble the matrix, do some small checks.
//     */

//    // Loop over node
//     PetscInt nstart, nend, estart, eend;
//     PetscBool ghost;
//     AtomData atomdata_i, atomdata_j;
//     BondData bonddata;
//     int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag, Rx, Ry, Rz;
//     bool found;
//     double *v=nullptr;

//     DMNetworkGetVertexRange(dm, &nstart, &nend);
//     for(int i=nstart;i<nend;++i){

//         DMNetworkIsGhostVertex(dm,i,&ghost);
//         if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
//             rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
//             // R vec is zero
//             // get block from org index atom_data.index
//             atmidx_i=static_cast<int>(atomdata_i.index);
//             flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_i,0,0,0, &m,&n,&v,&found);

//             // Set block at petsc index (i,i)
//             if(flag==0){
//                 if(found) SetBlockR2K(mat, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, v); /* Here we assume the matrix is initialized with zero value,
//                 So we do not handel the case when mat block is not founded.*/
//             }else{
//                 throw std::runtime_error("The h5file is corrupted !");
//             }

//             // Set block for petsc index (i,i) with different R
//             for(int iR=0; iR < atomdata_i.NR; ++iR){
//                 Rx=atomdata_i.R[3*iR]; Ry=atomdata_i.R[3*iR+1]; Rz=atomdata_i.R[3*iR+2];
//                 flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_i,Rx,Ry,Rz, &m,&n,&v,&found);

//                 if(flag==0){
//                     if(found) SetBlockR2K(mat, dm, i, i, inorb, inorb, Rx, Ry, Rz, Kx, Ky, Kz, v);
//                 }else{
//                     throw std::runtime_error("The h5file is corrupted !");
//             }
//             }
//         }
//     }

//     int i, j, eorb;
//     DMNetworkGetEdgeRange(dm, &estart, &eend);
//     for (int ie=estart; ie<eend; ie++)
//     {
//         /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
//         rsatb::backend::get_edge_index_from_dm(dm, ie, &i, &j);
//         rsatb::backend::get_attr_from_dm<BondData>(dm, "edge", ie, &bonddata, &eorb);

//         rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
//         rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", j, &atomdata_j, &jnorb);

//         if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
//             atmidx_i=static_cast<int>(atomdata_i.index);
//             atmidx_j=static_cast<int>(atomdata_j.index);

//             flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_j,bonddata.Rx,bonddata.Ry,bonddata.Rz, &m,&n,&v,&found);

//             if(flag==0){
                
//                 if(found) {
//                     SetBlockR2K(mat, dm, i, j, inorb, jnorb, bonddata.Rx,bonddata.Ry,bonddata.Rz, Kx, Ky, Kz, v);
//                 }
//             }else{
//                 throw std::runtime_error("The h5file is corrupted !");
//             }
//         }
//     }

//     MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
//     MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
   
// };

void setMatrixR2K(Mat mat, DMAtomic &dm_atomic, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz){
    /* Iterate over edge, node, and them:
        1. Get the atom original idx, and the R vector;
        2. Get the H(R)/S(R) block value from io interface;
        3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
        4. Assemble the matrix, do some small checks.
    */

   // Loop over node
    PetscInt nstart, nend, estart, eend;
    bool ghost;
    int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag, Rx, Ry, Rz;
    bool found;
    double *v=nullptr;
    
    dm_atomic.get_vertex_range_local(&nstart, &nend);
    for(int i=nstart;i<nend;++i){
        ghost = dm_atomic.atom_is_ghost(i);
        if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
            dm_atomic.get_atom_norb(i, &inorb);
            // R vec is zero
            // get block from org index atom_data.index
            dm_atomic.get_atom_id(i, &atmidx_i);
            flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_i,0,0,0, &m,&n,&v,&found);

            // Set block at petsc index (i,i)
            if(flag==0){
                if(found) SetLBlockR2K(mat, dm_atomic, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, v); /* Here we assume the matrix is initialized with zero value,
                So we do not handel the case when mat block is not founded.*/
            }else{
                throw std::runtime_error("The h5file is corrupted !");
            }
        }
    }

    int i, j, eorb;
    dm_atomic.get_edge_range_local(&estart, &eend);
    for (int ie=estart; ie<eend; ie++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        dm_atomic.get_connected_atoms(ie, &i, &j);
        dm_atomic.get_atom_norb(i, &inorb); dm_atomic.get_atom_norb(j, &jnorb);

        if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
            dm_atomic.get_atom_id(i, &atmidx_i);
            dm_atomic.get_atom_id(j, &atmidx_j);
            dm_atomic.get_edge_shift_vec(ie, &Rx, &Ry, &Rz);

            flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_j, Rx, Ry, Rz, &m,&n,&v,&found);

            if(flag==0){
                
                if(found) {
                    SetLBlockR2K(mat, dm_atomic, i, j, inorb, jnorb, Rx, Ry, Rz, Kx, Ky, Kz, v);
                }
            }else{
                throw std::runtime_error("The h5file is corrupted !");
            }
        }
    }

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
   
};

/*
    Matrix formation from 2 centre integrals
*/
// void setMatrixR2K(
//     Mat S, DM dm, const double* cell, int ntype, 
//     const std::string* file_orb0, const std::string& orbital_dir, 
//     double Kx, double Ky, double Kz, Mat Rx, Mat Ry, Mat Rz
// ) {
//     /* Iterate over edge, node, and them:
//         1. Get the atom original idx, and the R vector;
//         2. Get the H(R)/S(R) block value from io interface;
//         3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
//         4. Assemble the matrix, do some small checks.
//     */

//     MPI_Comm comm;
//     PetscObjectGetComm((PetscObject)dm, &comm);
//    // Loop over node
//     PetscInt nstart, nend, estart, eend;
//     PetscBool ghost;
//     AtomData atomdata_i, atomdata_j;
//     BondData bonddata;

//     TwoCenterBundle tcbd;
//     container::Tensor orb_map{container::DataType::DT_INT, container::TensorShape({0})};

//     tcbd.build_orb(ntype, file_orb0, orbital_dir, comm);
//     tcbd.tabulate();

//     // build the orbital mapping table: [itype][iorb] -> (l, zeta, m)
//     orb_map.resize({ntype, tcbd.orb_->nphi_max(),3});
//     for(int itype=0; itype<ntype; ++itype){
//         int iphi = 0;
//         for(int l=0; l<=tcbd.orb_->lmax(itype); ++l){
//             for(int izeta=0; izeta<tcbd.orb_->nzeta(itype, l); ++izeta){
//                 for(int m=-l; m<=l; ++m){
//                     orb_map.get_value<int>(itype, iphi, 0) = l;
//                     orb_map.get_value<int>(itype, iphi, 1) = izeta;
//                     orb_map.get_value<int>(itype, iphi, 2) = m;
//                     iphi++;
//                     // std::cout << iphi << " " << l << " " << m << " " << izeta << std::endl;
//                 }
//             }
//         }
//     }

//     int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag, Rx_, Ry_, Rz_;
//     int itype, il, izeta, im, jtype, jl, jzeta, jm, m_phase;
//     double rx, ry, rz;

//     ModuleBase::Vector3<double> vR(0.0, 0.0, 0.0);
//     ModuleBase::Vector3<double> R2(0.0, 0.0, 0.0);

//     DMNetworkGetVertexRange(dm, &nstart, &nend);
//     for(int i=nstart;i<nend;++i){

//         DMNetworkIsGhostVertex(dm,i,&ghost);
//         if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
//             rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
//             double vS[inorb*inorb], vX[inorb*inorb], vY[inorb*inorb], vZ[inorb*inorb];
//             // R vec is zero
//             // get block from org index atom_data.index
            
//             itype = static_cast<int>(atomdata_i.type);
//             jtype = static_cast<int>(atomdata_i.type);
            
//             vR.set(0.0, 0.0, 0.0);
//             R2.set(atomdata_i.x, atomdata_i.y, atomdata_i.z);
//             R2 *= 1.8897259886;

//             for(int io=0; io<inorb; io++){
//                 il=orb_map.get_value<int>(itype, io, 0);
//                 izeta=orb_map.get_value<int>(itype, io, 1);
//                 im=orb_map.get_value<int>(itype, io, 2);
//                 for(int jo=io; jo<inorb; jo++){
//                     jl=orb_map.get_value<int>(jtype, jo, 0);
//                     jzeta=orb_map.get_value<int>(jtype, jo, 1);
//                     jm=orb_map.get_value<int>(jtype, jo, 2);

//                     // std::cout << "Before Calculation ~~ itype: " << itype << " il: " << il << " izeta: " << izeta << " im: " << im << " jl: " << jl << " jzeta: " << jzeta << " jm: " << jm << " io: " << io << " jo: " << jo << std::endl;

//                     tcbd.overlap_orb->calculate(
//                         itype, il, izeta, im,
//                         jtype, jl, jzeta, jm,
//                         vR, R2,
//                         // output
//                         vS + inorb*io+jo, // S
//                         (Rx) ? vX + inorb*io+jo : nullptr, // Rx
//                         (Ry) ? vY + inorb*io+jo : nullptr, // Ry
//                         (Rz) ? vZ + inorb*io+jo : nullptr // Rz
//                     );

//                     /* remove phase factor */
//                     m_phase = (im+jm) % 2 == 0 ? 1 : -1;
//                     vS[inorb*io+jo] *= m_phase;

//                     // std::cout << "Calculating ~~ " << *(vS + inorb*io+jo) << std::endl;

//                     // copy the value to its complex conjugate (but here it should be real since we use real spherical harmonics)
//                     vS[inorb*jo+io] = vS[inorb*io+jo];
//                     if (Rx) { 
//                         vX[inorb*io+jo] *= m_phase;
//                         vX[inorb*io+jo] /= 1.8897259886;
//                         vX[inorb*jo+io] = vX[inorb*io+jo]; 
//                     }
//                     if (Ry) { 
//                         vY[inorb*io+jo] *= m_phase;
//                         vY[inorb*io+jo] /= 1.8897259886;
//                         vY[inorb*jo+io] = vY[inorb*io+jo]; 
//                     }
//                     if (Rz) { 
//                         vZ[inorb*io+jo] *= m_phase;
//                         vZ[inorb*io+jo] /= 1.8897259886;
//                         vZ[inorb*jo+io] = vZ[inorb*io+jo]; 
//                     }

//                     // if (il==0 && jl==0 && izeta==jzeta && im==jm && std::abs(vR[0]) <= 1e-14 && std::abs(vR[1]) <= 1e-14 && std::abs(vR[2]) <= 1e-14) { std::cout << *(vX + inorb*io+jo) << " " << R2[0] << " " << *(vS + inorb*io+jo) << std::endl;}
//                     // std::cout << "Copy Value ~~" << std::endl;
//                 }
//             }

//             SetBlockR2K(S, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vS);

//             if (Rx) { SetBlockR2K(Rx, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vX); }
//             if (Ry) { SetBlockR2K(Ry, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vY); }
//             if (Rz) { SetBlockR2K(Rz, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vZ); }

//             // Set block for petsc index (i,i) with different R
//             for(int iR=0; iR < atomdata_i.NR; ++iR){
//                 Rx_=atomdata_i.R[3*iR]; Ry_=atomdata_i.R[3*iR+1]; Rz_=atomdata_i.R[3*iR+2]; // cell shift vec for onsite
//                 rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6];
//                 ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
//                 rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
//                 vR.set(rx, ry, rz);
//                 R2.set(rx+atomdata_i.x, ry+atomdata_i.y, rz+atomdata_i.z);
//                 vR *= 1.8897259886; // to bohr
//                 R2 *= 1.8897259886;

//                 for(int io=0; io<inorb; io++){
//                     il=orb_map.get_value<int>(itype, io, 0);
//                     izeta=orb_map.get_value<int>(itype, io, 1);
//                     im=orb_map.get_value<int>(itype, io, 2);
//                     for(int jo=0; jo<inorb; jo++){
//                         jl=orb_map.get_value<int>(jtype, jo, 0);
//                         jzeta=orb_map.get_value<int>(jtype, jo, 1);
//                         jm=orb_map.get_value<int>(jtype, jo, 2);

//                         tcbd.overlap_orb->calculate(
//                             itype, il, izeta, im,
//                             jtype, jl, jzeta, jm,
//                             vR, R2,
//                             // output
//                             vS + inorb*io+jo, // S
//                             (Rx) ? vX + inorb*io+jo : nullptr, // Rx
//                             (Ry) ? vY + inorb*io+jo : nullptr, // Ry
//                             (Rz) ? vZ + inorb*io+jo : nullptr // Rz
//                         );

//                         m_phase = (im+jm) % 2 == 0 ? 1 : -1;
//                         vS[inorb*io+jo] *= m_phase;

//                         // copy the value to its complex conjugate
//                         // vS[inorb*jo+io] = vS[inorb*io+jo];
//                         if (Rx) { 
//                             vX[inorb*io+jo] *= m_phase;
//                             vX[inorb*io+jo] /= 1.8897259886;
//                             // vX[inorb*jo+io] = vX[inorb*io+jo]; 
//                         }
//                         if (Ry) { 
//                             vY[inorb*io+jo] *= m_phase;
//                             vY[inorb*io+jo] /= 1.8897259886;
//                             // vY[inorb*jo+io] = vY[inorb*io+jo]; 
//                         }
//                         if (Rz) { 
//                             vZ[inorb*io+jo] *= m_phase;
//                             vZ[inorb*io+jo] /= 1.8897259886;
//                             // vZ[inorb*jo+io] = vZ[inorb*io+jo]; 
//                         }
//                     }
//                 }

//                 // std::cout << i << inorb << std::endl;
//                 // std::cout << Rx_ << " " << Ry_ << " " << Rz_ << std::endl;
//                 // for (int i=0; i<inorb*inorb; ++i) std::cout << vS[i] << " ";
//                 // std::cout << std::endl;

//                 SetBlockR2K(S, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
//                 if (Rx) { SetBlockR2K(Rx, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX); }
//                 if (Ry) { SetBlockR2K(Ry, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY); }
//                 if (Rz) { SetBlockR2K(Rz, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ); }
//             }
//         }
//     }

//     int i, j, eorb;
//     DMNetworkGetEdgeRange(dm, &estart, &eend);
//     for (int ie=estart; ie<eend; ie++)
//     {
//         /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
//         rsatb::backend::get_edge_index_from_dm(dm, ie, &i, &j);
//         // std::cout << i << " " << j << std::endl;
//         rsatb::backend::get_attr_from_dm<BondData>(dm, "edge", ie, &bonddata, &eorb);
//         rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
//         rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", j, &atomdata_j, &jnorb);

//         double vS[inorb*jnorb], vX[inorb*jnorb], vY[inorb*jnorb], vZ[inorb*jnorb];

//         if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
        
//             itype = static_cast<int>(atomdata_i.type);
//             jtype = static_cast<int>(atomdata_j.type);

//             Rx_=bonddata.Rx; Ry_=bonddata.Ry; Rz_=bonddata.Rz; // cell shift for hopping blocks
//             rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6]; // cell shift in distance unit
//             ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
//             rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
//             R2.set(rx+atomdata_j.x, ry+atomdata_j.y, rz+atomdata_j.z);
//             vR.set(R2.x-atomdata_i.x, R2.y-atomdata_i.y, R2.z-atomdata_i.z);
//             vR *= 1.8897259886; // to bohr
//             R2 *= 1.8897259886;

//             for(int io=0; io<inorb; io++){
//                 il=orb_map.get_value<int>(itype, io, 0);
//                 izeta=orb_map.get_value<int>(itype, io, 1);
//                 im=orb_map.get_value<int>(itype, io, 2);
//                 for(int jo=0; jo<jnorb; jo++){
//                     jl=orb_map.get_value<int>(jtype, jo, 0);
//                     jzeta=orb_map.get_value<int>(jtype, jo, 1);
//                     jm=orb_map.get_value<int>(jtype, jo, 2);

//                     tcbd.overlap_orb->calculate(
//                         itype, il, izeta, im,
//                         jtype, jl, jzeta, jm,
//                         vR, R2,
//                         // output
//                         vS + jnorb*io+jo, // S
//                         (Rx) ? vX + jnorb*io+jo : nullptr, // Rx
//                         (Ry) ? vY + jnorb*io+jo : nullptr, // Ry
//                         (Rz) ? vZ + jnorb*io+jo : nullptr // Rz
//                     );
//                     m_phase = (im+jm) % 2 == 0 ? 1 : -1;
//                     vS[jnorb*io+jo] *= m_phase;

//                     if (Rx) { 
//                         vX[jnorb*io+jo] *= m_phase;
//                         vX[jnorb*io+jo] /= 1.8897259886;
//                     }
//                     if (Ry) { 
//                         vY[jnorb*io+jo] *= m_phase;
//                         vY[jnorb*io+jo] /= 1.8897259886;
//                     }
//                     if (Rz) { 
//                         vZ[jnorb*io+jo] *= m_phase;
//                         vZ[jnorb*io+jo] /= 1.8897259886;
//                     }
//                 }
//             }
//             SetBlockR2K(S, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
//             if (Rx) { SetBlockR2K(Rx, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX); }
//             if (Ry) { SetBlockR2K(Ry, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY); }
//             if (Rz) { SetBlockR2K(Rz, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ); }
//         }
//     }

//     MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
//     MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
//     // MatView(S, PETSC_VIEWER_STDOUT_WORLD);
//     if (Rx){
//         MatAssemblyBegin(Rx, MAT_FINAL_ASSEMBLY);
//         MatAssemblyEnd(Rx, MAT_FINAL_ASSEMBLY);
//     }
//     if (Ry){
//         MatAssemblyBegin(Ry, MAT_FINAL_ASSEMBLY);
//         MatAssemblyEnd(Ry, MAT_FINAL_ASSEMBLY);
//     }
//     if (Rz){
//         MatAssemblyBegin(Rz, MAT_FINAL_ASSEMBLY);
//         MatAssemblyEnd(Rz, MAT_FINAL_ASSEMBLY);   
//     }
//     orb_map.resize({0});
// };

void setMatrixR2K(
    Mat S, DMAtomic &dm_atomic, const double* cell, int ntype, 
    const std::string* file_orb0, const std::string& orbital_dir, 
    double Kx, double Ky, double Kz, Mat Rx, Mat Ry, Mat Rz
) {
    /* Iterate over edge, node, and them:
        1. Get the atom original idx, and the R vector;
        2. Get the H(R)/S(R) block value from io interface;
        3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
        4. Assemble the matrix, do some small checks.
    */

    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)(dm_atomic.dm()), &comm);
   // Loop over node
    PetscInt nstart, nend, estart, eend;
    bool ghost;

    TwoCenterBundle tcbd;
    container::Tensor orb_map{container::DataType::DT_INT, container::TensorShape({0})};

    tcbd.build_orb(ntype, file_orb0, orbital_dir, comm);
    tcbd.tabulate();

    // build the orbital mapping table: [itype][iorb] -> (l, zeta, m)
    orb_map.resize({ntype, tcbd.orb_->nphi_max(),3});
    for(int itype=0; itype<ntype; ++itype){
        int iphi = 0;
        for(int l=0; l<=tcbd.orb_->lmax(itype); ++l){
            for(int izeta=0; izeta<tcbd.orb_->nzeta(itype, l); ++izeta){
                for(int m=-l; m<=l; ++m){
                    orb_map.get_value<int>(itype, iphi, 0) = l;
                    orb_map.get_value<int>(itype, iphi, 1) = izeta;
                    orb_map.get_value<int>(itype, iphi, 2) = m;
                    iphi++;
                    // std::cout << iphi << " " << l << " " << m << " " << izeta << std::endl;
                }
            }
        }
    }

    int inorb, jnorb, m, n, flag, Rx_, Ry_, Rz_;
    int itype, il, izeta, im, jtype, jl, jzeta, jm, m_phase;
    double rx, ry, rz, x, y, z;

    ModuleBase::Vector3<double> vR(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> R2(0.0, 0.0, 0.0);

    dm_atomic.get_vertex_range_local(&nstart, &nend);
    for(int i=nstart;i<nend;++i){
        ghost = dm_atomic.atom_is_ghost(i);
        if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
            dm_atomic.get_atom_norb(i, &inorb);
            double vS[inorb*inorb], vX[inorb*inorb], vY[inorb*inorb], vZ[inorb*inorb];
            // R vec is zero
            // get block from org index atom_data.index
            dm_atomic.get_atom_type(i, &itype);
            
            jtype = itype;

            vR.set(0.0, 0.0, 0.0);
            dm_atomic.get_pos(i, &x, &y, &z);
            R2.set(x,y,z);
            R2 *= 1.8897259886;

            for(int io=0; io<inorb; io++){
                il=orb_map.get_value<int>(itype, io, 0);
                izeta=orb_map.get_value<int>(itype, io, 1);
                im=orb_map.get_value<int>(itype, io, 2);
                for(int jo=io; jo<inorb; jo++){
                    jl=orb_map.get_value<int>(jtype, jo, 0);
                    jzeta=orb_map.get_value<int>(jtype, jo, 1);
                    jm=orb_map.get_value<int>(jtype, jo, 2);

                    // std::cout << "Before Calculation ~~ itype: " << itype << " il: " << il << " izeta: " << izeta << " im: " << im << " jl: " << jl << " jzeta: " << jzeta << " jm: " << jm << " io: " << io << " jo: " << jo << std::endl;
                    tcbd.overlap_orb->calculate(
                        itype, il, izeta, im,
                        jtype, jl, jzeta, jm,
                        vR, R2,
                        // output
                        vS + inorb*io+jo, // S
                        (Rx) ? vX + inorb*io+jo : nullptr, // Rx
                        (Ry) ? vY + inorb*io+jo : nullptr, // Ry
                        (Rz) ? vZ + inorb*io+jo : nullptr // Rz
                    );

                    /* remove phase factor */
                    m_phase = (im+jm) % 2 == 0 ? 1 : -1;
                    vS[inorb*io+jo] *= m_phase;

                    // std::cout << "Calculating ~~ " << *(vS + inorb*io+jo) << std::endl;

                    // copy the value to its complex conjugate (but here it should be real since we use real spherical harmonics)
                    vS[inorb*jo+io] = vS[inorb*io+jo];
                    if (Rx) { 
                        vX[inorb*io+jo] *= m_phase;
                        vX[inorb*io+jo] /= 1.8897259886;
                        vX[inorb*jo+io] = vX[inorb*io+jo]; 
                    }
                    if (Ry) { 
                        vY[inorb*io+jo] *= m_phase;
                        vY[inorb*io+jo] /= 1.8897259886;
                        vY[inorb*jo+io] = vY[inorb*io+jo]; 
                    }
                    if (Rz) { 
                        vZ[inorb*io+jo] *= m_phase;
                        vZ[inorb*io+jo] /= 1.8897259886;
                        vZ[inorb*jo+io] = vZ[inorb*io+jo]; 
                    }

                    // if (il==0 && jl==0 && izeta==jzeta && im==jm && std::abs(vR[0]) <= 1e-14 && std::abs(vR[1]) <= 1e-14 && std::abs(vR[2]) <= 1e-14) { std::cout << *(vX + inorb*io+jo) << " " << R2[0] << " " << *(vS + inorb*io+jo) << std::endl;}
                    // std::cout << "Copy Value ~~" << std::endl;
                }
            }

            SetLBlockR2K(S, dm_atomic, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vS);

            if (Rx) { SetLBlockR2K(Rx, dm_atomic, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vX); }
            if (Ry) { SetLBlockR2K(Ry, dm_atomic, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vY); }
            if (Rz) { SetLBlockR2K(Rz, dm_atomic, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vZ); }

        }
    }

    int i, j, eorb;
    dm_atomic.get_edge_range_local(&estart, &eend);
    for (int ie=estart; ie<eend; ie++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        dm_atomic.get_connected_atoms(ie, &i, &j);
        dm_atomic.get_atom_norb(i, &inorb); dm_atomic.get_atom_norb(j, &jnorb);
        // std::cout << i << " " << j << std::endl;
        double vS[inorb*jnorb], vX[inorb*jnorb], vY[inorb*jnorb], vZ[inorb*jnorb];

        if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
            dm_atomic.get_atom_type(i, &itype);
            dm_atomic.get_atom_type(j, &jtype);

            dm_atomic.get_edge_shift_vec(ie, &Rx_, &Ry_, &Rz_);
            rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6]; // cell shift in distance unit
            ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
            rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
            dm_atomic.get_pos(j, &x, &y, &z);
            R2.set(rx+x, ry+y, rz+z); // Rj
            dm_atomic.get_pos(i, &x, &y, &z);
            vR.set(R2.x-x, R2.y-y, R2.z-z); // Rj - Ri
            vR *= 1.8897259886; // to bohr
            R2 *= 1.8897259886;

            // std::cout << i << " " << j << " " << Rx_ << " " << Ry_ << " " << Rz_ << std::endl;
            // std::cout << vR.x << " " << vR.y << " " << vR.z << std::endl;
            // std::cout << R2.x << " " << R2.y << " " << R2.z << std::endl;

            for(int io=0; io<inorb; io++){
                il=orb_map.get_value<int>(itype, io, 0);
                izeta=orb_map.get_value<int>(itype, io, 1);
                im=orb_map.get_value<int>(itype, io, 2);
                for(int jo=0; jo<jnorb; jo++){
                    jl=orb_map.get_value<int>(jtype, jo, 0);
                    jzeta=orb_map.get_value<int>(jtype, jo, 1);
                    jm=orb_map.get_value<int>(jtype, jo, 2);

                    tcbd.overlap_orb->calculate(
                        itype, il, izeta, im,
                        jtype, jl, jzeta, jm,
                        vR, R2,
                        // output
                        vS + jnorb*io+jo, // S
                        (Rx) ? vX + jnorb*io+jo : nullptr, // Rx
                        (Ry) ? vY + jnorb*io+jo : nullptr, // Ry
                        (Rz) ? vZ + jnorb*io+jo : nullptr // Rz
                    );
                    m_phase = (im+jm) % 2 == 0 ? 1 : -1;
                    vS[jnorb*io+jo] *= m_phase;

                    if (Rx) { 
                        vX[jnorb*io+jo] *= m_phase;
                        vX[jnorb*io+jo] /= 1.8897259886;
                        // std::cout << vX[jnorb*io+jo] << ", ";
                        // if (jo == inorb-1) {
                        //     std::cout << std::endl;
                        // }
                    }
                    if (Ry) { 
                        vY[jnorb*io+jo] *= m_phase;
                        vY[jnorb*io+jo] /= 1.8897259886;
                    }
                    if (Rz) { 
                        vZ[jnorb*io+jo] *= m_phase;
                        vZ[jnorb*io+jo] /= 1.8897259886;
                    }
                }
            }
            SetLBlockR2K(S, dm_atomic, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
            if (Rx) { SetLBlockR2K(Rx, dm_atomic, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX); }
            if (Ry) { SetLBlockR2K(Ry, dm_atomic, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY); }
            if (Rz) { SetLBlockR2K(Rz, dm_atomic, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ); }
        }
    }

    MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
    // MatView(S, PETSC_VIEWER_STDOUT_WORLD);
    if (Rx){
        MatAssemblyBegin(Rx, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Rx, MAT_FINAL_ASSEMBLY);
    }
    if (Ry){
        MatAssemblyBegin(Ry, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Ry, MAT_FINAL_ASSEMBLY);
    }
    if (Rz){
        MatAssemblyBegin(Rz, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Rz, MAT_FINAL_ASSEMBLY);   
    }

    orb_map.resize({0});
};

void setMatrixR2K(Mat S, rsatb::backend::DMAtomic &dm_atomic, const double* cell, int ntype, 
                const std::vector<std::string> orb_names, const std::string& orb_dir, const std::vector<std::string> psd_names, const std::string& psd_dir, 
                double Kx, double Ky, double Kz, Mat Vx=nullptr, Mat Vy=nullptr, Mat Vz=nullptr, const std::string& log_file)
{
        /* Iterate over edge, node, and them:
        1. Form the S and grad_S block, grad_S can be directly added onto the V(x/y/z).
        2. form the velocity
            i. loop over k
            ii. compute the orb beta integrals
            iii. compute v_comm, add to V(x/y/z)
    */

    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)(dm_atomic.dm()), &comm);
   // Loop over node
    PetscInt nstart, nend, estart, eend;
    bool ghost;

    Op2c oc(
        ntype, 1, 0, 
        orb_dir, orb_names, psd_dir, psd_names,
        comm, logfile
    );

    tcbd.build_orb(ntype, file_orb0, orbital_dir, comm);
    tcbd.tabulate();

    int i, j, eorb, idx, jdx, edx, ie; // idx, jdx: local index, edx: local edge index
    int inorb, jnorb, m, n, flag, Rx_, Ry_, Rz_;
    int itype, jtype, m_phase;
    double rx, ry, rz, x, y, z;
    std::vector<double>* vXp=nullptr, vYp=nullptr, vZp=nullptr;

    ModuleBase::Vector3<double> vR(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> R2(0.0, 0.0, 0.0);

    dm_atomic.get_vertex_range_local(&nstart, &nend);
    for(idx=nstart;idx<nend;++idx){
        ghost = dm_atomic.atom_is_ghost(idx);
        if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
            dm_atomic.get_atom_norb(idx, &inorb);
            std::vecctor<double> vS(inorb*inorb), vX, vY, vZ;
            if (Vx) {
                vX.resize(inorb*inorb);
                vXp = &vX;
            }
            if (Vy) {
                vY.resize(inorb*inorb);
                vYp = &vY;
            }
            if (Vz) {
                vZ.resize(inorb*inorb);
                vZp = &vZ;
            }
            
            // R vec is zero
            // get block from org index atom_data.index
            dm_atomic.get_atom_type(idx, &itype);
            
            jtype = itype;

            vR.set(0.0, 0.0, 0.0);
            dm_atomic.get_pos(idx, &x, &y, &z);
            R2.set(x,y,z);
            R2 *= 1.8897259886;

            oc.overlap(itype, jtype, vR, false, vS, vXp, vYp, vZp);

            SetLBlockR2K(S, dm_atomic, idx, idx, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vS.data());

            if (Vx) { SetLBlockR2K(Vx, dm_atomic, idx, idx, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vX.data(), IMAG_UNIT); }
            if (Vy) { SetLBlockR2K(Vy, dm_atomic, idx, idx, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vY.data(), IMAG_UNIT); }
            if (Vz) { SetLBlockR2K(Vz, dm_atomic, idx, idx, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vZ.data(), IMAG_UNIT); }
        }
    }

    dm_atomic.get_edge_range_local(&estart, &eend);
    for (int edx=estart; edx<eend; edx++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        dm_atomic.get_connected_atoms(edx, &idx, &jdx);
        dm_atomic.get_atom_norb(idx, &inorb); dm_atomic.get_atom_norb(jdx, &jnorb);
        // std::cout << i << " " << j << std::endl;
        double vS[inorb*jnorb], vX[inorb*jnorb], vY[inorb*jnorb], vZ[inorb*jnorb];

        if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
            dm_atomic.get_atom_type(idx, &itype);
            dm_atomic.get_atom_type(jdx, &jtype);

            std::vecctor<double> vS(inorb*jnorb), vX, vY, vZ;
            if (Vx) {
                vX.resize(inorb*jnorb);
                vXp = &vX;
            }
            if (Vy) {
                vY.resize(inorb*jnorb);
                vYp = &vY;
            }
            if (Vz) {
                vZ.resize(inorb*jnorb);
                vZp = &vZ;
            }

            dm_atomic.get_edge_shift_vec(edx, &Rx_, &Ry_, &Rz_);
            rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6]; // cell shift in distance unit
            ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
            rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
            dm_atomic.get_pos(jdx, &x, &y, &z);
            R2.set(rx+x, ry+y, rz+z); // Rj
            dm_atomic.get_pos(idx, &x, &y, &z);
            vR.set(R2.x-x, R2.y-y, R2.z-z); // Rj - Ri
            vR *= 1.8897259886; // to bohr
            R2 *= 1.8897259886;

            oc.overlap(itype, jtype, vR, false, vS, vXp, vYp, vZp);

            SetLBlockR2K(S, dm_atomic, idx, jdx, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
            if (Vx) { SetLBlockR2K(Vx, dm_atomic, idx, jdx, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX.data(), IMAG_UNIT); }
            if (Vy) { SetLBlockR2K(Vy, dm_atomic, idx, jdx, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY.data(), IMAG_UNIT); }
            if (Vz) { SetLBlockR2K(Vz, dm_atomic, idx, jdx, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ.data(), IMAG_UNIT); }
        }
    }

    MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);

    // step 2, form vcomm

    if (Vx || Vy || Vz) {
        std::vector<size_t> conn_type;
        std::vector<int> conn_idx;
        std::vector<ModuleBase::Vector3> Ri, Rshift;
        ModuleBase::Vector3 Rk;
        std::vector<ModuleBase::matrix> ob, oxb, oyb, ozb;
        int tp, nei, ie, Rkix, Rkiy, Rkiz, Rkjx, Rkjy, Rkjz;
        int ktype, tmp;
        double r, cutoff, err;

        for(int kdx=nstart;kdx<nend;++kdx){
            dm_atomic.get_pos(kdx, &rx, &ry, &rz;)
            Rk.set(rx, ry, rz);
            nei = dm_atomic.icomm[kdx].size();
            conn_type.resize(nei); Ri.resize(nei); ob.resize(nei); oyb.resize(nei); ozb.resize(nei), Rshift.resize(nei), conn_idx.resize(nei);
            for(int in=0; in<nei; ++in){
                ie = dm_atomic.iconn[k][in];
                dm_atomic.get_connected_atoms(ie, &tmp, &idx);
                dm_atomic.get_edge_vec(ie, &rx, &ry, &rz;) // Ri - Rk
                dm_atomic.get_edge_shift_vec(idx, &Rx_, &Ry_, &Rz_); // int Rx, Ry, Rz

                conn[in] = dm_atomic.get_atom_type(idx, &tp);
                conn_idx[in] = idx;
                Ri[in].set(rx+Rk.x, ry+Rk.y, rz+Rk.z);
                Rshift[in].set(Rx_, Ry_, Rz_);
                Ri[in] *= 1.8897259886;
                Rshift[in] *= 1.8897259886;
            }

            dm_atomic.get_atom_type(k, &ktype);

            oc.orb_r_beta(conn, ktype, Ri, Rk, false,
                ob, oxb, oyb, ozb
            );
            
            for(int in=0; in<conn.size(); ++in) {
                itype = conn[in];
                inorb = dm_atomic.type_norb()[itype];
                idx = conn_idx[in];
                for(int jn=0; jn<conn.size(); ++jn){
                    r = (Ri[jn] - Ri[in]).norm(); // can compare with the cutoff to decide whether this edge is needed.
                    cutoff = oc.tcdb.orb_.rcut_max(itype) + oc.tcdb.orb_.rcut_max(jtype)
                    if( r > cutoff) {
                        err = (ob[in].absmax() + obx[in].absmax() + oby[in].absmax() + obz[in].absmax()) / 3.0
                        if ( err > 1e-7 ) {
                            std::cout << "There are large values throwed away with magnitute when computing velocity operator: " << err << " ." std::endl;
                        }
                        continue;
                    }
                    jtype = conn[jn];
                    jnorb = dm_atomic.type_norb()[jtype];
                    std::vecctor<std::complex<double>> vX(inorb*jnorb, 0), vY(inorb*jnorb, 0), vZ(inorb*jnorb, 0);
                    
                    oc.ncomm_IKJ(
                        itype, in, ktype, jtype, in,
                        ob, oxb, oyb, ozb,
                        vX, vY, vZ
                    );

                    if (Vx) { SetGBlockR2K(Vx, dm_atomic, idx, jdx, inorb, jnorb, Rshift[jdx].x-Rshift[idx].x, Rshift[jdx].y-Rshift[idx].y, Rshift[jdx].z-Rshift[idx].z, Kx, Ky, Kz, vX.data(), 27.211386245981); }
                    if (Vy) { SetGBlockR2K(Vy, dm_atomic, idx, jdx, inorb, jnorb, Rshift[jdx].x-Rshift[idx].x, Rshift[jdx].y-Rshift[idx].y, Rshift[jdx].z-Rshift[idx].z, Kx, Ky, Kz, vY.data(), 27.211386245981); }
                    if (Vz) { SetGBlockR2K(Vz, dm_atomic, idx, jdx, inorb, jnorb, Rshift[jdx].x-Rshift[idx].x, Rshift[jdx].y-Rshift[idx].y, Rshift[jdx].z-Rshift[idx].z, Kx, Ky, Kz, vZ.data(), 27.211386245981); }
                }
            } // i-k-j loop
        } // k loop
    }

    if (Vx){
        MatAssemblyBegin(Vx, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Vx, MAT_FINAL_ASSEMBLY);
    }
    if (Vy){
        MatAssemblyBegin(Vy, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Vy, MAT_FINAL_ASSEMBLY);
    }
    if (Vz){
        MatAssemblyBegin(Vz, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Vz, MAT_FINAL_ASSEMBLY);   
    }
};

void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase){
    *phase = std::complex<double>(0.0, -2.0 * M_PI * ((Rx * Kx) + (Ry * Ky) + (Rz * Kz)));
    *phase = std::exp(*phase);
};

}