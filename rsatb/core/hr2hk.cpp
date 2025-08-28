#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"
#include "hr2hk.hpp"
#include <math.h>
#include <ATen/tensor.h>

using namespace rsatb::backend;
using namespace rsatb::io;
namespace rsatb{

PetscErrorCode SetBlockR2K(Mat mat, DM dm, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v){ // get the local idx-th edge and transform it, and add it on mat. 
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
    // get offset
    DMNetworkGetGlobalVecOffset(dm, i, 0, &offset_i); /* offset is the first point position of the Dof of vertices i in the local numbering*/
    DMNetworkGetGlobalVecOffset(dm, j, 0, &offset_j);
    // std::cout << "Offset i,j " << offset_i << " " << offset_j << "\n";
    // adjust data format
    
    rsatb::backend::double2petscscalar(inorb*jnorb, v, &v_petsc, &v_free);
    // complex phase to PETscScalar
    PetscScalar phase_p;
    #if defined(PETSC_USE_COMPLEX)
    phase_p = PetscCMPLX(phase.real(),phase.imag());
    #else
    phase_p = reinterpret_cast<PETSC_SCALAR>(phase.real());
    if(phase.imag != 0) throw std::runtime_error("The PETSc should be compiled with complex type when the system have periodicity.");
    #endif
    for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) v_petsc[k] = v_petsc[k]*phase_p;

    // for (PetscInt k=0; k<(PetscInt)inorb*jnorb; ++k) std::cout<<v_petsc[k];
    // std::cout<<phase_p<<"\n";

    // add the vcalue v to the mat.
    for(int im=0;im<inorb;im++) idxm[im] = im+offset_i;
    for(int in=0;in<jnorb;in++) idxn[in] = in+offset_j;

    MatSetValues(mat, inorb, idxm, jnorb, idxn, v_petsc, ADD_VALUES);

    if(v_free) {PetscCall(PetscFree(v_petsc));}

    PetscFunctionReturn(PETSC_SUCCESS);
};

void setMatrixR2K(Mat mat, DM dm, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz){
    /* Iterate over edge, node, and them:
        1. Get the atom original idx, and the R vector;
        2. Get the H(R)/S(R) block value from io interface;
        3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
        4. Assemble the matrix, do some small checks.
    */

   // Loop over node
    PetscInt nstart, nend, estart, eend;
    PetscBool ghost;
    AtomData atomdata_i, atomdata_j;
    BondData bonddata;
    int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag, Rx, Ry, Rz;
    bool found;
    double *v=nullptr;

    DMNetworkGetVertexRange(dm, &nstart, &nend);
    for(int i=nstart;i<nend;++i){

        DMNetworkIsGhostVertex(dm,i,&ghost);
        if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
            rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
            // R vec is zero
            // get block from org index atom_data.index
            atmidx_i=static_cast<int>(atomdata_i.index);
            flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_i,0,0,0, &m,&n,&v,&found);

            // Set block at petsc index (i,i)
            if(flag==0){
                if(found) SetBlockR2K(mat, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, v); /* Here we assume the matrix is initialized with zero value,
                So we do not handel the case when mat block is not founded.*/
            }else{
                throw std::runtime_error("The h5file is corrupted !");
            }

            // Set block for petsc index (i,i) with different R
            for(int iR=0; iR < atomdata_i.NR; ++iR){
                Rx=atomdata_i.R[3*iR]; Ry=atomdata_i.R[3*iR+1]; Rz=atomdata_i.R[3*iR+2];
                flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_i,Rx,Ry,Rz, &m,&n,&v,&found);

                if(flag==0){
                    if(found) SetBlockR2K(mat, dm, i, i, inorb, inorb, Rx, Ry, Rz, Kx, Ky, Kz, v);
                }else{
                    throw std::runtime_error("The h5file is corrupted !");
            }
            }
        }
    }

    int i, j, eorb;
    DMNetworkGetEdgeRange(dm, &estart, &eend);
    for (int ie=estart; ie<eend; ie++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        rsatb::backend::get_edge_index_from_dm(dm, ie, &i, &j);
        rsatb::backend::get_attr_from_dm<BondData>(dm, "edge", ie, &bonddata, &eorb);

        rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
        rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", j, &atomdata_j, &jnorb);

        if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.
            atmidx_i=static_cast<int>(atomdata_i.index);
            atmidx_j=static_cast<int>(atomdata_j.index);

            flag = (*handler).read_from_ijR_dense(atmidx_i,atmidx_j,bonddata.Rx,bonddata.Ry,bonddata.Rz, &m,&n,&v,&found);

            if(flag==0){
                
                if(found) {
                    SetBlockR2K(mat, dm, i, j, inorb, jnorb, bonddata.Rx,bonddata.Ry,bonddata.Rz, Kx, Ky, Kx, v);
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
void setMatrixR2K(
    Mat S, DM dm, const double* cell, int ntype, 
    const std::string* file_orb0, const std::string& orbital_dir, 
    double Kx, double Ky, double Kz, Mat Rx, Mat Ry, Mat Rz
) {
    /* Iterate over edge, node, and them:
        1. Get the atom original idx, and the R vector;
        2. Get the H(R)/S(R) block value from io interface;
        3. Put it into matrix through SetBlockR2K. (Actually, it might be possible to use multithreading to accelerate.)
        4. Assemble the matrix, do some small checks.
    */

   // Loop over node
    PetscInt nstart, nend, estart, eend;
    PetscBool ghost;
    AtomData atomdata_i, atomdata_j;
    BondData bonddata;

    TwoCenterBundle tcbd;
    container::Tensor orb_map{container::DataType::DT_INT, container::TensorShape({0})};

    tcbd.build_orb(ntype, file_orb0, orbital_dir);
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

    int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag, Rx_, Ry_, Rz_;
    int itype, il, izeta, im, jtype, jl, jzeta, jm, m_phase;
    double rx, ry, rz;

    ModuleBase::Vector3<double> vR(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> R2(0.0, 0.0, 0.0);

    DMNetworkGetVertexRange(dm, &nstart, &nend);
    for(int i=nstart;i<nend;++i){

        DMNetworkIsGhostVertex(dm,i,&ghost);
        if(!ghost){ /* only assign the non-ghost vertices with onsite-block (since it is the one truly owned by the rank)*/
            rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
            double vS[inorb*inorb], vX[inorb*inorb], vY[inorb*inorb], vZ[inorb*inorb];
            // R vec is zero
            // get block from org index atom_data.index
            atmidx_i=static_cast<int>(atomdata_i.index);
            
            itype = static_cast<int>(atomdata_i.type);
            jtype = static_cast<int>(atomdata_i.type);

            R2.set(atomdata_i.x, atomdata_i.y, atomdata_i.z);
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
            
            SetBlockR2K(S, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vS);
            if (Rx) { SetBlockR2K(Rx, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vX); }
            if (Ry) { SetBlockR2K(Ry, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vY); }
            if (Rz) { SetBlockR2K(Rz, dm, i, i, inorb, inorb, 0, 0, 0, Kx, Ky, Kz, vZ); }

            // Set block for petsc index (i,i) with different R
            for(int iR=0; iR < atomdata_i.NR; ++iR){
                Rx_=atomdata_i.R[3*iR]; Ry_=atomdata_i.R[3*iR+1]; Rz_=atomdata_i.R[3*iR+2]; // cell shift vec for onsite
                rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6];
                ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
                rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
                vR.set(rx, ry, rz);
                R2.set(rx+atomdata_i.x, ry+atomdata_i.y, rz+atomdata_i.z);
                vR *= 1.8897259886; // to bohr
                R2 *= 1.8897259886;

                

                for(int io=0; io<inorb; io++){
                    il=orb_map.get_value<int>(itype, io, 0);
                    izeta=orb_map.get_value<int>(itype, io, 1);
                    im=orb_map.get_value<int>(itype, io, 2);
                    for(int jo=io; jo<inorb; jo++){
                        jl=orb_map.get_value<int>(jtype, jo, 0);
                        jzeta=orb_map.get_value<int>(jtype, jo, 1);
                        jm=orb_map.get_value<int>(jtype, jo, 2);

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

                        m_phase = (im+jm) % 2 == 0 ? 1 : -1;
                        vS[inorb*io+jo] *= m_phase;

                        // copy the value to its complex conjugate
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
                    }
                }

                SetBlockR2K(S,  dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
                if (Rx) { SetBlockR2K(Rx, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX); }
                if (Ry) { SetBlockR2K(Ry, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY); }
                if (Rz) { SetBlockR2K(Rz, dm, i, i, inorb, inorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ); }
            }
        }
    }

    int i, j, eorb;
    DMNetworkGetEdgeRange(dm, &estart, &eend);
    for (int ie=estart; ie<eend; ie++)
    {
        /* We do not check for ghost since dm can only have ghost edge when overlap>0, which is not our case, by now. */
        rsatb::backend::get_edge_index_from_dm(dm, ie, &i, &j);
        rsatb::backend::get_attr_from_dm<BondData>(dm, "edge", ie, &bonddata, &eorb);
        rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", i, &atomdata_i, &inorb);
        rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", j, &atomdata_j, &jnorb);

        double vS[inorb*jnorb], vX[inorb*jnorb], vY[inorb*jnorb], vZ[inorb*jnorb];

        if (inorb > 0 && jnorb > 0){ // careful for those ghost nodes.

            atmidx_i=static_cast<int>(atomdata_i.index);
            atmidx_j=static_cast<int>(atomdata_j.index);
        
            itype = static_cast<int>(atomdata_i.type);
            jtype = static_cast<int>(atomdata_j.type);

            Rx_=bonddata.Rx; Ry_=bonddata.Ry; Rz_=bonddata.Rz; // cell shift for hopping blocks
            rx=Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6]; // cell shift in distance unit
            ry=Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
            rz=Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
            R2.set(rx+atomdata_j.x, ry+atomdata_j.y, rz+atomdata_j.z);
            vR.set(R2.x-atomdata_i.x, R2.y-atomdata_i.y, R2.z-atomdata_i.z);
            vR *= 1.8897259886; // to bohr
            R2 *= 1.8897259886;

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
                        vS + inorb*io+jo, // S
                        (Rx) ? vX + inorb*io+jo : nullptr, // Rx
                        (Ry) ? vY + inorb*io+jo : nullptr, // Ry
                        (Rz) ? vZ + inorb*io+jo : nullptr // Rz
                    );
                    m_phase = (im+jm) % 2 == 0 ? 1 : -1;
                    vS[inorb*io+jo] *= m_phase;

                    if (Rx) { 
                        vX[inorb*io+jo] *= m_phase;
                        vX[inorb*io+jo] /= 1.8897259886;
                    }
                    if (Ry) { 
                        vY[inorb*io+jo] *= m_phase;
                        vY[inorb*io+jo] /= 1.8897259886;
                    }
                    if (Rz) { 
                        vZ[inorb*io+jo] *= m_phase;
                        vZ[inorb*io+jo] /= 1.8897259886;
                    }
                }
            }
            SetBlockR2K(S,  dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vS);
            if (Rx) { SetBlockR2K(Rx, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vX); }
            if (Ry) { SetBlockR2K(Ry, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vY); }
            if (Rz) { SetBlockR2K(Rz, dm, i, j, inorb, jnorb, Rx_, Ry_, Rz_, Kx, Ky, Kz, vZ); }
        }
    }

    MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
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


void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase){
    *phase = std::complex<double>(0.0, -2.0 * M_PI * ((Rx * Ky) + (Ry * Ky) + (Rz * Kz)));
    *phase = std::exp(*phase);
};

}