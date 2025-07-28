#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"
#include "hr2hk.hpp"
#include <math.h>

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
    int inorb, jnorb, atmidx_i, atmidx_j, m, n, flag;
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

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
   
};


void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase){
    *phase = std::complex<double>(0.0, -2.0 * M_PI * ((Rx * Ky) + (Ry * Ky) + (Rz * Kz)));
    *phase = std::exp(*phase);
};

} 