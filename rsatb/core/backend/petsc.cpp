#include <stdexcept>
#include <petsc.h>
#include <mpi.h>
#include "petsc.hpp"
#include <string>
#include <iostream>
#include <algorithm>

// What does this look like when running in parrallel?

namespace rsatb::backend{

PetscErrorCode get_dm_atomic(
            int n_node,
            int n_edge,
            int *edge_index,
            int *atom_type, 
            int *type_weight,
            int *edge_shift_vec,
            double *pos,
            DM *dm,
            MPI_Comm comm
            )
{
    PetscInt atomdata_key, bonddata_key;
    PetscInt n_node_petsc = (PetscInt)n_node;
    PetscInt n_edge_petsc = (PetscInt)n_edge;

    PetscInt numnet=1;
    PetscInt zero=0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    DMNetworkCreate(comm, dm);

    DMNetworkRegisterComponent(*dm, "Atom", sizeof(AtomData), &atomdata_key);
    DMNetworkRegisterComponent(*dm, "Bond", sizeof(BondData), &bonddata_key);
    // register other components
    if (rank == 0) {DMNetworkSetNumSubNetworks(*dm, numnet, numnet);} // local, global num
    else {DMNetworkSetNumSubNetworks(*dm, zero, numnet);}
    
    if (rank == 0){
        PetscInt netnum=-1;

        PetscInt *edge_index_petsc = nullptr;
        PetscInt *atom_type_petsc  = nullptr;
        PetscInt *type_weight_petsc = nullptr;
        PetscInt *edge_shift_vec_petsc = nullptr;
        PetscReal *pos_petsc = nullptr;

        PetscBool free_atom_type;
        PetscBool free_edge_index;
        PetscBool free_type_weight;
        PetscBool free_edge_shift_vec;
        PetscBool free_pos_petsc;

        // handle the PETScInt and int mismatch
        PetscCall(int2petscint(n_node, atom_type, &atom_type_petsc, &free_atom_type));
        PetscCall(int2petscint(n_edge*2, edge_index, &edge_index_petsc, &free_edge_index));
        PetscCall(int2petscint(*std::max_element(atom_type , atom_type + n_node), type_weight, &type_weight_petsc, &free_type_weight));
        PetscCall(int2petscint(n_edge*3, edge_shift_vec, &edge_shift_vec_petsc, &free_edge_shift_vec));
        PetscCall(double2petscreal(n_node*3, pos, &pos_petsc, &free_pos_petsc));

        DMNetworkAddSubnetwork(
            *dm, "atomicdata", 
            n_edge_petsc,
            edge_index_petsc,
            // reinterpret_cast<PetscInt*>(edge_index), 
            &netnum
        );

        // std::cout << "NETNUM on rank " << rank << " " << netnum << "\n";

        DMNetworkLayoutSetUp(*dm);

        // add components
        PetscInt nstart, nend;
        DMNetworkGetVertexRange(*dm, &nstart, &nend);
        PetscInt atom_index=0;

        AtomData *atomdata = nullptr;
        // BondData *bonddata = nullptr;
        PetscCalloc1(n_node_petsc,&atomdata);      // PETSc-managed heap, lives as long as DM (Becareful about whether it is needed.)
        // PetscCalloc1(n_edge,&bonddata);
        for (PetscInt p = nstart; p < nend; p++)
        {
            atomdata[atom_index].index = atom_index;
            atomdata[atom_index].x = pos_petsc[atom_index*3]; // indxing [] op itself is the process of depointer, a[0] = *a, a[1] = *(a+1)
            atomdata[atom_index].y = pos_petsc[atom_index*3+1];
            atomdata[atom_index].z = pos_petsc[atom_index*3+2];
            atomdata[atom_index].type = atom_type_petsc[atom_index];

            DMNetworkAddComponent(*dm, p, atomdata_key, &atomdata[atom_index], type_weight_petsc[atom_type_petsc[atom_index]]);
            atom_index++;
        }

        PetscInt estart, eend;
        DMNetworkGetEdgeRange(*dm, &estart, &eend);
        PetscInt edge_count=0;

        BondData *bonddata = nullptr;
        PetscCalloc1(n_edge_petsc,&bonddata);
        for (PetscInt p = estart; p < eend; p++)
        {
            bonddata[edge_count].Rx = edge_shift_vec_petsc[edge_count*3];
            bonddata[edge_count].Ry = edge_shift_vec_petsc[edge_count*3+1];
            bonddata[edge_count].Rz = edge_shift_vec_petsc[edge_count*3+2];

            DMNetworkAddComponent(*dm, p, bonddata_key, &bonddata[edge_count], 0);
            edge_count += 1; // for xyz component
        }
        DMSetUp(*dm);

        DMNetworkDistribute(dm, 0);

        if(free_atom_type) {PetscCall(PetscFree(atom_type_petsc));}
        if(free_edge_index) {PetscCall(PetscFree(edge_index_petsc));}
        if(free_type_weight) {PetscCall(PetscFree(type_weight_petsc));}
        if(free_edge_shift_vec) {PetscCall(PetscFree(edge_shift_vec_petsc));}
        if(free_pos_petsc) {PetscCall(PetscFree(pos_petsc));}

    } else {
        PetscInt netnum=-1;

        DMNetworkAddSubnetwork(
            *dm, "atomicdata", 
            0,
            NULL,
            // reinterpret_cast<PetscInt*>(edge_index), 
            &netnum
        );

        // std::cout << "NETNUM on rank " << rank << " " << netnum << "\n";

        DMNetworkLayoutSetUp(*dm);
        
        DMSetUp(*dm);
        DMNetworkDistribute(dm, 0);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
};

PetscErrorCode ViewDMNetworkPartition(DM dm, bool detail=false)
{
    PetscErrorCode ierr;
    PetscInt       vStart,vEnd,eStart,eEnd;
    int            rank;
    MPI_Comm       comm = PetscObjectComm((PetscObject)dm);

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexRange(dm,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMNetworkGetEdgeRange  (dm,&eStart,&eEnd);CHKERRQ(ierr);

    PetscPrintf(comm,
                "-----------------------------\n"
                "Rank %d local vertices %lld-%lld, edges %lld-%lld\n",
                rank,(long long)vStart,(long long)(vEnd-1),
                    (long long)eStart,(long long)(eEnd-1));

    if(detail){
        /* Vertices */
        for (PetscInt p=vStart; p<vEnd; ++p){
            PetscInt goff;
            ierr = DMNetworkGetGlobalVecOffset(dm,p,ALL_COMPONENTS,&goff);CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_SELF,
                        "  [vertex] local p=%" PetscInt_FMT
                        "  globalOffset=%" PetscInt_FMT "\n",p,goff);
        }

        /* Edges */
        for (PetscInt p=eStart; p<eEnd; ++p){
            PetscInt goff;
            ierr = DMNetworkGetGlobalVecOffset(dm,p,ALL_COMPONENTS,&goff);CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_SELF,
                        "  [edge  ] local p=%" PetscInt_FMT
                        "  globalOffset=%" PetscInt_FMT "\n",p,goff);
        }
    }

    /* ensure nice ordering of output */
    ierr = PetscBarrier((PetscObject)dm);CHKERRQ(ierr);
    return PETSC_SUCCESS;
}

template<typename T>
void get_attr_from_dm(
            DM dm,
            std::string tp,
            int idx,
            // output
            T *attr,
            int *nvar
)
{
    PetscInt nvar_, idx_petsc;
    idx_petsc = (PetscInt)idx;

    if(tp == "atom"){
        PetscInt nstart, nend;
        DMNetworkGetVertexRange(dm, &nstart, &nend);
        // std::cout << "ns " << nstart << "ne " << nend << "\n";
        // if(idx_petsc>=nend || idx_petsc < nstart){
        //     throw std::invalid_argument( "idx out of range!" );
        // }
        PetscInt compkey;
        T *ptr = nullptr;
        DMNetworkGetComponent(dm, idx_petsc, 0, &compkey, (void**)&ptr, &nvar_);
        *attr = *ptr;
        *nvar = nvar_;
    }else if(tp == "edge"){
        PetscInt estart, eend;
        DMNetworkGetEdgeRange(dm, &estart, &eend);
        if(idx_petsc>=eend || idx_petsc < estart){
            throw std::invalid_argument( "idx out of range!" );
        }

        PetscInt compkey;
        T *ptr = nullptr;
        DMNetworkGetComponent(dm, idx_petsc, 0, &compkey, (void**)&ptr, &nvar_);
        *attr = *ptr;
        *nvar = nvar_;
        // This is a classical example of pointer's usage
        // If I pass T *attr to this function, and then pass &attr to DMNetworkGetComponent. What will happen?
        // first: The attr, which is a pointer of struct, will be altered. We will think that &attr now should point to the 
        // exact component right? Yes! That happened, but only in this function, because C++ have an exact copy based param!
        // So, attr as type T* here, is a copy! of the address of AtomicData/BondData, so, any alter of the attr would not change
        // the original data (that create attr via &data) outof this function. The only way a function can change the value of the data
        // of the pointer variable is by directly modify the value it point to *attr=....
    } else {
        throw std::invalid_argument( "wrong attr_name!" );
    }
};
template void get_attr_from_dm<AtomData>(DM, std::string, int, AtomData*, int *nvar);
template void get_attr_from_dm<BondData>(DM, std::string, int, BondData*, int *nvar);

void get_edge_index_from_dm(
    DM dm,
    int idx,
    // output
    int *i,
    int *j
){
    PetscInt nstart, nend, estart, eend, idx_petsc;
    idx_petsc = (PetscInt)idx;
    const PetscInt *index=nullptr;
    DMNetworkGetEdgeRange(dm, &estart, &eend);
    DMNetworkGetVertexRange(dm, &nstart, &nend);
    if(idx_petsc>=eend || idx_petsc < estart){
        // std::cout << p << " " << eend << "\n";
        throw std::invalid_argument( "idx out of range!" );
    }
    
    DMNetworkGetConnectedVertices(dm, idx_petsc, &index);
    *i = static_cast<int>(index[0]);
    *j = static_cast<int>(index[1]); // note: If pass a pointer int *i, to this function, where i is not point to any value, it will 
    // get empty pointer error here.
};

// The KSP setup

void setup_KSP(
    DM dm, 
    Mat S,
    int nrhs,
    MPI_Comm comm,
    //output
    KSP *ksp
){
    PetscInt m, n;
    MatGetSize(S, &m, &n);
    if(m!=n){throw std::invalid_argument("Wrong Mat dimension");}
    KSPCreate(comm, ksp);
    KSPSetDM(*ksp, dm);
    KSPSetDMActive(*ksp,PETSC_FALSE);
    KSPSetOperators(*ksp, S, S);

    std::string opts;
    if(m<3e5){
            opts =
            "-ksp_type preonly -ksp_max_it 100 "
            "-pc_type cholesky -ksp_reuse_preconditioner true";
    }else if(m>=3e5 && m<=1e6){
        if(nrhs<=64){
            opts =
                "-ksp_type hpddm -ksp_hpddm_type bcg "
                "-pc_type gamg -ksp_reuse_preconditioner true";
        }else if (nrhs > 64) // not very common
        {
            opts =
                "-ksp_type hpddm -ksp_hpddm_type bcg "
                "-pc_type hpddm -ksp_reuse_preconditioner true";
        }
    }else if (m>1e6)
    {
        // it should be only 1 RHS or less than 10 RHS now
        opts =
            "-ksp_type hpddm -ksp_hpddm_type bcg "
            "-pc_type  -ksp_reuse_preconditioner true";
    } else {
        throw std::invalid_argument( "mat size invalid !" );
    }
    
    PetscOptionsInsertString(NULL,opts.c_str());
    KSPSetFromOptions(*ksp);
    KSPSetTolerances(*ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 100);
    KSPSetUp(*ksp);
};


void setup_is_atomic(DM dm, IS *isSrc, IS *isDst){
    PetscInt nLocal=0;
    PetscBool ghost;
    PetscInt vStart,vEnd,off;
    DMNetworkGetVertexRange(dm,&vStart,&vEnd);
    for (PetscInt v=vStart; v<vEnd; ++v) {
        DMNetworkIsGhostVertex(dm,v,&ghost);
        if(!ghost) nLocal++;
    }
    PetscInt *src = (PetscInt*)malloc(nLocal*sizeof(*src));
    PetscInt *dst = (PetscInt*)malloc(nLocal*sizeof(*dst));

    int norb, count=0;
    AtomData atomdata;
    for (PetscInt v=vStart; v<vEnd; ++v) {
        DMNetworkIsGhostVertex(dm,v,&ghost);
        if(!ghost){

            rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", v, &atomdata, &norb);
            DMNetworkGetLocalVecOffset(dm,v,ALL_COMPONENTS,&off);
            for(int i=0; i<norb; ++i){
                src[count+i] = off+i;
                dst[count+i] = atomdata.index;
                count++;
            }
            
        }
    }

    ISCreateGeneral(PetscObjectComm((PetscObject)dm),nLocal,src,PETSC_COPY_VALUES,isSrc);
    ISCreateGeneral(PetscObjectComm((PetscObject)dm),nLocal,dst,PETSC_COPY_VALUES,isDst);
};

void setup_is_orbital(DM dm, int *orbtype, int *type_weight, int n_type, IS *isSrc, IS *isDst){
    PetscInt nLocal=0;
    PetscBool ghost;
    PetscInt vStart,vEnd,off;
    DMNetworkGetVertexRange(dm,&vStart,&vEnd);
    for (PetscInt v=vStart; v<vEnd; ++v) {
        DMNetworkIsGhostVertex(dm,v,&ghost);
        if(!ghost) nLocal++;
    }
    int* orbtypemap[n_type];
    orbtypemap[0] = orbtype;
    for(int i=0;i<n_type-1;++i){
        orbtypemap[i+1] = orbtype + type_weight[i];
    }

    PetscInt *src = (PetscInt*)malloc(nLocal*sizeof(*src));
    PetscInt *dst = (PetscInt*)malloc(nLocal*sizeof(*dst));
    int norb, count=0;
    AtomData atomdata;
    for (PetscInt v=vStart; v<vEnd; ++v) {
        DMNetworkIsGhostVertex(dm,v,&ghost);
        if(!ghost){
            rsatb::backend::get_attr_from_dm<AtomData>(dm, "atom", v, &atomdata, &norb);
            DMNetworkGetLocalVecOffset(dm,v,ALL_COMPONENTS,&off);
            for(int i=0; i<norb; ++i){
                src[count+i] = off+i;
                dst[count+i] = orbtypemap[atomdata.type][i];
                count++;
            }
            
        }
    }

    ISCreateGeneral(PetscObjectComm((PetscObject)dm),nLocal,src,PETSC_COPY_VALUES,isSrc);
    ISCreateGeneral(PetscObjectComm((PetscObject)dm),nLocal,dst,PETSC_COPY_VALUES,isDst);

};

}