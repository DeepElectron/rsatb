import numpy as np
from ..utils.constants import anglrMId, norb_dict
from typing import Tuple, Union, Dict
from ..data.transforms import OrbitalMapper
from ..data import AtomicDataDict
from ..data.AtomicDataDict import with_edge_vectors
from ..utils.tools import float2comlex
import sparse


def sparseR2Kmpi(
        block, 
        partition, 
        cell, 
        atom_types, 
        edge_index, 
        edge_cell_shift, 
        positions, 
        atom_norb, 
        kpoints, 
        upper=False,
        comm=None) -> sparse.COO:

    # construct bond wise hamiltonian block from obital pair wise node/edge features
    # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

    # for gGA mapping, there are two circumstances, one is spin-deg, including soc, in this case, the physical system does not have spin degree of freedom
    atom_types = atom_types.flatten()
    natoms = len(atom_types)
    ctype = float2comlex(block[list(block.keys())[0]][:].dtype)

    rank = comm.Get_rank()
    mask = partition.eq(rank)

    norb_startend = np.cumsum(np.append([0],atom_norb[atom_types])) # get the start and end index of orbitals of each atom
    norb_all = norb_startend[-1]
    
    # R2K procedure can be done for all kpoint at once.
    # from now on, any spin degeneracy have been resolved.
    hk = sparse.COO(np.zeros((1,1,1), dtype=ctype))
    hk.shape = (kpoints.shape[0], norb_all, norb_all)

    oblock_list = []
    for i in range(natoms):
        if mask[i]: # which means the ith atom is hold by this rank
            ist = norb_startend[i]
            kk = "_".join([i, i, 0,0,0])
            oblock = sparse.COO(block[kk][:])
            oblock.shape = hk.shape[1:] # we modify the attributes of COO object directly, this might be a little danger, but.... no other way available currently
            oblock.coords[0] += ist
            oblock.coords[1] += ist
            # transform oblock to sparse matrix that have the same size as block
            oblock_list.append(oblock)

    hk += sparse.stack(oblock_list).sum(axis=0)[None,]

    # for hopping block

    cell_inv = np.linalg.inv(cell)
    hblock_list = []
    shifts_vec = edge_cell_shift
    for i, edge in enumerate(edge_index.T):
        iatom = edge[0]
        jatom = edge[1]
        if mask[iatom]:
            shifts_in_vec = positions[jatom] - positions[jatom]
            shifts_in_vec = cell_inv @ shifts_in_vec
            ist = norb_startend[iatom]
            jst = norb_startend[jatom]
            shifts_vec_ij = shifts_vec[i]
            kk = "_".join([iatom, jatom, shifts_vec_ij[0],shifts_vec_ij[1],shifts_vec_ij[2]])
            kk_rev = "_".join([jatom, iatom, -shifts_vec_ij[0],-shifts_vec_ij[1],-shifts_vec_ij[2]])
            if kk in block:
                hblock = block[kk][:]
            elif kk_rev in block:
                hblock = block[kk][:].T # not taking conjugate since all numbers are real
            else:
                continue

            hblock = hblock[None,...].astype(hk.dtype) * \
                np.exp(1j * 2 * np.pi * (kpoints @ (shifts_vec_ij+shifts_in_vec))).reshape(-1,1,1)
            # transform hblock to sparse type

            hblock = sparse.COO(hblock)
            hblock.shape = hk.shape
            hblock.coords[1] += ist
            hblock.coords[2] += jst

            hblock_list.append(hblock)
    
    hk += sparse.stack(hblock_list).sum(axis=0)
    
    if upper:
        coords, elements = hk.coords, hk.data
        # row first order
        mask = coords[1,:] <= coords[2,:]
        coords = coords[:,mask]
        elements = elements[mask]
        hk = sparse.COO(coords, elements, shape=(kpoints.shape[0], norb_all, norb_all))

    return hk # here output hamiltonian have their spin orbital connected between each other

