import numpy as np
from src.utils.constants import anglrMId, norb_dict
from typing import Tuple, Union, Dict
from src.data.transforms import OrbitalMapper
from src.data import AtomicDataDict
from src.data.AtomicDataDict import with_edge_vectors
from src.utils.tools import float2comlex
import sparse

class R2KSparse:
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            spin_deg=True,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            upper=False, # if Upper is true, output only upper half. If False, output full
            dtype=np.float32,
            ):

        self.dtype = dtype
        self.ctype = float2comlex(self.dtype)

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", spin_deg=spin_deg)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            assert idp.method == "e3tb", "The method of idp should be e3tb."
            self.idp = idp

            assert spin_deg == self.idp.spin_deg, "The input spin_deg is not consistent with the input idp"

        # get mask_to_basis incase that spin is neglected
        self.mask_to_basis = self.idp.mask_to_basis.copy() # mask
        self.upper = upper

        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

        # for gGA mapping, there are two circumstances, one is spin-deg, including soc, in this case, the physical system does not have spin degree of freedom
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        atom_types = data.get(AtomicDataDict.ATOM_TYPE_KEY)
        if atom_types is None:
            data = self.idp(data)
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        atom_types = atom_types.flatten()

        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        norb = self.idp.full_basis_norb
        bondwise_hopping = np.zeros((len(orbpair_hopping), norb, norb), dtype=self.dtype)

        onsite_block = np.zeros((len(atom_types), norb, norb,), dtype=self.dtype)
        kpoints = data[AtomicDataDict.KPOINT_KEY]

        ist = 0
        for i,io in enumerate(self.idp.full_basis):
            jst = 0
            iorb = self.idp.flistnorbs[i] * self.idp.spin_factor
            for j,jo in enumerate(self.idp.full_basis):
                jorb = self.idp.flistnorbs[j] * self.idp.spin_factor
                orbpair = io+"-"+jo
                
                # constructing hopping blocks
                if io == jo:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                    onsite_block[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                
                jst += jorb
            ist += iorb

        self.onsite_block = {}
        self.bondwise_hopping = {}
        norb = 0
        for sym, at in self.idp.chemical_symbol_to_type.items():
            mask = self.mask_to_basis[at]
            atmask = atom_types.flatten() == at
            self.onsite_block[sym] = onsite_block[atmask][:,mask][:,:,mask]
            norb += self.onsite_block[sym].shape[1] * self.onsite_block[sym].shape[0]
        
        for bsym, bt in self.idp.bond_to_type.items():
            sym1, sym2 = bsym.split("-")
            at1, at2 = self.idp.chemical_symbol_to_type[sym1], self.idp.chemical_symbol_to_type[sym2]
            mask1 = self.mask_to_basis[at1]
            mask2 = self.mask_to_basis[at2]
            btmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() == bt
            self.bondwise_hopping[bsym] = bondwise_hopping[btmask][:,mask1][:,:,mask2]
        
        # R2K procedure can be done for all kpoint at once.
        # from now on, any spin degeneracy have been removed. All following blocks consider spin degree of freedom
        # block = np.zeros((kpoints.shape[0], norb, norb), dtype=self.ctype)
        block = sparse.COO(np.zeros((1,1,1), dtype=self.ctype))
        block.shape = (kpoints.shape[0], norb, norb)

        atom_id_to_indices = {}
        ist = 0
        type_count = [0] * len(self.idp.type_names)
        oblock_list = []
        for i, at in enumerate(atom_types.flatten()):
            idx = type_count[at]
            sym = self.idp.type_names[at]
            oblock = sparse.COO(self.onsite_block[sym][idx])
            lo = oblock.shape[0]
            oblock.shape = block.shape[1:] # we modify the attributes of COO object directly, this might be a little danger, but.... no other way available currently
            oblock.coords[0] += ist
            oblock.coords[1] += ist
            # transform oblock to sparse matrix that have the same size as block
            oblock_list.append(oblock)
            atom_id_to_indices[i] = slice(ist, ist+oblock.shape[0])
            ist += lo

            type_count[at] += 1

        block += sparse.stack(oblock_list).sum(axis=0)[None,]

        cell_inv = np.linalg.inv(data[AtomicDataDict.CELL_KEY])
        hblock_list = []
        for bsym, btype in self.idp.bond_to_type.items():
            bmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() == btype
            shifts_vec = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][bmask]
            for i, edge in enumerate(edge_index.T[bmask]):
                iatom = edge[0]
                jatom = edge[1]
                shifts_in_vec = data[AtomicDataDict.POSITIONS_KEY][jatom] - data[AtomicDataDict.POSITIONS_KEY][jatom]
                shifts_in_vec = cell_inv @ shifts_in_vec
                iatom_indices = atom_id_to_indices[int(iatom)]
                jatom_indices = atom_id_to_indices[int(jatom)]
                hblock = self.bondwise_hopping[bsym][i]
                hblock = hblock[None,...].astype(block.dtype) * \
                    np.exp(1j * 2 * np.pi * (kpoints @ (shifts_vec[i]+shifts_in_vec))).reshape(-1,1,1)
                # transform hblock to sparse type

                hblock = sparse.COO(hblock)
                hblock.shape = block.shape
                hblock.coords[1] += iatom_indices.start
                hblock.coords[2] += jatom_indices.start

                hblock_list.append(hblock)
        

        block += sparse.stack(hblock_list).sum(axis=0)

        block = block + block.transpose(axes=(0,2,1)).conj()
        
        if self.upper:
            coords, elements = block.coords, block.data
            # row first order
            mask = coords[1,:] <= coords[2,:]
            coords = coords[:,mask]
            elements = elements[mask]
            block = sparse.COO(coords, elements, shape=(kpoints.shape[0], norb, norb))

        data[self.out_field] = block

        return data # here output hamiltonian have their spin orbital connected between each other

