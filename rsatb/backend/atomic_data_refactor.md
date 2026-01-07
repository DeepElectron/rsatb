Neighbour list:
For the neighbour list search, ase have a very efficient algorithm, I will past it here for reference. @neighbourlist.py   You can create a new file .hpp to implement this algorithm. The neighbour list search should support both the PBC case and non-pbc case. Indeed, the pbc will be a vector of three bools. It indicates the PBC open or close in the three direction. The cell will be a 3x3 matrix. The points will be a Nx3 matrix. The neighbour list will be a list of list of int. Please consider the efficiency as of the top priority, since it might need to perform neighbour search over a system with 1000k atoms.


the goal is to refactor @atomic_data.hpp   to match the version in python. The python version is in @AtomicData.py  
The core function needed:

1. constructor from points, the function in python version is AtomicData.from_points. It allows the user gives only the points, cell and atom type (instead of atom type, we use atomic number here), and it will generate the neighbor list and other necessary data. This function will be support both serial and parrallel calls, where the neighbour list will be constructed in rank 0 and bcast to others. Then, the graph partition will be performed in parrallel, and reconstruct to finalize the distributed atomic data. the rmax of this function will be a list of rmax for each atom type.

2. constructor from files. The AtomicData support this by reading ASE data, where ase provide the reading function for many file formats. The function in python version is AtomicData.from_ase. Here in c++, the function should read directly on rank 0 from the structure files, it should support many different file formats. For example, POSCAR, CIF, XYZ. The function should read the essential data and then call the from_points. You can add a file dedicated to work on read files. After reading, the atomic information will be processed to atom type, cell, points, and then call the from_points. The r_max will be a list of rmax for each atom type as input of this function also, and pass to from_points.

3. Graph partition. This can be implemented as a private function with no exposed interface to avoid complication. This will only be called in the constructor from points. This allows the user to read the graph serially, and then partition the graph to differnt ranks to perform parrallel calculation. The partition should use ParMETIS for this task. The partition and the split can be combined to a single function in c++.

The image container class.
The image container class will be created. Between the atomic_data class, where it stores the distributed graph and neighbour list information, and the block csr class, which stores the block sparse matrix. 

constructor: input: the atomic_data. It add a reference to the graph of the atomic_data as base_graph of this image container, and then, it get the neighbour list from the atomic_data for each shift vector R, to construct a independent graph. The base graph actaully corresponding to a summation of graph for all shifted vector R.

function:
1. add block: it should mimic the function of the block_csr class. While it need to check whether the csr for that R block is already exist. If it is, it should call the add block of the block_csr class to add, if not, it should allocate the block_csr using the graph corresponding to that R.

2. Sample K: it will do a R to K transform using FFT. It will call the atomic_data to get the atomic positional and cell information, and then, it will create a block_csr using the base_graph. Then, it will fill the new created block_csr with the block_csr corresponding to each shift vector R, with the multiplication of their specific block of the fourier phase. Two convention is needed. One convention compute the phase use the R and K, while the other also consider the position inforamtion of the atoms.