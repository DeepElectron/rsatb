The goal is to refactor atomic_data to match the version in python. The python version is in rsatb/data/AtomicData.py
The core function needed:

1. constructor from points, the function in python version is AtomicData.from_points. It allows the user gives only the points, cell and atomic numbers, and it will generate the neighbor list and other necessary data. Here, we only need one r_max, one neighbour list is enough.

For the neighbour list search, ase have a very efficient algorithm, I will past it here for reference. @neighbourlist.py You can create a new file .hpp to implement this algorithm.

2. constructor from files. The AtomicData support this by reading ASE data, where ase provide the reading function for many file formats. The function in python version is AtomicData.from_ase. Here in c++, the function should read directly on rank 0 from the structure files, it should support many different file formats. For example, POSCAR, CIF, XYZ. The function should read the essential data and then call the from_points.

3. Graph partition. This allows the user to read the graph serially, and then partition the graph to differnt ranks to perform parrallel calculation. The partition should use ParMETIS for this task.

4. The communicator build can follow from current implementation.

5. 