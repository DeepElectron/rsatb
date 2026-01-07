import numpy as np
import ase.io
import os
import time
from rsatb.data import AtomicData

def calculate_edge_cut(ad):
    """
    Calculate the total edge cut of the partitioned graph.
    Edge cut is the number of edges (u, v) such that partition[u] != partition[v].
    """
    if not hasattr(ad, 'partition'):
        print("Error: Graph not partitioned.")
        return -1

    partition = ad.partition
    edge_index = ad.edge_index
    
    # edge_index is [2, num_edges]
    u = edge_index[0]
    v = edge_index[1]
    
    part_u = partition[u]
    part_v = partition[v]
    
    # Count edges where partitions differ
    cut_edges = np.sum(part_u != part_v)
    
    return cut_edges

def benchmark_random(ad, npart):
    """
    Perform random partitioning.
    """
    t_start = time.time()
    partition = np.random.randint(0, npart, size=ad.num_nodes, dtype=np.int32)
    
    # We need to finalize to set the partition attribute and reorder if needed
    # But for random, we just want to set the partition attribute to calculate edge cut.
    # However, calculate_edge_cut expects ad.partition.
    # Also, _finalize_partition does a lot of work (sorting, etc) which might not be needed just for edge cut check
    # but is needed for a fair timing comparison if we consider "partitioning time" to include setup.
    # But Random is a baseline. Let's just set the attribute.
    
    # To be fair with other methods that reorder atoms, we should probably not reorder for random 
    # unless we want to measure that overhead too. 
    # Let's just set the partition array directly.
    ad.partition = partition
    t_end = time.time()
    
    return t_end - t_start

def benchmark_metis(ad, npart):
    """
    Perform Metis partitioning.
    """
    t_start = time.time()
    try:
        # partition_graph uses pymetis and does finalize
        ad.partition_graph(npart, split=False)
    except ImportError:
        print("Metis (pymetis) not installed.")
        return None
    except Exception as e:
        print(f"Metis failed: {e}")
        return None
    t_end = time.time()
    return t_end - t_start

def benchmark_hilbert(ad, npart):
    """
    Perform Hilbert partitioning.
    """
    t_start = time.time()
    ad.partition_graph_hilbert(npart, split=False)
    t_end = time.time()
    return t_end - t_start

def benchmark_rib(ad, npart):
    """
    Perform Recursive Inertial Bisection (RIB) partitioning.
    """
    t_start = time.time()
    ad.partition_graph_rib(npart, split=False)
    t_end = time.time()
    return t_end - t_start

def benchmark_file(filepath, r_max, nparts):
    print(f"\nBenchmarking {os.path.basename(filepath)} with r_max={r_max}")
    
    try:
        atoms = ase.io.read(filepath, format='vasp')
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return

    print(f"Number of atoms: {len(atoms)}")
    
    t0 = time.time()
    ad_orig = AtomicData.from_ase(atoms, r_max=r_max)
    t1 = time.time()
    print(f"Neighbor list build time: {t1 - t0:.4f} s")
    print(f"Number of edges: {ad_orig.edge_index.shape[1]}")
    
    print(f"{'npart':<6} | {'Method':<10} | {'Edge Cut':<10} | {'Time (s)':<10} | {'Ratio (vs Rnd)':<15}")
    print("-" * 65)

    for npart in nparts:
        # --- Random ---
        ad_rnd = AtomicData.from_ase(atoms, r_max=r_max) # Fresh copy
        time_rnd = benchmark_random(ad_rnd, npart)
        cut_rnd = calculate_edge_cut(ad_rnd)
        print(f"{npart:<6} | {'Random':<10} | {cut_rnd:<10} | {time_rnd:<10.4f} | {'1.00':<15}")

        # --- Metis ---
        ad_metis = AtomicData.from_ase(atoms, r_max=r_max) # Fresh copy
        time_metis = benchmark_metis(ad_metis, npart)
        if time_metis is not None:
            cut_metis = calculate_edge_cut(ad_metis)
            ratio_metis = cut_metis / cut_rnd if cut_rnd > 0 else 0
            print(f"{'':<6} | {'Metis':<10} | {cut_metis:<10} | {time_metis:<10.4f} | {ratio_metis:<15.4f}")
        
        # --- Hilbert ---
        ad_hilbert = AtomicData.from_ase(atoms, r_max=r_max) # Fresh copy
        time_hilbert = benchmark_hilbert(ad_hilbert, npart)
        cut_hilbert = calculate_edge_cut(ad_hilbert)
        ratio_hilbert = cut_hilbert / cut_rnd if cut_rnd > 0 else 0
        print(f"{'':<6} | {'Hilbert':<10} | {cut_hilbert:<10} | {time_hilbert:<10.4f} | {ratio_hilbert:<15.4f}")
        
        # --- RIB ---
        ad_rib = AtomicData.from_ase(atoms, r_max=r_max) # Fresh copy
        time_rib = benchmark_rib(ad_rib, npart)
        cut_rib = calculate_edge_cut(ad_rib)
        ratio_rib = cut_rib / cut_rnd if cut_rnd > 0 else 0
        print(f"{'':<6} | {'RIB':<10} | {cut_rib:<10} | {time_rib:<10.4f} | {ratio_rib:<15.4f}")
        
        print("-" * 65)

def main():
    # Define datasets and parameters
    datasets = [
        ("/home/zhanghao/softwares/rsatb/test/data/diamond/diamond_sc888.vasp", 7.5),
        ("/home/zhanghao/softwares/rsatb/test/data/moire/moire_test.vasp", 6.0),
        ("/home/zhanghao/softwares/rsatb/test/data/Al/Al282828.vasp", 6.0),
        ("/home/zhanghao/softwares/rsatb/test/data/Al/Al434343.vasp", 6.0),
    ]
    
    nparts_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    for filepath, r_max in datasets:
        if os.path.exists(filepath):
            # Read atoms first to determine valid nparts
            try:
                atoms = ase.io.read(filepath, format='vasp')
                num_atoms = len(atoms)
                # Filter nparts that are feasible (e.g., <= num_atoms)
                # Also maybe we don't want 1 atom per partition, but let's allow up to num_atoms for now if listed.
                valid_nparts = [n for n in nparts_list if n <= num_atoms]
                
                if not valid_nparts:
                    print(f"Skipping {os.path.basename(filepath)}: Too small for smallest npart ({nparts_list[0]})")
                    continue
                    
                benchmark_file(filepath, r_max, valid_nparts)
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
