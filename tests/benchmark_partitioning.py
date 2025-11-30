import numpy as np
import time

def hilbert_curve_3d(coords, n_bits):
    """
    Map 3D coordinates to Hilbert curve index.
    coords: (N, 3) int array, values in [0, 2**n_bits - 1]
    Returns: (N,) int array of Hilbert indices
    """
    # Using Z-curve (Morton order) as a proxy for geometric locality speed 
    x = coords[:, 0].astype(np.uint64)
    y = coords[:, 1].astype(np.uint64)
    z = coords[:, 2].astype(np.uint64)
    
    code = np.zeros(x.shape, dtype=np.uint64)
    for i in range(n_bits):
        code |= ((x >> i) & 1) << (3*i)
        code |= ((y >> i) & 1) << (3*i + 1)
        code |= ((z >> i) & 1) << (3*i + 2)
        
    return code

def benchmark_partitioning(n_atoms=100000, n_parts=16):
    print(f"Benchmarking partitioning for {n_atoms} atoms into {n_parts} parts...")
    
    # 1. Generate synthetic crystal (simple cubic grid)
    L = int(n_atoms**(1/3))
    x = np.linspace(0, 1, L)
    y = np.linspace(0, 1, L)
    z = np.linspace(0, 1, L)
    xx, yy, zz = np.meshgrid(x, y, z)
    pos = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    # Add some noise/disorder
    pos += np.random.uniform(-0.01, 0.01, pos.shape)
    
    # Ensure in [0, 1]
    pos -= pos.min()
    pos /= pos.max()
    
    print(f"Generated {len(pos)} atoms.")
    
    # 2. Build Neighbor List (Cutoff 0.1) to simulate bonds
    # Simple brute force for benchmark (N is small enough for this demo, or use cdist)
    # Actually for 100k atoms brute force is too slow. Let's assume simple cubic connectivity.
    # Each atom connected to 6 neighbors.
    # We can infer connectivity from grid indices.
    
    # Grid indices
    grid_idx = np.arange(len(pos)).reshape(L, L, L)
    edges = []
    # X neighbors
    edges.append(np.stack([grid_idx[:-1, :, :].ravel(), grid_idx[1:, :, :].ravel()], axis=1))
    # Y neighbors
    edges.append(np.stack([grid_idx[:, :-1, :].ravel(), grid_idx[:, 1:, :].ravel()], axis=1))
    # Z neighbors
    edges.append(np.stack([grid_idx[:, :, :-1].ravel(), grid_idx[:, :, 1:].ravel()], axis=1))
    
    edge_index = np.vstack(edges).T # (2, E)
    num_edges = edge_index.shape[1]
    print(f"Generated {num_edges} edges (simple cubic connectivity).")

    start_time = time.time()
    
    # 3. Discretize
    n_bits = 21 
    max_val = (1 << n_bits) - 1
    coords = (pos * max_val).astype(np.int32)
    
    # 4. Compute Space Filling Curve Index
    indices = hilbert_curve_3d(coords, n_bits)
    
    # 5. Sort
    sort_idx = np.argsort(indices)
    
    # 6. Partition
    # split into equal chunks
    chunk_size = len(pos) // n_parts
    partition = np.zeros(len(pos), dtype=np.int32)
    for i in range(n_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_parts - 1 else len(pos)
        partition[sort_idx[start:end]] = i
        
    end_time = time.time()
    
    print(f"Partitioning took {end_time - start_time:.4f} seconds.")
    
    # 7. Calculate Edge Cut
    # Edges where partition[u] != partition[v]
    part_u = partition[edge_index[0]]
    part_v = partition[edge_index[1]]
    cut_edges = np.sum(part_u != part_v)
    
    print(f"Edge Cut: {cut_edges} / {num_edges} ({cut_edges/num_edges*100:.2f}%)")
    print(f"This metric directly correlates to SpMV communication volume.")
    
    return end_time - start_time

if __name__ == "__main__":
    benchmark_partitioning()
