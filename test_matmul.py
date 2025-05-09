from sparse_dot_mkl import dot_product_mkl
import scipy.sparse as sp
import numpy as np
import time

if __name__ == "__main__":

    tt = []
    rng = np.random.default_rng()

    for n in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        D = 500./n
        data = rng.random((int(n*n*D)), dtype=np.float32)
        rol = np.random.randint(0, n, len(data), dtype=np.int32)
        col = np.random.randint(0, n, len(data), dtype=np.int32)
        A = sp.csr_array((data, (rol, col)), dtype=np.float32)
        B = rng.random((n, 1000), dtype=np.float32)

        start = time.perf_counter()
        dot_product_mkl(A, B)
        end = time.perf_counter()

        tt.append(end-start)

        print(tt[-1])