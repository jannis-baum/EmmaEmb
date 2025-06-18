import numba as nb
import numpy as np

# parallelized version of `np.argsort(matrix, axis=1)`
@nb.njit(parallel=True)
def row_argsort_parallel(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b
