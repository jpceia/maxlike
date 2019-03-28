import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array


cdef np.ndarray last_diag(np.ndarray arr, int axis1, int axis2):
    return np.diagonal(arr, axis1=axis1, axis2=axis2).\
           swapaxes(-1, axis2 - 1)


cpdef np.ndarray arr_take_diag(np.ndarray arr, int q, int p, array mapping):
    cdef int k, f
    k = q
    for f in mapping:
        if f < 0:
            continue
        arr = np.expand_dims(last_diag(arr, k, p + f), k)
        k += 1
    return arr


cpdef np.ndarray arr_expand_diag(np.ndarray arr, int p, array map1, array map2):
    cdef int i, j, n, c
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    k = 0
    for j in map2:
        i = map1[k]
        k += 1
        if i == j:
            continue
        n = arr.shape[p + i]
        idx[p + i] = n
        idx[p + j] = n
        arr = arr * np.eye(n).reshape(idx)
        idx[p + i] = 1
        idx[p + j] = 1
    return arr


cpdef array compose_mappings(array map1, array map2):
    cdef int k
    cdef array result = array('b')
    for k in map2:
        if k < 0:
            result.append(-1)
        else:
            result.append(map1[k])
    return result


cpdef np.ndarray arr_swapaxes(np.ndarray arr, int q, int p, array mapping):
    cdef int k, f
    k = q
    for f in mapping:
        if f >= 0:
            arr = arr.swapaxes(k, p + f)
        k += 1
    return arr


cdef np.ndarray arr_swapaxes_cross(np.ndarray arr, int p, array map1, array map2):
    cdef k, f, l, n
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    k = p
    for f in map2:
        n = arr.shape[k - p]
        if map1 and f in map1:
            l = map1.index(f)
            idx[l] = n
            idx[k] = n
            val = val * np.eye(n).reshape(idx)
            idx[l] = 1
            idx[k] = 1
        else:
            arr = arr.swapaxes(k, p + f)
        k += 1
    return arr
