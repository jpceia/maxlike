#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import cython
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
np.import_array()



cpdef np.ndarray arr_take_diag(np.ndarray arr, char q, char p, char[:] mapping):
    cdef char k, f
    for k in range(len(mapping)):
        f = mapping[k]
        if f < 0:
            continue
        arr = np.PyArray_Diagonal(arr, 0, q + k, p + f)  # np.diagonal
        arr = arr.swapaxes(-1, p + f - 1)
        arr = np.expand_dims(arr, k)
    return arr


cpdef np.ndarray arr_expand_diag(np.ndarray arr, char q, char p, char[:] mapping):
    cdef char k, f
    cdef int n
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    for k in range(len(mapping)):
        f = mapping[k]
        if f < 0:
            continue
        n = arr.shape[p + f]
        idx[q + k] = n
        idx[p + f] = n
        arr = arr * np.eye(n).reshape(idx)
        idx[q + k] = 1
        idx[p + f] = 1
    return arr


cpdef np.ndarray arr_expand_cross_diag(np.ndarray arr, char p, char[:] map1, char[:] map2):
    cdef char i, j
    cdef int n
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    for k in range(len(map2)):
        i = map1[k]
        j = map2[k]
        if i == j:
            continue
        n = arr.shape[p + i]
        idx[p + i] = n
        idx[p + j] = n
        arr = arr * np.broadcast_to(np.eye(n), idx)
        idx[p + i] = 1
        idx[p + j] = 1
    return arr


cpdef array compose_mappings(char[:] map1, char[:] map2):
    cdef char k, f
    cdef array result = array('b')
    for k in range(len(map2)):
        f = map2[k]
        if f < 0:
            result.append(-1)
        else:
            result.append(map1[f])
    return result


cpdef np.ndarray arr_swapaxes(np.ndarray arr, char q, char p, char[:] mapping):
    cdef char k, f
    for k in range(len(mapping)):
        f = mapping[k]
        if f >= 0:
            arr = arr.swapaxes(q + k, p + f)
    return arr


cpdef np.ndarray arr_swapaxes_cross(np.ndarray arr, char p, char[:] map1, char[:] map2):
    cdef char k, f, l
    cdef size_t N = len(map1)
    cdef int n
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    for k in range(len(map2)):
        f = map2[k]
        for l in range(N):
            if map1[l] == f:
                n = arr.shape[k]
                idx[l] = n
                idx[p + k] = n
                val = val * np.broadcast_to(np.eye(n), idx)
                idx[l] = 1
                idx[p + k] = 1
                break
        else:
            arr = arr.swapaxes(p + k, p + f)
    return arr
