import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array


cdef np.ndarray last_diag(np.ndarray arr, int axis1, int axis2):
    return np.diagonal(arr, axis1=axis1, axis2=axis2).\
           swapaxes(-1, axis2 - 1)


cpdef np.ndarray arr_take_diag(np.ndarray arr, int q, int p, mapping):
    cdef int k, f
    k = q
    for f in mapping:
        if f < 0:
            continue
        arr = np.expand_dims(last_diag(arr, k, p + f), k)
        k += 1
    return arr


cpdef np.ndarray arr_expand_diag(np.ndarray arr, int p, array map1, array map2):
    cdef int i, j, n
    cdef int[:] idx = np.ones(arr.ndim, dtype=np.int)
    if map2:
        for i, j in zip(map1, map2):
            if i != j:
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
    if map2:
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
        