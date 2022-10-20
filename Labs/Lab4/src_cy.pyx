# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
cimport cython
import numpy as np
cimport numpy as cnp



cpdef float KNN_c(double[:, :] dataset, double[:, :] dataset_t, double[:] target,  double[:] target_test, int k):
    classes = cvarray(shape=(400, ), itemsize=sizeof(double), format="d", mode='c')
    cdef double [:] classes_view = classes

    distances = cvarray(shape=(400, ), itemsize=sizeof(double), format="d", mode='c')
    cdef double [:] distances_view = distances
    nearest_ind = cvarray(shape=(100, ), itemsize=sizeof(double), format="d", mode='c')
    cdef double [:] nearest_ind_view = nearest_ind

    x = cvarray(shape=(2, ), itemsize=sizeof(double), format="d", mode='c')
    cdef double [:] x_view = x
    y = cvarray(shape=(2, ), itemsize=sizeof(double), format="d", mode='c')
    cdef double [:] y_view = y
    cdef double min_distance
    cdef double item, classe
    cdef int i, j, p, a, ind_iter, min_ind, error, ind
    i = 0
    classes_view = np.zeros(target.shape[0])
    while i < dataset.shape[0]:
        x_view = dataset[i]
        j = 0
        nearest_ind_view = np.zeros(k)
        distances_view = np.zeros(dataset_t.shape[0], dtype=np.double)
        while j < dataset_t.shape[0]:
            y_view = dataset_t[j]
            for coord in range(y_view.shape[0]):
                distances_view[j] += (y_view[coord] - x_view[coord])**2
            j = j + 1
        p = 0
        while p < k:
            min_distance = min(distances_view)
            # get the index of the min
            ind_iter = 0
            while ind_iter < len(distances_view):
                if distances_view[ind_iter] == min_distance:
                    min_ind = ind_iter
                    break

                ind_iter = ind_iter + 1
            # get the target
            target_min = target[min_ind]
            nearest_ind_view[p] = target_min
            # remove the target and the distance from the corresponding lists
            distances_view[min_ind] = max(distances_view)
            p = p + 1
        classe = 2
        a = 0

        for item in nearest_ind_view:
            if item == 1:
                a = a + 1
        if a > k//2:
            classe = 1
        classes_view[i] = classe
        i = i + 1
    ind = 0
    error = 0
    while ind < target_test.shape[0]:
        if target_test[ind] != classes_view[ind]:
            error = error + 1
        ind = ind + 1
    return error/target_test.shape[0]
