# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
cimport cython
import numpy as np
cimport numpy as cnp


cpdef double[:] KNN_c(double[:, :] dataset, double[:, :] dataset_t, double[:] target, int k):
    classes = cvarray(shape=(400, ), itemsize=sizeof(double), format="i")
    cdef double [:] classes_view = classes
    distances = cvarray(shape=(400, ), itemsize=sizeof(double), format="i")
    cdef double [:] distances_view = distances
    cdef cnp.int_t classe
    nearest_ind = cvarray(shape=(100, ), itemsize=sizeof(long), format="i")
    cdef long [:] nearest_ind_view = nearest_ind
    
    x = cvarray(shape=(2, ), itemsize=sizeof(double), format="i")
    cdef double [:] x_view = x
    y = cvarray(shape=(2, ), itemsize=sizeof(double), format="i")
    cdef double [:] y_view = y
    cdef int i, j, p, a, ind_dist, classe
    i = 0
    while i < dataset.shape[0]:
        x_view = dataset[i]
        distances_view = np.zeros(dataset_t.shape[0])

        # loop to get the list of distances
        j = 0
        while j < dataset_t.shape[0]:
            y_view = dataset_t[i]
            for coord in range(y_view.shape[0]):
                distances_view[j] += (y_view[coord] - x_view[coord])**2 
        # loop to find the list of minimum and the target
        p = 0
        while p < k:
            min_distance = distances_view.index(min(distances_view))
            # get the index of the min
            ind_dist = 0
            while ind_dist < len(distances_view):
                if distances_view[ind_dist] == min(distances_view):
                    min_ind = ind_dist
                    break
                ind_dist = ind_dist + 1
            # get the target
            target_min = target[ind_dist]
            nearest_ind_view.append(target_min)
            # remove the target and the distance from the corresponding lists 
            distances_view.pop(min_distance)
            target_min.pop(min_distance)
            p = p + 1
        
        # get the class count and return the most recurrent one
        classe = 1
        for item in nearest_ind_view:
            a = 0
            for s in nearest_ind_view:
                if s == x:
                    a = a + 1
            if a > classe:
                classe = a
        classes_view[i] = classe
    return classes_view

cpdef float error_rate_c(double[:,:] dataset, double[:] target_test, double[:] target_train, double[:, :] dataset_t, int K):
    y_pred = cvarray(shape=(400, ), itemsize=sizeof(double), format="i")
    cdef double [:] pred_view = y_pred
    pred_view = KNN_c(dataset, dataset_t, target_train, K)
    cdef int error, ind
    ind = 0
    while ind < target_test.shape[0]:
        if target_test[ind] != pred_view[ind]:
            error = error + 1
        ind = ind + 1
    return error/ind
