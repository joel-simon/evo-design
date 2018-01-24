cimport numpy as np
import numpy as np

cdef class MorphogenGrid:
    cdef public int nx, ny, nz
    cdef public double diffU, diffV, F, K
    cdef public double[:,:,:] U, V, dU, dV

    cpdef void setV(self, int x, int y, int z, double v) except *
    cpdef double[:,:,:] gray_scott(self, int steps, unsigned char[:,:,:] mask) except *
