# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

cimport numpy as np
import numpy as np

from cymesh.structures cimport Vert

cdef class MorphogenGrid:
    def __init__(self, nx, ny, nz, diffU, diffV, F, K):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.diffU = diffU
        self.diffV = diffV
        self.F = F
        self.K = K
        self.U = np.zeros((nx+2, ny+2, nz+2))
        self.V = np.zeros((nx+2, ny+2, nz+2))
        self.dU = np.zeros((nx+2, ny+2, nz+2))
        self.dV = np.zeros((nx+2, ny+2, nz+2))

    cpdef void setV(self, int x, int y, int z, double v) except *:
        self.V[x+1, y+1, z+1] = v

    cpdef double[:,:,:] gray_scott(self, int steps, int[:,:,:] mask) except *:
        cdef int x, y, z
        cdef double uvv, u, v, lapU, lapV
        cdef double FK = self.F + self.K

        assert (mask.shape[0], mask.shape[1], mask.shape[2]) == \
                                              (self.nx+2, self.ny+2, self.nz+2)

        np.asarray(self.U).fill(0)

        for _ in range(steps):
            for x in range(1, self.nx+1):
                for y in range(1, self.ny+1):
                    for z in range(1, self.nz+1):
                        if not mask[x, y, z]:
                            continue

                        u = self.U[x, y, z]
                        v = self.V[x, y, z]

                        uvv = u*v*v
                        lapU = -(6*u)
                        lapV = -(6*v)

                        lapU += self.U[x-1, y, z] + self.U[x+1, y, z] + \
                                self.U[x, y-1, z] + self.U[x, y+1, z] + \
                                self.U[x, y, z-1] + self.U[x, y, z+1]

                        lapV += self.V[x-1, y, z] + self.V[x+1, y, z] + \
                                self.V[x, y-1, z] + self.V[x, y+1, z] + \
                                self.V[x, y, z-1] + self.V[x, y, z+1]

                        self.dU[x, y, z] = self.diffU*lapU - uvv + self.F*(1-u)
                        self.dV[x, y, z] = self.diffV*lapV + uvv - FK*v

                        # i += 1

            for x in range(1, self.nx+1):
                for y in range(1, self.ny+1):
                    for z in range(1, self.nz+1):
                        if mask[x, y, z]:
                            self.U[x, y, z] += self.dU[x, y, z]
                            self.V[x, y, z] += self.dV[x, y, z]

        return self.U[1:-1, 1:-1, 1:-1]

