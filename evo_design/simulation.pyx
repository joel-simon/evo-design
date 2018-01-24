# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

import pickle
import time
import math
import numpy as np
from evo_design.morphogens import MorphogenGrid

cdef void calulate_neighbors(unsigned char[:,:,:] grid, unsigned char[:,:,:] neighbors):
    cdef int x, y, z
    for x in range(1, grid.shape[0]-1):
        for y in range(1, grid.shape[1]-1):
            for z in range(1, grid.shape[2]-1):
                neighbors[x, y, z] = grid[x-1, y, z] + grid[x+1, y, z] + \
                                     grid[x, y-1, z] + grid[x, y+1, z] + \
                                     grid[x, y, z-1] + grid[x, y, z+1]

cpdef object run_simulation(network, int net_depth, dict traits, config, fitness):
    assert net_depth > 0
    cdef int nx, ny, nz, x, y, z, i, step, n_growth, n_death, input_indx, max_i, out_indx, mbin
    cdef double max_o
    nx, ny, nz = config['shape']

    cdef int num_inputs = network.NumInputs()
    cdef int num_outputs = network.NumOutputs()
    cdef int n_signals = config['n_signals']
    cdef int n_memory = config['n_memory']
    cdef int n_morphogens = config['n_morphogens']
    cdef int morphogen_thresholds = config['morphogen_thresholds']

    cdef unsigned char[:,:,:] grid = np.zeros((nx+2, ny+2, nz+2), dtype='uint8')
    cdef unsigned char[:,:,:] neighbors = np.zeros((nx+2, ny+2, nz+2), dtype='uint8')
    cdef double[:,:,:,:] grid_memory = np.zeros((nx+2, ny+2, nz+2, config['n_memory']))
    cdef double[:,:,:,:] signals = np.zeros((nx+2, ny+2, nz+2, config['n_signals']))
    cdef double[:,:,:,:] outputs = np.zeros((nx+2, ny+2, nz+2, num_outputs))
    cdef double[:] output
    cdef double[:] inputs = np.zeros(num_inputs)
    cdef int[:,:] directions = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                                         [0, -1, 0],[1, 0, 0], [-1, 0, 0]], dtype='i')
    cdef int[:] d
    cdef object network_out
    cdef unsigned char[:,:,:] mask = np.ones_like(grid)

    cdef list morphogens = []
    for i in range( config['n_morphogens'] ):
        F = traits[ 'F%i'%i ]
        K = traits[ 'K%i'%i ]
        diffU = traits[ 'diffU%i'%i ]
        diffV = traits[ 'diffV%i'%i ]
        morphogens.append( MorphogenGrid(nx, ny, nz, diffU, diffV, F, K) )

    grid[3:-3, 3:-3, 3:-3] = 1

    cdef int stagnation = 0
    for step in range(config['steps']):
        ###################### Update Simulation #######################
        calulate_neighbors(grid, neighbors)
        for morphogen in morphogens:
            morphogen.gray_scott(200, mask)

        n_growth = 0
        n_death = 0

        ########################### Create Inputs ##############################
        for x in range(1, nx+1):
            for y in range(1, ny+1):
                for z in range(1, nz+1):
                    if grid[ x, y, z ] == 0:
                        continue

                    inputs[:] = 0

                    ############################################################
                    # Basic Inputs
                    inputs[0] = neighbors[x, y, z] / 6.0
                    input_indx = 1

                    # Memory
                    for i in range(n_memory):
                        inputs[input_indx] = grid_memory[x, y, z, i]
                        input_indx += 1

                    # Signals
                    for i in range(n_signals):
                        inputs[input_indx] = (signals[x-1, y, z, i] + signals[x+1, y, z, i] + \
                                              signals[x, y-1, z, i] + signals[x, y+1, z, i] + \
                                              signals[x, y, z-1, i] + signals[x, y, z+1, i]) / 6.0
                        input_indx += 1

                    # Morphogens
                    for morphogen in morphogens:
                        u = morphogen.U[ x, y, z ]
                        mbin = int(math.floor(u * morphogen_thresholds))
                        mbin = min(morphogen_thresholds - 1, mbin)
                        if mbin > 0:
                            inputs[input_indx + (mbin-1)] = 1
                        input_indx += (morphogen_thresholds-1)

                    ############################################################

                    # assert input_indx == num_inputs-1, (input_indx, network.NumInputs())
                    # assert len(inputs.shape) == 1, inputs.shape
                    # assert inputs.shape[0] == num_inputs

                    for i in range(inputs.shape[0]):
                        inputs[i] = (inputs[i]*2)-1 # Map [0, 1] --> [-1, 1]
                    inputs[input_indx] = 1 # Bias

                    network.Flush()
                    network.Input(list(inputs))  # can input numpy arrays, too

                    for _ in range(net_depth):
                        network.ActivateFast()

                    network_out = network.Output()
                    for i in range(num_outputs):
                        outputs[x, y, z, i] = network_out[i]

        ########################################################################
        # Kill cells. So that cells can grow into ones that die that same step.
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if grid[x, y, z] and outputs[x, y, z, 6] > .75:
                        grid[x, y, z] = 0
                        grid_memory[x, y, z, :] = 0
                        n_death += 1

        ############################# Act On Outputs ###########################
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if grid[ x, y, z ] == 0:
                        continue

                    output = outputs[x, y, z]

                    # Find growth direction that is largest.
                    max_o = output[0]
                    max_i = 0
                    for i in range(1, 6):
                        if output[i] > max_o:
                            max_o = output[i]
                            max_i = i

                    if max_o >= 0.75:
                        d = directions[max_i]
                        if not grid[x+d[0], y+d[1], z+d[2]]:
                            grid[x+d[0], y+d[1], z+d[2]] = 1
                            n_growth += 1

                    out_indx = 7

                    for i in range(n_memory):
                        grid_memory[x, y, z, i] = output[ out_indx ]
                        out_indx += 1

                    for i in range(n_signals):
                        signals[x, y, z, i] = output[ out_indx ]
                        out_indx += 1

                    for morphogen in morphogens:
                        if output[out_indx] > .75:
                            morphogen.V[ x, y, z ] = 1

        if n_growth + n_death == 0:
            stagnation += 1
        else:
            stagnation = 0

        if stagnation == config['max_stagnation']:
            break

        if step % config['fitness_bar_period'] == 0:
            if fitness(grid[1:-1, 1:-1, 1:-1]) < (step / float(config['steps'])):
                break

    return np.asarray(grid[1:-1, 1:-1, 1:-1], dtype='uint8')

