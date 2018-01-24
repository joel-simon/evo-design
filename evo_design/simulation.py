import pickle
import time
import math
import numpy as np
from evo_design.morphogens import MorphogenGrid

def calulate_neighbors(grid, neighbors):
    for x in range(1, grid.shape[0]-1):
        for y in range(1, grid.shape[1]-1):
            for z in range(1, grid.shape[2]-1):
                neighbors[x, y, z] = grid[x-1, y, z] + grid[x+1, y, z] + \
                                     grid[x, y-1, z] + grid[x, y+1, z] + \
                                     grid[x, y, z-1] + grid[x, y, z+1]

def run_simulation(network, net_depth, traits, config, fitness):
    assert net_depth > 0
    nx, ny, nz = config['shape']

    n_signals = config['n_signals']
    n_memory = config['n_memory']
    n_morphogens = config['n_morphogens']
    morphogen_thresholds = config['morphogen_thresholds']

    grid = np.zeros((nx+2, ny+2, nz+2), dtype='i')
    grid_memory = np.zeros((nx+2, ny+2, nz+2, config['n_memory']))
    signals = np.zeros((nx+2, ny+2, nz+2, config['n_signals']))
    neighbors = np.zeros((nx+2, ny+2, nz+2), dtype='uint8')

    inputs = np.zeros(network.NumInputs())
    outputs = np.zeros((nx+2, ny+2, nz+2, network.NumOutputs()))

    morphogens = []
    for i in range( config['n_morphogens'] ):
        F = traits[ 'F%i'%i ]
        K = traits[ 'K%i'%i ]
        diffU = traits[ 'diffU%i'%i ]
        diffV = traits[ 'diffV%i'%i ]
        morphogens.append( MorphogenGrid(nx, ny, nz, diffU, diffV, F, K) )


    grid[3:-3, 3:-3, 3:-3] = 1

    stagnation = 0
    for step in range(config['steps']):
        ###################### Update Simulation #######################
        calulate_neighbors(grid, neighbors)
        for morphogen in morphogens:
            morphogen.gray_scott(200, grid)

        n_growth = 0
        n_death = 0

        ########################### Create Inputs ##############################
        for x in range(1, nx+1):
            for y in range(1, ny+1):
                for z in range(1, nz+1):
                    if grid[ x, y, z ] == 0:
                        continue

                    inputs.fill(0)

                    ############################################################
                    # Basic Inputs
                    n_neighbors = neighbors[x, y, z]
                    inputs[0] = n_neighbors / 6.0
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

                    assert input_indx == network.NumInputs()-1, (input_indx, network.NumInputs())

                    inputs = (inputs*2)-1 # Map [0, 1] --> [-1, 1]
                    inputs[input_indx] = 1 # Bias

                    network.Flush()
                    network.Input(inputs)  # can input numpy arrays, too

                    for _ in range(net_depth):
                        network.ActivateFast()

                    outputs[x, y, z] = network.Output()

        ########################################################################
        # Kill cells. So that cells can grow into ones that die that same step.
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if grid[x, y, z] and outputs[x, y, z, 6] > .75:
                        grid[x, y, z] = 0
                        grid_memory[x, y, z] = 0
                        n_death += 1

        ############################# Act On Outputs ###########################
        directions = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                               [0, -1, 0],[1, 0, 0], [-1, 0, 0]])
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

    return grid[1:-1, 1:-1, 1:-1]

