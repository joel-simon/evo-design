import os, sys, time, random, string
import numpy as np
from datetime import datetime
import MultiNEAT as NEAT
sys.path.append(os.path.abspath('.'))
from evo_design.map_utils import largest_contiguous
from evo_design.simulation import run_simulation
from evo_design.classification import balanced_accuracy_score_np
from evo_design.morphogens import MorphogenGrid

def simulate_genome(genome, config):
    """
    """
    network = NEAT.NeuralNetwork()
    traits = genome.GetGenomeTraits()
    genome.BuildPhenotype(network)
    genome.CalculateDepth()
    depth = genome.GetDepth()
    morphogens = []

    def fitness(grid):
        return balanced_accuracy_score_np(pred=grid, true=config['target'])

    grid = run_simulation(network, depth, traits, config, fitness)
    return fitness(grid), grid

def addTrait(params, name, vrange, ttype='float'):
    trait = {
        'details': {
            'max': max(vrange),
            'min': min(vrange),
            'mut_power': abs(vrange[1] - vrange[0]) / 4,
            'mut_replace_prob': 0.1
        },
        'importance_coeff': 1.0,
        'mutation_prob': 0.3,
        'type': ttype
    }
    params.SetGenomeTraitParameters(name, trait)

def optimize_neat(config, n_inputs, n_hidden, n_outputs, out_dir):
    print('Starting Optimization')
    params = NEAT.Parameters()
    params.PopulationSize = 60
    params.OldAgeTreshold = 10
    params.SpeciesMaxStagnation = 20
    params.AllowLoops = False

    for i in range(config['n_morphogens']):
        addTrait(params, 'K%i'%i, (.03, .08))
        addTrait(params, 'F%i'%i, (.01, .06))
        addTrait(params, 'diffU%i'%i, (.005, .02))
        addTrait(params, 'diffV%i'%i, (.0025, .01))

    ######################## Create NEAT objects ###############################
    fs_neat = False
    seed_type = 0
    out_type = NEAT.ActivationFunction.UNSIGNED_SIGMOID
    hidden_type = NEAT.ActivationFunction.UNSIGNED_SIGMOID
    genome_prototye = NEAT.Genome(0, n_inputs, n_hidden, n_outputs, fs_neat, \
                                  out_type, hidden_type, seed_type, params, 0)
    rand_seed = int(time.time())
    pop = NEAT.Population( genome_prototye, params, True, 1.0, rand_seed )

    ######################## Main evolution loop ###############################
    top_fitness = 0 # Fitness function is defined in [0, 1]
    top_grid = None

    for generation in range(config['generations']):
        print('Starting generation', generation)
        genomes = NEAT.GetGenomeList(pop)
        fitness_list = [ simulate_genome(g, config)[0] for g in genomes ]
        NEAT.ZipFitness(genomes, fitness_list)
        max_fitness = max(fitness_list)

        print('Generation complete')
        print('Max fitness', max_fitness)
        print('Mean fitness', np.mean(fitness_list))

        if max_fitness > top_fitness:
            print('New best fitness')
            best_genome = genomes[ fitness_list.index(max_fitness) ]
            _, best_grid = simulate_genome(best_genome, config)

            top_fitness = max_fitness
            top_grid = best_grid

            np.save(out_dir+'/grid_%i' % generation, best_grid)
            best_genome.Save(out_dir+'/genome_%i' % generation)

        pop.Epoch()
        print()

def generate_id(n):
    options = string.ascii_uppercase + string.digits
    return ''.join(random.choice(options) for _ in range(n))

def main(config):
    assert config['shape'] == config['target'].shape

    rid = generate_id(4)
    dir_name = 'outputs/' + rid +"__{:%B_%d_%Y_%H_%M}".format(datetime.now())
    os.makedirs(dir_name)

    # Base inputs are n_neighbors and bias.
    in_per_morph = (config['morphogen_thresholds']-1)
    n_inputs = 2 + config['n_morphogens'] * in_per_morph + config['n_memory'] + config['n_signals']
    n_hidden = 0
    # Base outputs are 6 growth directions and apoptosis.
    n_outputs = 7 + config['n_memory'] + config['n_signals'] + config['n_morphogens']

    optimize_neat(config, n_inputs, n_hidden, n_outputs, dir_name)

target = np.ones((9, 9, 9), dtype='uint8')
r = target.shape[0] // 3
target[r:-r, :, r:-r] = 0
target[:, r:-r, r:-r] = 0
target[r:-r, r:-r, :] = 0

main({
    'target': target,
    'generations': 100,
    'steps': 50,
    'shape': (9, 9, 9),
    'seed_shape': (5, 5, 5),
    'n_memory': 2,
    'n_signals': 2,
    'n_morphogens': 0,
    'morphogen_thresholds': 2,
    'max_stagnation': 5,
    'fitness_bar_period': 25,
    'neat_params_path': 'neatconfig.txt',
    'save_growth': False
})