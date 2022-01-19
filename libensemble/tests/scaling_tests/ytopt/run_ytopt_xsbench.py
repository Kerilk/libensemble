"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_persistent_ytopt_gen.py
   python3 test_persistent_ytopt_gen.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.ytopt_obj_xsbench import init_obj as sim_f
from libensemble.gen_funcs.ytopt_gen_xsbench import persistent_ytopt as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ytopt.search.optimizer import Optimizer

nworkers, is_manager, libE_specs, _ = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': sim_f,
    'in': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
    'out': [('RUN_TIME', float)],
}

cs = CS.ConfigurationSpace(seed=1234)
# Initialize the ytopt ask/tell interface (to be used by the gen_f)
p0 = CSH.OrdinalHyperparameter(name='p0', sequence=[2, 3, 4, 5, 6, 7, 8], default_value=8)
# block size for openmp dynamic schedule
p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[10, 20, 40, 64, 80, 100, 128, 160, 200], default_value=100)
# clang unrolling
p2 = CSH.CategoricalHyperparameter(name='p2', choices=["#pragma clang loop unrolling full", " "], default_value=' ')
# omp parallel
p3 = CSH.CategoricalHyperparameter(name='p3', choices=["#pragma omp parallel for", " "], default_value=' ')
# tile size for one dimension for 2D tiling
p4 = CSH.OrdinalHyperparameter(name='p4', sequence=[2, 4, 8, 16, 32, 64, 96, 128, 256], default_value=96)
# tile size for another dimension for 2D tiling
p5 = CSH.OrdinalHyperparameter(name='p5', sequence=[2, 4, 8, 16, 32, 64, 96, 128, 256], default_value=256)
# omp placement
p6 = CSH.CategoricalHyperparameter(name='p6', choices=['cores', 'threads', 'sockets'], default_value='cores')
p7 = CSH.CategoricalHyperparameter(name='p7',
                                   choices=['compact', 'scatter', 'balanced', 'none', 'disabled', 'explicit'],
                                   default_value='none')

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7])
ytoptimizer = Optimizer(
    num_workers=num_sim_workers,
    space=cs,
    learner='RF',
    liar_strategy='cl_max',
    acq_func='gp_hedge',
)

# Declare the gen_f that will generator points for the sim_f, and the various input/outputs
gen_specs = {
    'gen_f': gen_f,
    'out': [('p0', int, (1,)), ('p1', int, (1,)), ('p2', "<U34", (1,)), ('p3', "<U24", (1,)),
            ('p4', int, (1,)), ('p5', int, (1,)), ('p6', "<U7", (1,)), ('p7', "<U8", (1,)), ],
    'persis_in': sim_specs['in'] + ['RUN_TIME'],
    'user': {
        'ytoptimizer': ytoptimizer,
        'num_sim_workers': num_sim_workers,
    },
}

alloc_specs = {
    'alloc_f': alloc_f,
    'user': {'async_return': True},
}

exit_criteria = {'sim_max': 100}

# Perform the libE run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs)

if is_manager:
    assert np.sum(H['returned']) == exit_criteria['sim_max']
    print("\nlibEnsemble has perform the correct number of evaluations")
    save_libE_output(H, persis_info, __file__, nworkers)
