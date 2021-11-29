"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_1d_sampling.py
   python3 test_1d_sampling.py --nworkers 3 --comms local
   python3 test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.ytopt_obj import one_d_example as sim_f
from libensemble.gen_funcs.ytopt_gen import persistent_ytopt as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util

nworkers, is_manager, libE_specs, _ = parse_args()
num_sim_workers = nworkers - 1 # Subtracting one because one worker will be the generator

sim_specs = {
    'sim_f': sim_f,
    'in': ['BLOCK_SIZE'],
    'out': [('RUN_TIME', float)],
}


cs = CS.ConfigurationSpace(seed=1234)
p0= CSH.UniformIntegerHyperparameter(name='BLOCK_SIZE', lower=1, upper=10, default_value=5)
cs.add_hyperparameters([p0])
input_space = cs

ytoptimizer = Optimizer(
    num_workers=num_sim_workers, 
    space=input_space,
    learner='RF', liar_strategy='cl_max', acq_func='gp_hedge')

gen_specs = {
    'gen_f': gen_f,
    'out': [('BLOCK_SIZE', int, (1,))],
    'persis_in': ['RUN_TIME', 'BLOCK_SIZE'],
    'user': {'ytoptimizer': ytoptimizer,
             'num_sim_workers': num_sim_workers,
        },
}

alloc_specs = {'alloc_f': alloc_f}

persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

exit_criteria = {'sim_max': 10}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs)

if is_manager:
    assert np.sum(H['returned']) == exit_criteria['sim_max']
    print("\nlibEnsemble has perform the correct number of evaluations")
    save_libE_output(H, persis_info, __file__, nworkers)
