"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_persistent_ytopt_gen.py
   python3 test_persistent_ytopt_gen.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

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

nworkers, is_manager, libE_specs, _ = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': sim_f,
    'in': ['NUM_THREADS', 'BLOCK_SIZE', 'OMP_PARALLEL'],
    'out': [('RUN_TIME', float)],
}

# Initialize the ytopt ask/tell interface (to be used by the gen_f)
cs = CS.ConfigurationSpace(seed=1234)
NUM_THREADS= CSH.UniformIntegerHyperparameter(name='NUM_THREADS', lower=4, upper=8, default_value=8)
BLOCK_SIZE= CSH.OrdinalHyperparameter(name='BLOCK_SIZE', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
OMP_PARALLEL= CSH.CategoricalHyperparameter(name='OMP_PARALLEL', choices=["#pragma omp parallel for", " "], default_value=' ')
cs.add_hyperparameters([NUM_THREADS, BLOCK_SIZE, OMP_PARALLEL])
input_space = cs
ytoptimizer = Optimizer(
    num_workers=num_sim_workers,
    space=input_space,
    learner='RF',
    liar_strategy='cl_max',
    acq_func='gp_hedge',
)

# Declare the gen_f that will generator points for the sim_f, and the various input/outputs
gen_specs = {
    'gen_f': gen_f,
    'out': [('NUM_THREADS', int, (1,)), ('BLOCK_SIZE', int, (1,)), ('OMP_PARALLEL', "<U50", (1,))],
    'persis_in': ['RUN_TIME', 'NUM_THREADS', 'BLOCK_SIZE', 'OMP_PARALLEL'],
    'user': {
        'ytoptimizer': ytoptimizer,
        'num_sim_workers': num_sim_workers,
    },
}

alloc_specs = {
    'alloc_f': alloc_f,
    'user': {'async_return': True},
}

persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

exit_criteria = {'sim_max': 10}

# Perform the libE run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs)

if is_manager:
    assert np.sum(H['returned']) == exit_criteria['sim_max']
    print("\nlibEnsemble has perform the correct number of evaluations")
    save_libE_output(H, persis_info, __file__, nworkers)
