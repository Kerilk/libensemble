# """
# Runs libEnsemble with uniform random sampling and writes results into sim dirs.
#   tests  per-calculation sim_dir capabilities
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_worker_exceptions.py
#    python3 test_worker_exceptions.py --nworkers 3 --comms local
#    python3 test_worker_exceptions.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np
import os

from libensemble.libE import libE
from libensemble.libE_manager import ManagerException
from libensemble.tests.regression_tests.support import write_func as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

sim_input_dir = './sim_input_dir'
dir_to_copy = sim_input_dir + '/copy_this'
dir_to_symlink = sim_input_dir + '/symlink_this'
e_ensemble = './ensemble_workdirs_w' + str(nworkers) + '_' + libE_specs.get('comms')

for dir in [sim_input_dir, dir_to_copy, dir_to_symlink]:
    if is_master and not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

libE_specs['make_sim_dirs'] = True
libE_specs['sim_dir_path'] = e_ensemble
libE_specs['sim_dirs_per_worker'] = True
libE_specs['sim_dir_copy_files'] = [dir_to_copy]
libE_specs['sim_dir_symlink_files'] = [dir_to_symlink]

libE_specs['abort_on_exception'] = False

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, (1,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3]),
                      'ub': np.array([3]),
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 21}

return_flag = 1
try:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info, libE_specs=libE_specs)
except ManagerException as e:
    print("Caught deliberate exception: {}".format(e))
    return_flag = 0

if is_master:
    assert return_flag == 0
