# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_uniform_sampling.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

import sys, os             # for adding to path
import numpy as np

if len(sys.argv) > 1 and sys.argv[1] == "--threads":
    from libensemble.libE_thread import libE
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    is_master = True
    libE_specs = {'nworkers': nworkers}
elif len(sys.argv) > 1 and sys.argv[1] == "--processes":
    from libensemble.libE_process import libE
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    is_master = True
    libE_specs = {'nworkers': nworkers}
else:
    from mpi4py import MPI #
    from libensemble.libE import libE
    nworkers = MPI.COMM_WORLD.Get_size()-1
    is_master = MPI.COMM_WORLD.Get_rank() == 0
    libE_specs = {'comm': MPI.COMM_WORLD, 'color': 0}

# Import sim_func
from libensemble.sim_funcs.six_hump_camel import six_hump_camel

# Import gen_func
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample

script_name = os.path.splitext(os.path.basename(__file__))[0]

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'save_every_k': 400
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2)],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 500,
             'save_every_k': 300
             }

# Tell libEnsemble when to stop
exit_criteria = {'gen_max': 501}

np.random.seed(1)
persis_info = {}
for i in range(1,nworkers+1):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_master:
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(nworkers+1)
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

    minima = np.array([[ -0.089842,  0.712656],
                       [  0.089842, -0.712656],
                       [ -1.70361,  0.796084],
                       [  1.70361, -0.796084],
                       [ -1.6071,   -0.568651],
                       [  1.6071,    0.568651]])
    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x']-m)**2,1)) < tol

    print("\nlibEnsemble with Uniform random sampling has identified the 6 minima within a tolerance " + str(tol))


