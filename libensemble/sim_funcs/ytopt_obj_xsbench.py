"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['one_d_example']

import numpy as np
# from autotune import TuningProblem
# from autotune.space import *
# import os, sys, time, json, math
# import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
# from skopt.space import Real, Integer, Categorical

from plopper import Plopper


def one_d_example(H, persis_info, sim_specs, libE_info):
    params = {
        "NUM_THREADS": np.squeeze(H['NUM_THREADS']),
        "BLOCK_SIZE": np.squeeze(H['BLOCK_SIZE']),
        "OMP_PARALLEL": np.squeeze(H['OMP_PARALLEL'])
    }
    y = myobj(params, libE_info['workerID'])  # ytopt objective wants a dict
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['RUN_TIME'] = y

    return H_o, persis_info


# obj = Plopper('./mmm_block.cpp', './')
obj = Plopper('./mmp.c', './')


def myobj(point: dict, workerID):
    def plopper_func(value):
        params = ['NUM_THREADS', 'BLOCK_SIZE', 'OMP_PARALLEL']
        result = obj.findRuntime(value, params, workerID)
        return result

    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x)
    # print('CONFIG and OUTPUT', [point, results], flush=True)
    return results
