"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['init_obj']

import numpy as np
# from autotune import TuningProblem
# from autotune.space import *
# import os, sys, time, json, math
# import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
# from skopt.space import Real, Integer, Categorical

from plopper import Plopper


def init_obj(H, persis_info, sim_specs, libE_info):

    params = {}
    for field in sim_specs['in']:
        params[field] = np.squeeze(H[field])

    y = myobj(params, sim_specs['in'], libE_info['workerID'])  # ytopt objective wants a dict
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['RUN_TIME'] = y

    return H_o, persis_info


obj = Plopper('./mmp.c', './')


def myobj(point: dict, fields, workerID):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)
        value = [point[field] for field in fields]
        result = obj.findRuntime(value, fields, workerID)
        return result

    # x = np.array([point[f'p{i}'] for i in range(len(point))])
    x = np.array([point[field] for field in fields])
    results = plopper_func(x)
    # print('CONFIG and OUTPUT', [point, results], flush=True)
    return results
