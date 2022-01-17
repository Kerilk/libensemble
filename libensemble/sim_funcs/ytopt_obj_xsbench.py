"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['init_obj']

import numpy as np
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


def myobj(point: dict, params, workerID):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)
        value = [point[param] for param in params]
        params = [i.upper() for i in params]
        result = obj.findRuntime(value, params, workerID)
        return result

    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x)
    # print('CONFIG and OUTPUT', [point, results], flush=True)
    return results
