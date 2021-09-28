import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def only_persistent_gens_for_inverse_bayes(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    Starts up to gen_count number of persistent generators.
    These persistent generators produce points (x) in batches and subbatches.
    The points x are given in subbatches to workers to perform a calculation.
    When all subbatches have returned, their output is given back to the
    corresponding persistent generator.

    The first time called there are no persis_w 1st for loop is not done
    """

    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})
    manage_resources = 'resource_sets' in H.dtype.names

    support = AllocSupport(W, manage_resources, persis_info, sched_opts)
    Work = {}
    gen_count = support.count_persis_gens()

    # If wid is idle, but in persistent mode, and generated work has all returned
    # give output back to wid. Otherwise, give nothing to wid
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG):

        # if > 1 persistant generator, assign the correct work to it
        inds_generated_by_wid = (H['gen_worker'] == wid)
        if support.all_returned(H, inds_generated_by_wid):

            # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            batch_ids = H['batch'][inds_generated_by_wid]
            last_batch_inds = (batch_ids == np.max(batch_ids))
            inds_to_send_back = np.where(np.logical_and(inds_generated_by_wid,
                                                        last_batch_inds))[0]
            if H['batch'][-1] > 0:
                n = gen_specs['user']['subbatch_size']*gen_specs['user']['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]

            Work[wid] = support.gen_work(wid, ['like'], inds_to_send_back,
                                         persis_info.get(wid), persistent=True)

    points_to_evaluate = ~H['given'] & ~H['cancel_requested']
    for wid in support.avail_worker_ids(persistent=False):
        if np.any(points_to_evaluate):

            # perform sim evaluations (if any point hasn't been given).
            sim_subbatches = H['subbatch'][points_to_evaluate]
            sim_inds = (sim_subbatches == np.min(sim_subbatches))
            sim_ids_to_send = np.nonzero(points_to_evaluate)[0][sim_inds]

            try:
                Work[wid] = support.sim_work(wid, H, sim_specs['in'], sim_ids_to_send, [])
            except InsufficientFreeResources:
                break
            points_to_evaluate[sim_ids_to_send] = False

        elif gen_count == 0:

            # Finally, generate points since there is nothing else to do.
            try:
                Work[wid] = support.gen_work(wid, gen_specs['in'], [], persis_info.get(wid),
                                             persistent=True)
            except InsufficientFreeResources:
                break
            gen_count += 1
    del support
    return Work, persis_info
