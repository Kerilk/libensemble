"""Contains parameter selection and obviation methods for cwpCalibration."""

import numpy as np
import scipy.stats as sps


class thetaprior:
    """Define the class instance of priors provided to the methods."""

    def lpdf(theta):
        """Return log prior density."""
        return (np.sum(sps.norm.logpdf(theta, 1, 0.5), 1)).reshape((len(theta), 1))

    def rnd(n):
        """Return random draws from prior."""
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n, 4))))


def gen_true_theta():
    """Generate one parameter to be the true parameter for calibration."""
    theta0 = np.atleast_2d(np.array([0.5] * 4))

    return theta0


def gen_thetas(n):
    """Generate and return n parameters for the test function."""
    thetas = thetaprior.rnd(n)
    return thetas


def gen_xs(nx, persis_info):
    """Generate and returns n inputs for the modified Borehole function."""
    randstream = persis_info['rand_stream']

    xs = randstream.uniform(0, 1, (nx, 3))
    xs[:, 2] = xs[:, 2] > 0.5

    return xs, persis_info


def gen_observations(fevals, obsvar_const, persis_info):
    """Generate observations."""
    randstream = persis_info['rand_stream']
    n_x = len(np.squeeze(fevals))
    obsvar = np.maximum(obsvar_const*fevals, 5)
    obsvar = np.squeeze(obsvar)
    obs = fevals + randstream.normal(0, np.sqrt(obsvar), n_x).reshape((n_x))

    obs = np.squeeze(obs)
    return obs, obsvar


def select_next_theta(numnewtheta, cal, emu, pending, numexplore):
    # numnewtheta += 2
    thetachoices = cal.theta(numexplore)
    choicescost = np.ones(thetachoices.shape[0])
    thetaneworig, info = emu.supplement(size=numnewtheta, thetachoices=thetachoices,
                                        choicescost=choicescost,
                                        cal=cal, overwrite=True,
                                        args={'includepending': True,
                                              'costpending': 0.01 + 0.99 * np.mean(pending, 0),
                                              'pending': pending})
    thetaneworig = thetaneworig[:numnewtheta, :]
    thetanew = thetaneworig
    return thetanew, info


def obviate_pend_theta(info, pending):
    return pending, info['obviatesugg']
