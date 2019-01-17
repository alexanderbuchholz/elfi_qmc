import time

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)  # sometimes this is required to enable logging inside Jupyter


# Set an arbitrary seed and a global random state to keep the randomly generated quantities the same between runs
#seed = 42  # this will be separately given to ELFI
#np.random.seed(seed)

def MA2(t_samples, n_obs=200, batch_size=1, random_state=None):
    # Make inputs 2d arrays for numpy broadcasting with w
    t1 = t_samples[:,0]
    t2 = t_samples[:,1]
    t1 = np.asanyarray(t1).reshape((-1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random

    w = random_state.randn(batch_size, n_obs+2)  # i.i.d. sequence ~ N(0,1)
    x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
    return x

# true parameters
t1_true = 0.6
t2_true = 0.2
t_true = np.array([t1_true, t2_true])[np.newaxis,:]
y_obs = MA2(t_true)

# Plot the observed sequence
if False:
    plt.figure(figsize=(11, 6));
    plt.plot(y_obs.ravel());

    # To illustrate the stochasticity, let's plot a couple of more observations with the same true parameters:
    plt.plot(MA2(t_true).ravel());
    plt.plot(MA2(t_true).ravel());
import elfi

# define prior for t1 as in Marin et al., 2012 with t1 in range [-b, b]
class CustomPrior_MC(elfi.Distribution):
    def rvs(b_bound=2, a_bound=1, size=(1,), random_state=None):
        #import ipdb; ipdb.set_trace()
        if not isinstance(size, tuple):
            size = (size,)
        size = size + (2,)
        #import ipdb; ipdb.set_trace()
        u_random = scipy.stats.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        u_1 = u_random[:,0]
        u_2 = u_random[:,1]
        t1 = np.where(u_1<0.5, np.sqrt(2.*u_1)*b_bound-b_bound, -np.sqrt(2.*(1.-u_1))*b_bound+b_bound)
        locs = np.maximum(-a_bound-t1, t1-a_bound)
        scales = a_bound - locs
        t2 = u_2*scales+locs 
        return np.vstack((t1,t2)).transpose()

t_prior_mc = elfi.Prior(CustomPrior_MC)

from qmc_py import sobol_sequence
class CustomPrior_RQMC(elfi.Distribution):
    def rvs(b_bound=2, a_bound=1, size=(1,), random_state=None):
        #import ipdb; ipdb.set_trace()
        if not isinstance(size, tuple):
            size = (size,)
        size = size + (2,)
        #import ipdb; ipdb.set_trace()
        if random_state is None:
            random_state = np.random.randint(10**5)
            u_random = sobol_sequence(N=size[0], DIMEN=size[1], IFLAG=1, iSEED=random_state)
        else: 
            u_random = sobol_sequence(N=size[0], DIMEN=size[1], IFLAG=1, iSEED=random_state.randint(10**5))
        #u_random = scipy.stats.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        u_1 = u_random[:,0]
        u_2 = u_random[:,1]
        t1 = np.where(u_1<0.5, np.sqrt(2.*u_1)*b_bound-b_bound, -np.sqrt(2.*(1.-u_1))*b_bound+b_bound)
        locs = np.maximum(-a_bound-t1, t1-a_bound)
        scales = a_bound - locs
        t2 = u_2*scales+locs 
        return np.vstack((t1,t2)).transpose()

def autocov(x, lag=1):
    C = np.mean(x[:,lag:] * x[:,:-lag], axis=1)
    return C


#CustomPrior_RQMC.rvs(size=100)
t_prior_rqmc = elfi.Prior(CustomPrior_RQMC)
#import ipdb; ipdb.set_trace()

def new_sample(MA2, t_prior, t_prior_name, N = 500, y_obs=y_obs):
    # ELFI also supports giving the scipy.stats distributions as strings
    Y = elfi.Simulator(MA2, t_prior, observed=y_obs)
    S1 = elfi.Summary(autocov, Y)
    S2 = elfi.Summary(autocov, Y, 2)  # the optional keyword lag is given the value 2
    d = elfi.Distance('euclidean', S1, S2)
    rej = elfi.Rejection(d, batch_size=5000, seed=np.random.randint(10**5))
    
    result = rej.sample(N, quantile=0.01)
    return result.samples[t_prior_name].mean(axis=0)
M_rep = 20
mc_list = []
rqmc_list = []
for mrep in range(M_rep):
    mc_list.append(new_sample(MA2, t_prior_mc, "t_prior_mc"))
    rqmc_list.append(new_sample(MA2, t_prior_rqmc, "t_prior_rqmc"))
print(np.array(mc_list).var(axis=0))
print(np.array(rqmc_list).var(axis=0))
#plt.scatter(result.samples['t_prior'][:,0], result.samples['t_prior'][:,1])
import ipdb; ipdb.set_trace()
