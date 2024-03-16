import numpy as np

import tqdm

from imu_relpose_estim.utils.nextxcov_helper import NextStateCovarianceHelper
from imu_relpose_estim.utils.noisecov_helper import NoiseCovarianceHelper
from imu_relpose_estim.utils.dataclasses import ObservationData, StateData
from bingham.math.sampler import BinghamSampler

import argparse

parser = argparse.ArgumentParser(
                    prog='Test for nextxcov_helper',
                    description='test covariance / mean calculation based on Monte Carlo method',)

parser.add_argument('--dt', type=float, default=0.05,
                    help='value of dt')
parser.add_argument('--iter', type=int, default=100,
                    help='iteration number of sampling')
parser.add_argument('--sample', type=int, default=100,
                    help='sampling number of sampling per iteration')
parser.add_argument('--seed', type=int, default=None,
                    help='sampling number of sampling per iteration')
args = parser.parse_args()



def generate_random_obs(dt):
    dim=3
    gparams = [0] * 4
    for i in range(4):
        sqrtA = np.random.randn(dim,dim)
        cov = np.dot(sqrtA, sqrtA.T)
        mu = np.random.randn(dim)
        gparams[i] = {"sqrtA": sqrtA, "mu": mu, "cov": cov}

    obs_true = ObservationData(dt,
                               gparams[0]["mu"], gparams[1]["mu"], gparams[2]["mu"], gparams[3]["mu"],
                               gparams[0]["cov"], gparams[1]["cov"], gparams[2]["cov"], gparams[3]["cov"])
    param_for_sampling = {"f": {"mu": gparams[0]["mu"], "sqrtA": gparams[0]["sqrtA"]},
                          "df": {"mu": gparams[1]["mu"], "sqrtA": gparams[1]["sqrtA"]},
                          "g": {"mu": gparams[2]["mu"], "sqrtA": gparams[2]["sqrtA"]},
                          "dg": {"mu": gparams[3]["mu"], "sqrtA": gparams[3]["sqrtA"]},}
    return obs_true, param_for_sampling


def generate_random_states():
    dim=6
    x_sqrtcov = np.random.randn(dim,dim)
    x_cov = np.dot(x_sqrtcov, x_sqrtcov.T)
    x_mu = np.random.randn(dim)

    A = np.random.randn(4,4)
    A = A + A.T
    bingham_Z, bingham_M = np.linalg.eigh(A)
    ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(bingham_Z, N_nc=50)
    Eq4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(bingham_M, ddnc_10D)
    return Eq4th, A, x_sqrtcov, x_cov, x_mu


def sampling_realized(dt,
                      obsv_param_for_sampling, bingham_A,
                      x_sqrtcov, x_mu, N=1000, sample_only_obs=False):
    samples = [0]*4
    for i,k in enumerate(["f", "df", "g", "dg"]):
        covsqrt = obsv_param_for_sampling[k]["sqrtA"]
        mu = obsv_param_for_sampling[k]["mu"]
        samples[i] = np.dot(covsqrt, np.random.randn(3, N)).T + mu
    
    obsdata = [ObservationData.realized_data(dt, s,t,u,v) for s,t,u,v in zip(*samples)]

    if sample_only_obs:
        return obsdata, None, None
    
    ## states

    states = np.dot(x_sqrtcov, np.random.randn(6, N)).T + x_mu

    bs = BinghamSampler(dim=3)
    bingham_samples = bs.sampling_from_bingham(A=bingham_A, sampling_N=N)

    Eq4th_realizes = []
    for s in bingham_samples:
        bingham_realize_M = np.eye(4)
        bingham_realize_M[:,0] = s
        bingham_realize_Z = np.array([1e+15, 0, 0, 0])
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(bingham_realize_Z, N_nc=50)
        Eq4th_realizes.append(NoiseCovarianceHelper.calc_Bingham_4thMoment(bingham_realize_M, ddnc_10D))
    
    return obsdata, states, Eq4th_realizes


def calc_montecarlo(dt,
                    state_sqrtcov, state_mu, 
                    bingham_A, 
                    param_for_sampling_i, param_for_sampling_j, 
                    N=10000):

    realized_i, realized_states, realized_Eq4ths = sampling_realized(dt, param_for_sampling_i, bingham_A, state_sqrtcov, state_mu, N=N)
    realized_j, _, _ = sampling_realized(dt, param_for_sampling_j, bingham_A, state_sqrtcov, state_mu, N=N, sample_only_obs=True)

    realized_nextx_list = []

    for sts, eq4th, robs_i, robs_j in zip(realized_states, realized_Eq4ths, realized_i, realized_j):
        states = StateData(sts, np.zeros((6,6)), eq4th)
        realized_nextx_list.append(
            NextStateCovarianceHelper.calc_E_next_x(states,
                                                    robs_i, robs_j))

    nextxs = np.array(realized_nextx_list)

    mean_numerial = np.mean(nextxs, axis=0)
    cov_numerical = np.cov(nextxs.T)

    return mean_numerial, cov_numerical


def update_mean_and_cov(old_xbar, old_cov,
                        append_xbar, append_cov,
                        N, accumulate_N):
    M = accumulate_N
    new_xbar = (N*append_xbar + M*old_xbar) / (N+M)
    new_cov = 1/(N+M) * \
            (N* (append_cov + np.outer(append_xbar, append_xbar))\
            + M*(old_cov + np.outer(old_xbar, old_xbar))) \
            - np.outer(new_xbar, new_xbar)
    
    return new_xbar, new_cov


def calculate_montecarlo_estimation(dt, N_iter, N_sample):
    ## generate observation
    obs_true_i, param_for_sampling_i = generate_random_obs(dt)
    obs_true_j, param_for_sampling_j = generate_random_obs(dt)
    Eq4th, bingham_A, state_sqrtcov, state_cov, state_mu = generate_random_states()

    ## initialize
    N = N_sample
    accumulate_N = N
    old_mean, old_cov = calc_montecarlo(dt, state_sqrtcov, state_mu, 
                                        bingham_A, 
                                        param_for_sampling_i, param_for_sampling_j,
                                        N)

    for _ in tqdm.tqdm(range(N_iter)):
        mean_numerial_append, cov_numerical_append = calc_montecarlo(dt,
                                                        state_sqrtcov, state_mu, 
                                                        bingham_A, 
                                                        param_for_sampling_i, param_for_sampling_j, N)

        accumulate_N += N
        new_xbar, new_cov = update_mean_and_cov(old_mean, old_cov,
                                                mean_numerial_append, cov_numerical_append,
                                                N, accumulate_N)
        old_mean = new_xbar
        old_cov = new_cov

    states = StateData(state_mu, state_cov, Eq4th)

    mean_analytical = NextStateCovarianceHelper.calc_E_next_x(states,
                                                obs_true_i, obs_true_j)

    cov_analytical = NextStateCovarianceHelper.calc_nextstate_cov(states,
                                            obs_true_i, obs_true_j)
    
    return old_mean, mean_analytical, old_cov, cov_analytical


if __name__ == "__main__":
    ## arguments
    dt = args.dt
    N_iter = args.iter
    N_sample = args.sample
    if not args.seed is None:
        np.random.seed(seed=args.seed)

    print(args)

    ## calculate for test
    mean_numerical, mean_analytical, cov_numerical, cov_analytical = calculate_montecarlo_estimation(dt, N_iter, N_sample)

    print("numerical mean :\n", mean_numerical)
    print("analytical mean:\n", mean_analytical)
    print("---")
    print("numerical cov :\n", cov_numerical)
    print("analytical cov:\n", cov_analytical)
    print("---")
    Z_numerical, M_numerical = np.linalg.eigh(cov_numerical)
    Z_analytical, M_analytical = np.linalg.eigh(cov_analytical)
    print("numerical cov eigvals :\n", Z_numerical)
    print("analytical cov eigvals:\n", Z_analytical)
    print("numerical cov eigvects :\n", M_numerical)
    print("analytical cov eigvects:\n", M_analytical)
    print("---")
    print("diff (analytical - numerical)")
    print("mean: \n", mean_analytical - mean_numerical)
    print("cov : \n", cov_analytical - cov_numerical)

