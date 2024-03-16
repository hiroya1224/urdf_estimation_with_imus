import numpy as np

from imu_relpose_estim.estimator.leastsq import EstimateImuRelativePoseLeastSquare
from imu_relpose_estim.utils.dataclasses import ObservationData, StateDataLeastSquare
from imu_relpose_estim.utils.rotation_helper import RotationHelper


# import argparse

# parser = argparse.ArgumentParser(
#                     prog='Test for nextxcov_helper',
#                     description='test covariance / mean calculation based on Monte Carlo method',)

# parser.add_argument('--dt', type=float, default=0.05,
#                     help='value of dt')
# parser.add_argument('--iter', type=int, default=100,
#                     help='iteration number of sampling')
# parser.add_argument('--sample', type=int, default=100,
#                     help='sampling number of sampling per iteration')
# parser.add_argument('--seed', type=int, default=None,
#                     help='sampling number of sampling per iteration')
# args = parser.parse_args()


def generate_random_states():
    ## initialize
    state = StateDataLeastSquare.empty()

    ## translation parameter
    position = np.random.randn(3)
    sqrtC = np.random.randn(3,3)
    position_cov = np.dot(sqrtC, sqrtC.T)

    ## bingham parameter
    A = np.random.randn(4,4) * 100.
    A = A + A.T

    covQinv = np.random.randn(4,4)
    covQinv = np.dot(covQinv.T, covQinv)

    ## update
    return state.update(position, position_cov, A, covQinv)


def generate_random_obs():
    def generate_psd_matrix():
        C = np.random.randn(3,3)
        return np.dot(C, C.T)
    
    obs = ObservationData.empty()
    obs.dt = 0.05
    obs.E_gyro = np.random.randn(3)
    obs.E_dgyro = np.random.randn(3)
    obs.E_force = np.random.randn(3)
    obs.E_dforce = np.random.randn(3)
    obs.Cov_gyro = generate_psd_matrix()
    obs.Cov_dgyro = generate_psd_matrix()
    obs.Cov_force = generate_psd_matrix()
    obs.Cov_dforce = generate_psd_matrix()

    return obs


def test_leastsq():
    ## dummies
    state_prevt = generate_random_states()
    this_obs_t = generate_random_obs()
    child_obs_t = generate_random_obs()

    print(state_prevt)

    ## test
    next_x = EstimateImuRelativePoseLeastSquare.update(state_prevt, this_obs_t, child_obs_t,
                                                       0.99, 0.99)

    return True


if __name__ == '__main__':
    print(test_leastsq())