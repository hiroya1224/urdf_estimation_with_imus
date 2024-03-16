import numpy as np

from imu_relpose_estim.estimator.kalman import EstimateImuRelativePoseKalmanFilter
from imu_relpose_estim.utils.dataclasses import ObservationData, StateDataKalmanFilter
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
    ## translation parameter
    state = StateDataKalmanFilter.empty()
    state.trans = np.random.randn(6)
    sqrtC = np.random.randn(6,6)
    state.trans_cov = np.dot(sqrtC, sqrtC.T)

    ## relative position of joint
    state.jointposition_wrt_thisframe = np.random.randn(3)
    state.jointposition_wrt_childframe = np.random.randn(3)

    ## bingham parameter
    A = np.random.randn(4,4) * 100.
    A = A + A.T
    state.bingham_param = RotationHelper.decompose_Amat(A)

    return state


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


def test_kalman():
    ## dummies
    state_prevt = generate_random_states()
    this_obs_t = generate_random_obs()
    child_obs_t = generate_random_obs()

    ## test
    next_x = EstimateImuRelativePoseKalmanFilter.update(state_prevt, this_obs_t, child_obs_t)

    return True


if __name__ == '__main__':
    print(test_kalman())