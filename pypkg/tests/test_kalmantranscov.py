import numpy as np

from imu_relpose_estim.utils.nextxcov_helper import NextStateCovarianceHelper
from imu_relpose_estim.utils.noisecov_helper import NoiseCovarianceHelper
from imu_relpose_estim.utils.dataclasses import ObservationData, StateDataKalmanFilter
from imu_relpose_estim.utils.rotation_helper import RotationHelper

def generate_random_states(deterministic=False):
    if deterministic:
        scale = 0.0
    else:
        scale = 1.0

    ## translation parameter
    state = StateDataKalmanFilter.empty()
    state.trans = np.random.randn(6)
    sqrtC = np.random.randn(6,6)
    state.trans_cov = np.dot(sqrtC, sqrtC.T) * scale

    ## relative position of joint
    state.jointposition_wrt_thisframe = np.random.randn(3)
    state.jointposition_wrt_childframe = np.random.randn(3)

    ## bingham parameter
    if deterministic:
        M = np.linalg.qr(np.random.randn(4,4))[0]
        A = M @ np.diag([1e+30,0,0,0]) @ M.T
    else:
        A = np.random.randn(4,4) * 100.
        A = A + A.T
    state.bingham_param = RotationHelper.decompose_Amat(A)

    return state

def generate_random_obs(deterministic=False):
    def generate_psd_matrix():
        C = np.random.randn(3,3)
        return np.dot(C, C.T)
    
    if deterministic:
        scale = 0.0
    else:
        scale = 1.0

    obs = ObservationData.empty()
    obs.dt = 0.05
    obs.E_gyro = np.random.randn(3)
    obs.E_dgyro = np.random.randn(3)
    obs.E_force = np.random.randn(3)
    obs.E_dforce = np.random.randn(3)
    obs.Cov_gyro = generate_psd_matrix() * scale
    obs.Cov_dgyro = generate_psd_matrix() * scale
    obs.Cov_force = generate_psd_matrix() * scale
    obs.Cov_dforce = generate_psd_matrix() * scale

    return obs


def test_kalman_transobs():
    ## dummies
    state_prevt = generate_random_states(deterministic=True)
    this_obs_t = generate_random_obs(deterministic=True)
    child_obs_t = generate_random_obs(deterministic=True)


    E_tranobs = NoiseCovarianceHelper.calc_E_translation_obsv(
                                                  state_prevt.bingham_param.E_q4th,
                                                  this_obs_t.E_gyro,
                                                  child_obs_t.E_gyro,
                                                  state_prevt.jointposition_wrt_thisframe,
                                                  state_prevt.jointposition_wrt_childframe)
    C_tranobs = NoiseCovarianceHelper.calc_Cov_translation_obsv(
                                                  state_prevt.bingham_param.E_q4th,
                                                  this_obs_t.E_gyro,
                                                  child_obs_t.E_gyro,
                                                  this_obs_t.Cov_gyro,
                                                  child_obs_t.Cov_gyro,
                                                  state_prevt.jointposition_wrt_thisframe,
                                                  state_prevt.jointposition_wrt_childframe)
    print("the following covmatrix must be zero matrix:")
    print(C_tranobs)
    if not np.all(np.isclose(C_tranobs, np.zeros((6,6)))):
        raise ValueError
    print("---")

    Rot_ij = RotationHelper.quat_to_rotmat(*state_prevt.bingham_param.mode)
    gyro_i_cm = np.cross(np.eye(3), this_obs_t.E_gyro)
    gyro_j_cm = np.cross(np.eye(3), child_obs_t.E_gyro)

    E_tranobs_direct = np.append(
        state_prevt.jointposition_wrt_thisframe - Rot_ij @ state_prevt.jointposition_wrt_childframe,
        gyro_i_cm @ state_prevt.jointposition_wrt_thisframe - Rot_ij @ gyro_j_cm @ state_prevt.jointposition_wrt_childframe
    )

    print("the following two must coincide:")
    print(E_tranobs)
    print(E_tranobs_direct)
    if not np.all(np.isclose(E_tranobs, E_tranobs_direct)):
        raise ValueError
    
    return True


def test_nextxcov():
    ## dummies (deterministic)
    state_prevt = generate_random_states(deterministic=True)
    this_obs_t = generate_random_obs(deterministic=True)
    child_obs_t = generate_random_obs(deterministic=True)

    ## aliases
    dt = this_obs_t.dt
    Rot_ij = RotationHelper.quat_to_rotmat(*state_prevt.bingham_param.mode)
    gyro_i_cm = np.cross(np.eye(3), this_obs_t.E_gyro)
    gyro_j_cm = np.cross(np.eye(3), child_obs_t.E_gyro)
    dRot_ij = -gyro_i_cm @ Rot_ij + Rot_ij @ gyro_j_cm
    Amatrix = np.zeros((6,6))
    Amatrix[:3,:3] = -gyro_i_cm
    Amatrix[3:,3:] = -gyro_i_cm
    Amatrix[:3,3:] = np.eye(3)
    dAmatrix = np.zeros((6,6))
    dAmatrix[:3,:3] = -np.cross(np.eye(3), this_obs_t.E_dgyro)
    dAmatrix[3:,3:] = -np.cross(np.eye(3), this_obs_t.E_dgyro)


    print("calc_E_Xcoeff")
    Xcoeff_direct = np.eye(6) + Amatrix * dt + (dt**2 / 2.) * (dAmatrix + np.dot(Amatrix, Amatrix))
    Xcoeff = NextStateCovarianceHelper.calc_E_Xcoeff(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Xcoeff_direct, Xcoeff)):
        raise ValueError
    

    print("calc_E_Ucoeff")
    Ucoeff_direct = dt * np.eye(6) + (dt**2 / 2.) * Amatrix
    Ucoeff = NextStateCovarianceHelper.calc_E_Ucoeff(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Ucoeff_direct, Ucoeff)):
        raise ValueError
    

    print("calc_E_Xcoeff_kron_Xcoeff")
    Xc_kron_Xc_direct = np.kron(Xcoeff_direct, Xcoeff_direct)
    Xc_kron_Xc = NextStateCovarianceHelper.calc_E_Xcoeff_kron_Xcoeff(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Xc_kron_Xc_direct, Xc_kron_Xc)):
        raise ValueError


    print("calc_E_Xcoeff_kron_Ucoeff")
    Xc_kron_Uc_direct = np.kron(Xcoeff_direct, Ucoeff_direct)
    Xc_kron_Uc = NextStateCovarianceHelper.calc_E_Xcoeff_kron_Ucoeff(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Xc_kron_Uc_direct, Xc_kron_Uc)):
        raise ValueError


    print("calc_E_Ucoeff_kron_Ucoeff")
    Uc_kron_Uc_direct = np.kron(Ucoeff_direct, Ucoeff_direct)
    Uc_kron_Uc = NextStateCovarianceHelper.calc_E_Ucoeff_kron_Ucoeff(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Uc_kron_Uc_direct, Uc_kron_Uc)):
        raise ValueError


    print("calc_E_du_duT")
    du_direct = np.zeros(6)
    du_direct[3:] = dRot_ij @ child_obs_t.E_force + Rot_ij @ child_obs_t.E_dforce - this_obs_t.E_dforce
    duduT_direct = np.outer(du_direct, du_direct)
    duduT = NextStateCovarianceHelper.calc_E_du_duT(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(duduT_direct, duduT)):
        raise ValueError


    print("calc_E_u_uT")
    u_direct = np.zeros(6)
    u_direct[3:] = Rot_ij @ child_obs_t.E_force - this_obs_t.E_force
    uuT_direct = np.outer(u_direct, u_direct)
    uuT = NextStateCovarianceHelper.calc_E_u_uT(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(uuT_direct, uuT)):
        raise ValueError
    

    print("calc_E_Xcoeffx_duT")
    Xcoeffx_duT_direct = np.outer(np.dot(Xcoeff_direct, state_prevt.trans), du_direct)
    Xcoeffx_duT = NextStateCovarianceHelper.calc_E_Xcoeffx_duT(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Xcoeffx_duT_direct, Xcoeffx_duT)):
        raise ValueError
    

    print("calc_E_Ucoeffu_duT")
    Ucoeffu_duT_direct = np.outer(np.dot(Ucoeff_direct, u_direct), du_direct)
    Ucoeffu_duT = NextStateCovarianceHelper.calc_E_Ucoeffu_duT(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(Ucoeffu_duT_direct, Ucoeffu_duT)):
        raise ValueError
    

    print("calc_E_next_x")
    E_next_x_direct = np.dot(Xcoeff_direct, state_prevt.trans) + np.dot(Ucoeff_direct, u_direct) + (dt**2 / 2.) * du_direct
    E_next_x = NextStateCovarianceHelper.calc_E_next_x(state_prevt, this_obs_t, child_obs_t)
    if not np.all(np.isclose(E_next_x_direct, E_next_x)):
        raise ValueError


    print("calc_nextstate_cov")
    C_nextx = NextStateCovarianceHelper.calc_nextstate_cov(state_prevt, this_obs_t, child_obs_t)
    print("the following covmatrix must be zero matrix:")
    print(C_nextx)
    if not np.all(np.isclose(C_nextx, np.zeros((6,6)))):
        raise ValueError
    
    return True


def test_rotation_update():
    ## dummies (deterministic)
    state_prevt = generate_random_states(deterministic=True)
    this_obs_t = generate_random_obs(deterministic=True)
    child_obs_t = generate_random_obs(deterministic=True)

    ## aliases
    dt = this_obs_t.dt
    Rot_ij = RotationHelper.quat_to_rotmat(*state_prevt.bingham_param.mode)
    E_q4th = state_prevt.bingham_param.E_q4th
    gyro_i = this_obs_t.E_gyro
    gyro_j = child_obs_t.E_gyro
    gyro_noisecov_i = this_obs_t.Cov_gyro
    gyro_noisecov_j = child_obs_t.Cov_gyro

    rotation_pred = NoiseCovarianceHelper.calc_rotation_prediction_noise(E_q4th,
                                                                         gyro_i, gyro_j,
                                                                         gyro_noisecov_i, gyro_noisecov_j, dt)
    print(np.linalg.eigh(rotation_pred))
    # exp_calc

    nvec_unnorm = 0.5 * dt * (gyro_j - Rot_ij.T @ gyro_i)
    nvec_norm = np.linalg.norm(nvec_unnorm)
    nvec = nvec_unnorm / nvec_norm

    exp_omega = np.append([np.cos(nvec_norm)], nvec * np.sin(nvec_norm))
    print(exp_omega)

    print(dt)

    return True

np.set_printoptions(linewidth=np.inf)
print(test_kalman_transobs())
print(test_nextxcov())
print(test_rotation_update())