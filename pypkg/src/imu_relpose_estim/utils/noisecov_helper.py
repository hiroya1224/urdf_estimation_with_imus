from ._noisecov_helper import (_calc_10D_ddnc,
                               _calc_4thmoment_coeff,
                               _calc_Bingham_4thMoment,
                               _calc_covQ_from_4thmoment,
                               _calc_nextq_Cov,
                               _calc_rotation_prediction_noise,
                               _calc_E_R,
                               _calc_E_R_kron_R,
                               _calc_E_vCMsq_kron_vCMsq,
                               _calc_E_vCM_kron_vCM,
                               _calc_translation_observation_1stmoment,
                               _calc_translation_observation_2ndmoment)
import numpy as np

class NoiseCovarianceHelper:

    #### aliases for _noisecov_helper ####

    @staticmethod
    def calc_10D_ddnc(binghamparam_eigvals, N_nc=50):
        return _calc_10D_ddnc(binghamparam_eigvals, N_nc)
    

    @staticmethod
    def calc_4thmoment_coeff(M_orthogonal):
        return _calc_4thmoment_coeff(M_orthogonal)
    

    @staticmethod
    def calc_Bingham_4thMoment(binghamparam_orthomat, ddnc_10D):
        return _calc_Bingham_4thMoment(binghamparam_orthomat, ddnc_10D)


    @staticmethod
    def calc_covQ_from_4thmoment(E_q4th):
        return _calc_covQ_from_4thmoment(E_q4th)


    @staticmethod
    def calc_nextq_Cov(Cov_currentq, Cov_noise):
        return _calc_nextq_Cov(Cov_currentq, Cov_noise)


    @staticmethod
    def calc_rotation_prediction_noise(E_q4th, gyro_i, gyro_j, gyro_noisecov_i, gyro_noisecov_j, dt):
        ## helper
        # def Rmat_quat(w,x,y,z):
        #     ## to rotate bingham, K @ A @ K.T
        #     return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])
        
        # omega_composite = 0.5 * dt * (gyro_j - Rij.T @ gyro_i)
        # norm = np.linalg.norm(omega_composite)
        # norm_vec = omega_composite / norm
        # ## exp
        # exp_omega = np.append([np.cos(norm)], np.sin(norm) * norm_vec)
        # approx_exp = np.append([1.], omega_composite)
        # approx_exp = approx_exp / np.linalg.norm(approx_exp)
        # ## modification
        # modify_matrix = Rmat_quat(*exp_omega) @ Rmat_quat(*approx_exp).T
        # noise_pred = _calc_rotation_prediction_noise(E_q4th, gyro_i, gyro_j, gyro_noisecov_i, gyro_noisecov_j, dt)
        # return modify_matrix @ noise_pred @ modify_matrix.T
        return _calc_rotation_prediction_noise(E_q4th, gyro_i, gyro_j, gyro_noisecov_i, gyro_noisecov_j, dt)


    @staticmethod
    def calc_E_R(E_q4th):
        return _calc_E_R(E_q4th)


    @staticmethod
    def calc_E_R_kron_R(E_q4th):
        return _calc_E_R_kron_R(E_q4th)


    @staticmethod
    def calc_E_vCMsq_kron_vCMsq(v, cov_v):
        return _calc_E_vCMsq_kron_vCMsq(v, cov_v)


    @staticmethod
    def calc_E_vCM_kron_vCM(v, cov_v):
        return _calc_E_vCM_kron_vCM(v, cov_v)
    

    @staticmethod
    def calc_E_translation_obsv(E_q4th, gyro_i, gyro_j, joint_wrt_i, joint_wrt_j):
        return _calc_translation_observation_1stmoment(E_q4th, gyro_i, gyro_j, joint_wrt_i, joint_wrt_j)
    

    @classmethod
    def calc_Cov_translation_obsv(cls, E_q4th, gyro_i, gyro_j, gyro_noisecov_i, gyro_noisecov_j, joint_wrt_i, joint_wrt_j):
        E_OOT = _calc_translation_observation_2ndmoment(E_q4th, gyro_i, gyro_j, gyro_noisecov_i, gyro_noisecov_j, joint_wrt_i, joint_wrt_j)
        E_O = cls.calc_E_translation_obsv(E_q4th, gyro_i, gyro_j, joint_wrt_i, joint_wrt_j)
        return E_OOT - np.outer(E_O, E_O)


    #### newly added functions ####

    @staticmethod
    def calc_Cov_Ax(E_AkronA, E_A, CovX, E_X):
        assert CovX.shape[0] == CovX.shape[1]
        m = CovX.shape[0]
        
        assert E_X.shape[0] == m, "E_X.shape[0] == {} != m == {}".format(E_X.shape[0], m)
        assert len(E_X.shape) == 1 or E_X.shape[1] == 1, "E_X.shape == {}".format(E_X.shape)

        vec_CovX = CovX.transpose(0,1).reshape(-1,1)
        vec_EXEXT = np.outer(E_X, E_X).transpose(0,1).reshape(-1,1)
        EA_kron_EA = np.kron(E_A, E_A)
        vec_CovAx = np.dot(E_AkronA, vec_CovX + vec_EXEXT) - np.dot(EA_kron_EA, vec_EXEXT)

        return vec_CovAx.reshape(m,m).transpose(0,1)
    

    @staticmethod
    def calc_E_vCMsq(v, cov_v):
        ## calc E[(w_i - mu_i) (w_j - mu_j)] = E[w_i w_j] - mu_i mu_j first
        C = cov_v
        Cov_omegaCMsq = np.zeros((3,3))
        Cov_omegaCMsq[0,1] = C[0,1]
        Cov_omegaCMsq[0,2] = C[0,2]
        Cov_omegaCMsq[1,2] = C[1,2]

        ## set symmetric parts
        Cov_omegaCMsq = Cov_omegaCMsq + Cov_omegaCMsq.T

        ## set diagonal parts
        Cov_omegaCMsq[0,0] = -C[1,1] - C[2,2]
        Cov_omegaCMsq[1,1] = -C[0,0] - C[2,2]
        Cov_omegaCMsq[2,2] = -C[0,0] - C[1,1]

        ## then add mu_i mu_j
        E_omegaCM = np.cross(np.eye(3), v)
        return Cov_omegaCMsq + np.dot(E_omegaCM, E_omegaCM)


    @staticmethod
    def calc_E_vCM(v):
        return np.cross(np.eye(3), v)
    

    @staticmethod
    def calc_covQupdate(covQ, covA, covB, const_Nmat):
        covAB = np.zeros((6,6))
        covAB[:3,:3] = covA
        covAB[3:,3:] = covB
        N = const_Nmat
        return N @ np.kron(covAB, covQ) @ N.T


    @staticmethod
    def get_Nmat():
        ## initialize
        N = np.zeros((4, 24))
        ## set non-zero elements
        N[0, 1] = -1.
        N[0, 6] = -1.
        N[0, 11] = -1.
        N[0, 13] = 1.
        N[0, 18] = 1.
        N[0, 23] = 1.
        N[1, 0] = 1.
        N[1, 7] = -1.
        N[1, 10] = 1.
        N[1, 12] = -1.
        N[1, 19] = -1.
        N[1, 22] = 1.
        N[2, 3] = 1.
        N[2, 4] = 1.
        N[2, 9] = -1.
        N[2, 15] = 1.
        N[2, 16] = -1.
        N[2, 21] = -1.
        N[3, 2] = -1.
        N[3, 5] = 1.
        N[3, 8] = 1.
        N[3, 14] = -1.
        N[3, 17] = 1.
        N[3, 20] = -1.
        return N


    @staticmethod
    def get_Hmat(vec3D_bef_rot, vec3D_aft_rot):
        ## alias
        v0 = vec3D_aft_rot
        v1 = vec3D_bef_rot

        ## initialize
        H = np.zeros((4,4))

        # ## pseudo-measurement matrix
        # # H[0,0] = 0
        # H[0,1] = -v0[0] + v1[0]
        # H[0,2] = -v0[1] + v1[1]
        # H[0,3] = -v0[2] + v1[2]
        # H[1,0] = v0[0] - v1[0]
        # # H[1,1] = 0
        # H[1,2] = v0[2] + v1[2]
        # H[1,3] = -v0[1] - v1[1]
        # H[2,0] = v0[1] - v1[1]
        # H[2,1] = -v0[2] - v1[2]
        # # H[2,2] = 0
        # H[2,3] = v0[0] + v1[0]
        # H[3,0] = v0[2] - v1[2]
        # H[3,1] = v0[1] + v1[1]
        # H[3,2] = -v0[0] - v1[0]
        # # H[3,3] = 0


        ## pseudo-measurement matrix
        # H[0,0] = 0
        H[0,1] = -v0[0] + v1[0]
        H[0,2] = -v0[1] + v1[1]
        H[0,3] = -v0[2] + v1[2]
        H[1,0] = v0[0] - v1[0]
        # H[1,1] = 0
        H[1,2] = -v0[2] - v1[2]
        H[1,3] = v0[1] + v1[1]
        H[2,0] = v0[1] - v1[1]
        H[2,1] = v0[2] + v1[2]
        # H[2,2] = 0
        H[2,3] = -v0[0] - v1[0]
        H[3,0] = v0[2] - v1[2]
        H[3,1] = -v0[1] - v1[1]
        H[3,2] = v0[0] + v1[0]
        # H[3,3] = 0

        return H
