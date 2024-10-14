#!/usr/bin/env python3
import numpy as np
from ..utils.noisecov_helper import NoiseCovarianceHelper
from ..utils.rotation_helper import RotationHelper
from ..utils.dataclasses import ObservationData, StateDataGeneral, BinghamParameterGeneral, NormalDistributionData
from ..utils.sequential_leastsq import SequentialLeastSquare
from ..utils.hdbingham_helper import HDBinghamHelper

import rospy

class BinghamParameterExtLeastSquare(BinghamParameterGeneral):
    init_varlen = 2

    def __init__(self, Amat, CovQ_inv):
        self.Amat = Amat
        self.CovQ_inv = CovQ_inv
        self.update_parameters(Amat)

    @classmethod
    def empty(cls):
        return cls(*([None] * cls.init_varlen))

    @classmethod
    def initialize(cls):
        return cls(np.zeros((4, 4)), np.diag([4, 4, 4, 4]))

    @staticmethod
    def calc_Eq4th(bingham_Z, bingham_M, N_nc=50):
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(bingham_Z, N_nc=N_nc)
        Eq4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(bingham_M, ddnc_10D)
        return ddnc_10D, Eq4th

    @staticmethod
    def update_parameters(Amat, N_nc=50):
        Z, M = np.linalg.eigh(Amat)
        mode_quat = M[:, np.argmax(Z)]
        mode_rotmat = RotationHelper.quat_to_rotmat(*mode_quat)
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(Z, N_nc=N_nc)
        Eq4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(M, ddnc_10D)
        CovQ = NoiseCovarianceHelper.calc_covQ_from_4thmoment(Eq4th)
        return mode_quat, mode_rotmat, ddnc_10D, Eq4th, CovQ

    def update(self, Amat, CovQ_inv=None, N_nc=50, return_new=False):
        mode_quat, mode_rotmat, ddnc_10D, Eq4th, CovQ = self.update_parameters(Amat, N_nc=N_nc)
        cls = self.empty() if return_new else self

        # Update attributes
        cls.Amat = Amat
        cls.mode_quat = mode_quat
        cls.mode_rotmat = mode_rotmat
        cls.ddnc_10D = ddnc_10D
        cls.Eq4th = Eq4th
        cls.CovQ = CovQ
        cls.CovQ_inv = CovQ_inv if CovQ_inv is not None else cls.CovQ_inv

        return cls if return_new else None


class StateDataExtLeastSquare(StateDataGeneral):
    def __init__(self, position, position_cov, bingham_param: BinghamParameterExtLeastSquare):
        self.relative_position = NormalDistributionData(position, position_cov, name="relative_position")
        self.jointposition_wrt_thisframe = None
        self.jointposition_wrt_childframe = None

        self.bingham_param = bingham_param if bingham_param else BinghamParameterExtLeastSquare.empty()
        self.dt = None

    def update(self, position: np.ndarray, position_cov: np.ndarray, Amat: np.ndarray, CovQ_inv=None, return_new=False):
        cls = self.empty() if return_new else self
        cls.relative_position.position = position
        cls.relative_position.covariance = position_cov

        cls.bingham_param.update(Amat, CovQ_inv)

        return cls if return_new else None

    @classmethod
    def empty(cls):
        return cls(None, None, None)

    @classmethod
    def initialize(cls):
        return cls(np.zeros(3), np.eye(3), BinghamParameterExtLeastSquare.initialize())

    def is_jointposition_registered(self):
        flags = [
            self.jointposition_wrt_thisframe is None,
            self.jointposition_wrt_childframe is None
        ]

        if all(flags):
            return False
        elif any(flags):
            return True
        else:
            raise ValueError("Both `jointposition_wrt_thisframe` and `jointposition_wrt_childframe` must be defined or undefined.")

    def jointposition_registration(self, joint_position_wrt_i_array, joint_position_wrt_j_array, stddev=2e-3):
        def create_normaldist_from_pos_and_cov(mean_position, covariance):
            return None if mean_position is None else NormalDistributionData(position=mean_position, covariance=covariance)

        self.joint_position_cov_wrt_i = np.eye(3) * stddev**2
        self.joint_position_cov_wrt_j = np.eye(3) * stddev**2

        self.jointposition_wrt_thisframe = create_normaldist_from_pos_and_cov(joint_position_wrt_i_array, self.joint_position_cov_wrt_i)
        self.jointposition_wrt_childframe = create_normaldist_from_pos_and_cov(joint_position_wrt_j_array, self.joint_position_cov_wrt_j)


class BinghamHolder:
    def __init__(self):
        self.CovInv = None
        self.empty()

    def empty(self):
        self.Amat = np.zeros((4, 4))
        self.eigvals = np.zeros(4)
        self.orthomat = np.eye(4)
        self.mode = np.array([1, 0, 0, 0])

    @staticmethod
    def calc_deltaAmat(vec3d_after_rotation: np.ndarray, vec3d_before_rotation: np.ndarray, CovInv: np.ndarray):
        Hmat = NoiseCovarianceHelper.get_Hmat(vec3d_before_rotation, vec3d_after_rotation)
        return -0.5 * Hmat.T @ CovInv @ Hmat

    def update_Amat(self, Amat, CovInv=None):
        self.CovInv = CovInv
        self.eigvals, self.orthomat = np.linalg.eigh(Amat)
        amax = np.argmax(self.eigvals)
        self.Amat = Amat - np.eye(4) * self.eigvals[amax]
        self.mode = self.orthomat[:, amax]

    def update_Qinv(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData):
        raise NotImplementedError

    def update(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData, forgetting_factor_rotation):
        raise NotImplementedError

    def set_Qinv_once(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData):
        if self.CovInv is None:
            self.update_Qinv(this_state, this_obsdata, child_obsdata)


class BinghamHolderGyro(BinghamHolder):
    def update_Qinv(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData):
        cov_gyro_i = this_obsdata.Cov_gyro
        cov_gyro_j = child_obsdata.Cov_gyro

        gyro_cov = np.zeros((6, 6))
        gyro_cov[:3, :3] = cov_gyro_i
        gyro_cov[3:, 3:] = cov_gyro_j

        N = NoiseCovarianceHelper.get_Nmat()
        Q = N @ np.kron(gyro_cov, np.eye(4) * 0.25) @ N.T
        self.CovInv = np.linalg.inv(Q)

    def update(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData, forgetting_factor_rotation):
        w_i = this_obsdata.E_gyro
        w_j = child_obsdata.E_gyro
        deltaA_gyro = self.calc_deltaAmat(w_i, w_j, self.CovInv)
        self.update_Amat(self.Amat * forgetting_factor_rotation + deltaA_gyro)

        # rospy.logwarn(w_i)
        # rospy.logwarn(w_j)
        # rospy.logwarn("---")


class BinghamHolderForce(BinghamHolder):
    def update_Qinv(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData):
        def calc_E_coeff_kron_coeff_and_E_coeff(obsdata: ObservationData):
            E_gyrocmsq_kron_gyrocmsq = NoiseCovarianceHelper.calc_E_vCMsq_kron_vCMsq(obsdata.E_gyro, obsdata.Cov_gyro)
            E_dgyrosq_kron_dgyrosq = NoiseCovarianceHelper.calc_E_vCM_kron_vCM(obsdata.E_dgyro, obsdata.Cov_dgyro)
            E_gyrocmsq = NoiseCovarianceHelper.calc_E_vCMsq(obsdata.E_gyro, obsdata.Cov_gyro)
            E_dgyrocm = NoiseCovarianceHelper.calc_E_vCM(obsdata.E_dgyro)

            E_coeff_kron_coeff = (
                E_gyrocmsq_kron_gyrocmsq + E_dgyrosq_kron_dgyrosq +
                np.kron(E_gyrocmsq, E_dgyrocm) + np.kron(E_dgyrocm, E_gyrocmsq)
            )
            E_coeff = E_gyrocmsq + E_dgyrocm

            return E_coeff_kron_coeff, E_coeff

        E_coeff_kron_coeff_i, E_coeff_i = calc_E_coeff_kron_coeff_and_E_coeff(this_obsdata)
        E_coeff_kron_coeff_j, E_coeff_j = calc_E_coeff_kron_coeff_and_E_coeff(child_obsdata)

        joint_wrt_i = this_state.jointposition_wrt_thisframe
        joint_wrt_j = this_state.jointposition_wrt_childframe

        cov_force_i = this_obsdata.Cov_force + NoiseCovarianceHelper.calc_Cov_Ax(E_coeff_kron_coeff_i, E_coeff_i, joint_wrt_i.covariance, joint_wrt_i.position)
        cov_force_j = child_obsdata.Cov_force + NoiseCovarianceHelper.calc_Cov_Ax(E_coeff_kron_coeff_j, E_coeff_j, joint_wrt_j.covariance, joint_wrt_j.position)

        force_cov = np.zeros((6, 6))
        force_cov[:3, :3] = cov_force_i
        force_cov[3:, 3:] = cov_force_j

        N = NoiseCovarianceHelper.get_Nmat()
        Q = N @ np.kron(force_cov, np.eye(4) * 0.25) @ N.T
        self.CovInv = np.linalg.inv(Q)

    @classmethod
    def calc_observation_Amat(cls, this_obsdata: ObservationData, child_obsdata: ObservationData, joint_wrt_i: np.ndarray, joint_wrt_j: np.ndarray, rotmat=None):
        if rotmat is None:
            rotmat = np.eye(3)

        f_i = this_obsdata.E_force
        f_j = child_obsdata.E_force

        wcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_gyro)
        wcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_gyro)
        dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
        dwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_dgyro)

        N = NoiseCovarianceHelper.get_Nmat()

        A = wcm_i @ wcm_i + dwcm_i
        b = f_i
        C = wcm_j @ wcm_j + dwcm_j
        d = f_j

        EA = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_gyro, this_obsdata.Cov_gyro) + dwcm_i
        Eb = f_i
        EC = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_dgyro, this_obsdata.Cov_dgyro) + dwcm_j
        Ed = f_j
        Cov_b = this_obsdata.Cov_force
        Cov_d = child_obsdata.Cov_force

        EAkronA = HDBinghamHelper.calc_E_A_kron_A(this_obsdata.E_gyro, this_obsdata.E_dgyro, this_obsdata.Cov_gyro, this_obsdata.Cov_dgyro)
        ECkronC = HDBinghamHelper.calc_E_A_kron_A(child_obsdata.E_gyro, child_obsdata.E_dgyro, child_obsdata.Cov_gyro, child_obsdata.Cov_dgyro)

        Cov_Axb_Cyd_plain = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
            EAkronA, ECkronC,
            np.kron(Eb.reshape(-1, 1), EA), np.kron(Ed.reshape(-1, 1), EC),
            EA, EC,
            Eb, Ed,
            Cov_b, Cov_d,
            np.eye(6),  # Assuming cov_theta is identity
            np.zeros(6)  # Assuming theta is a zero vector
        )

        av = rotmat @ (A @ joint_wrt_i.position + b)
        bv = rotmat @ (C @ joint_wrt_j.position + d)

        Sigma_h = N @ (np.kron(Cov_Axb_Cyd_plain, np.eye(4) * 0.25) @ N.T)
        HTH_plain = cls.calc_deltaAmat(av, bv, CovInv=np.linalg.pinv(Sigma_h))

        Amat_ee = 2 * wcm_i + np.eye(3)
        Bmat_ee = 2 * wcm_j + np.eye(3)
        av_ee = Amat_ee @ joint_wrt_i.position
        bv_ee = -Bmat_ee @ joint_wrt_j.position

        Cov_Axb_Cyd_ee = np.eye(6)
        Cov_Axb_Cyd_ee[:3, :3] = 4 * NoiseCovarianceHelper.calc_Cov_Ax(
            NoiseCovarianceHelper.calc_E_vCM_kron_vCM(this_obsdata.E_gyro, this_obsdata.Cov_gyro),
            wcm_i,
            np.eye(3),
            joint_wrt_i.position
        )
        Cov_Axb_Cyd_ee[3:, 3:] = 4 * NoiseCovarianceHelper.calc_Cov_Ax(
            NoiseCovarianceHelper.calc_E_vCM_kron_vCM(child_obsdata.E_gyro, child_obsdata.Cov_gyro),
            wcm_j,
            np.eye(3),
            joint_wrt_j.position
        )
        Cov_Axb_Cyd_ee[:3, 3:] = -4 * (np.dot(wcm_i, wcm_j.T))
        Cov_Axb_Cyd_ee[3:, :3] = Cov_Axb_Cyd_ee[:3, 3:].T

        Sigma_ee = N @ (np.kron(Cov_Axb_Cyd_ee, np.eye(4) * 0.25) @ N.T)
        HTH_ee = cls.calc_deltaAmat(av_ee, bv_ee, CovInv=np.linalg.pinv(Sigma_ee))

        deltaA_force = HTH_plain  # + HTH_ee
        return deltaA_force

    @classmethod
    def calc_prediction_Amat(cls, this_obsdata: ObservationData, child_obsdata: ObservationData, Eq4th, CovQ, dt):
        rotexp_cov = NoiseCovarianceHelper.calc_rotation_prediction_noise(Eq4th,
                                                                        this_obsdata.E_gyro, child_obsdata.E_gyro,
                                                                        this_obsdata.Cov_gyro, child_obsdata.Cov_gyro,
                                                                        dt)
        rotpred_noisecov = NoiseCovarianceHelper.calc_nextq_Cov(CovQ, rotexp_cov)
        return RotationHelper.decompose_CovQ(rotpred_noisecov).Amat

    def update(self, this_state: StateDataExtLeastSquare, this_obsdata: ObservationData, child_obsdata: ObservationData, forgetting_factor_rotation, forgetting_factor_position):
        self.update_rotation(this_state, this_obsdata, child_obsdata, self.force_bingham, self.gyro_bingham, forgetting_factor_rotation)

        if this_state.is_jointposition_registered():
            this_state.bingham_param.update(self.force_bingham.Amat)
        else:
            this_state.bingham_param.update(self.gyro_bingham.Amat)

        this_state.relative_position = self.update_position(self.direct_relpos_estim, this_state, this_obsdata, child_obsdata, forgetting_factor_position, use_child_gyro=self.use_child_gyro)
        return this_state

    def reset_estimation(self):
        self.gyro_bingham.update_Amat(np.zeros((4, 4)))
        self.force_bingham.update_Amat(np.zeros((4, 4)))
        self.direct_relpos_estim.initialize()


class EstimateImuRelativePoseExtendedLeastSquare:
    def __init__(self,
                 joint_position_wrt_i_measured=None,
                 joint_position_wrt_j_measured=None,
                 use_child_gyro=True):
        self.N = NoiseCovarianceHelper.get_Nmat()

        # Initialize rotation matrix
        self.rotation_Amat = np.eye(4) * 1e-9

        # Initialize SequentialLeastSquare depending on use_child_gyro
        if use_child_gyro:
            self.jointpos_estim = SequentialLeastSquare(6, 6, "joint")
            self.direct_relpos_estim = SequentialLeastSquare(6, 6, "relposition")
        else:
            self.jointpos_estim = SequentialLeastSquare(3, 3, "joint")
            self.direct_relpos_estim = SequentialLeastSquare(3, 3, "relposition")
        
        self.force_bingham = BinghamHolderForce()
        self.gyro_bingham = BinghamHolderGyro()

        self.joint_position_wrt_i_measured = joint_position_wrt_i_measured
        self.joint_position_wrt_j_measured = joint_position_wrt_j_measured
        self.use_child_gyro = use_child_gyro

    @staticmethod
    def update_rotation(this_state, this_obsdata, child_obsdata, force_bingham, gyro_bingham, forgetting_factor_rotation):
        # Update rotation based on joint position registration status
        if this_state.is_jointposition_registered():
            force_bingham.set_Qinv_once(this_state, this_obsdata, child_obsdata)
            force_bingham.update(this_state, this_obsdata, child_obsdata, forgetting_factor_rotation)
        else:
            gyro_bingham.set_Qinv_once(this_state, this_obsdata, child_obsdata)
            gyro_bingham.update(this_state, this_obsdata, child_obsdata, forgetting_factor_rotation)

    @staticmethod
    def calc_gyro_estimate_coeffs(this_state, this_obsdata, child_obsdata, use_child_gyro=False):
        # Compute rotation matrix and force difference
        E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
        DeltaFi = np.dot(E_R, child_obsdata.E_force) - this_obsdata.E_force

        # Calculate covariance matrices
        dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
        wcmsq_i = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_gyro, this_obsdata.Cov_gyro)
        dwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_dgyro)
        wcmsq_j = NoiseCovarianceHelper.calc_E_vCMsq(child_obsdata.E_gyro, child_obsdata.Cov_gyro)

        Omega_i = wcmsq_i + dwcm_i
        Omega_j = wcmsq_j + dwcm_j

        extOmega = np.eye(6)
        if use_child_gyro:
            extOmega[:3, :3] = Omega_i
            extOmega[3:, 3:] = np.dot(E_R, Omega_j).dot(E_R.T)
            extDeltaFi = np.concatenate([DeltaFi, DeltaFi])
        else:
            extOmega = 0.5 * (Omega_i + np.dot(E_R, Omega_j).dot(E_R.T))
            extDeltaFi = DeltaFi

        return extOmega, extDeltaFi

    @classmethod
    def update_position(cls, relpos_estimator, this_state, this_obsdata, child_obsdata, forgetting_factor_position, use_child_gyro=False):
        if this_state.is_jointposition_registered():
            E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
            E_R_kron_R = NoiseCovarianceHelper.calc_E_R_kron_R(this_state.bingham_param.Eq4th)

            # Estimate relative position based on force
            joint_wrt_i = this_state.jointposition_wrt_thisframe
            joint_wrt_j = this_state.jointposition_wrt_childframe
            position = joint_wrt_i.position - np.dot(E_R, joint_wrt_j.position)
            cov_position = joint_wrt_i.covariance + NoiseCovarianceHelper.calc_Cov_Ax(
                E_R_kron_R, E_R, joint_wrt_j.covariance, joint_wrt_j.position
            )
            return NormalDistributionData(position, cov_position)
        else:
            extOmega, DeltaFi = cls.calc_gyro_estimate_coeffs(this_state, this_obsdata, child_obsdata, use_child_gyro=use_child_gyro)
            relpos_estimator.update(extOmega, DeltaFi, forgetting_factor=forgetting_factor_position)
            return relpos_estimator.get_estimates()

    def update(self, this_state, this_obsdata, child_obsdata, forgetting_factor_rotation, forgetting_factor_position):
        # Update rotation and position
        self.update_rotation(this_state, this_obsdata, child_obsdata, self.force_bingham, self.gyro_bingham, forgetting_factor_rotation)

        if this_state.is_jointposition_registered():
            this_state.bingham_param.update(self.force_bingham.Amat)
        else:
            this_state.bingham_param.update(self.gyro_bingham.Amat)

        this_state.relative_position = self.update_position(
            self.direct_relpos_estim, this_state, this_obsdata, child_obsdata,
            forgetting_factor_position, use_child_gyro=self.use_child_gyro
        )

        return this_state

    def reset_estimation(self):
        # Reset all estimation data
        self.gyro_bingham.update_Amat(np.zeros((4, 4)))
        self.force_bingham.update_Amat(np.zeros((4, 4)))
        self.direct_relpos_estim.initialize()
