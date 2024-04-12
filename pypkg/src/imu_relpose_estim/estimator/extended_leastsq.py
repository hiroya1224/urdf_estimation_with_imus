#!/usr/bin/env python3
import numpy as np
from ..utils.noisecov_helper import NoiseCovarianceHelper
from ..utils.rotation_helper import RotationHelper
from ..utils.dataclasses import ObservationData, StateDataGeneral, BinghamParameterGeneral, NormalDistributionData
from ..utils.sequential_leastsq import SequentialLeastSquare


## NOTE: you should check the comments with "CHECK" notes.

class BinghamParameterExtLeastSquare(BinghamParameterGeneral):
    init_varlen = 2
    def __init__(self,
                Amat,
                CovQ_inv):
        self.Amat = Amat
        self.CovQ_inv = CovQ_inv
        self.mode_quat, self.mode_rotmat, self.ddnc_10D, self.Eq4th, self.CovQ = self._update(Amat)


    @classmethod
    def empty(cls):
        return cls(*([None]*cls.init_varlen))
    

    @classmethod
    def initialize(cls):
        new_cls = cls(np.zeros((4,4)), np.diag([4, 4, 4, 4]))
        return new_cls
    

    @staticmethod
    def calc_Eq4th(bingham_Z, bingham_M, N_nc=50):
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(bingham_Z, N_nc=N_nc)
        Eq4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(bingham_M, ddnc_10D)
        return ddnc_10D, Eq4th


    @staticmethod
    def _update(Amat, N_nc=50):
        Z,M = np.linalg.eigh(Amat)
        mode_quat = M[:, np.argmax(Z)]
        mode_rotmat = RotationHelper.quat_to_rotmat(*mode_quat)
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(Z, N_nc=N_nc)
        Eq4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(M, ddnc_10D)
        CovQ = NoiseCovarianceHelper.calc_covQ_from_4thmoment(Eq4th)
        return mode_quat, mode_rotmat, ddnc_10D, Eq4th, CovQ


    def update(self, Amat, CovQ_inv=None, N_nc=50, return_new=False):
        mode_quat, mode_rotmat, ddnc_10D, Eq4th, CovQ = self._update(Amat, N_nc=N_nc)
        if return_new:
            cls = self.empty()
        else:
            cls = self

        ## update
        cls.Amat = Amat
        cls.mode_quat = mode_quat
        cls.mode_rotmat = mode_rotmat
        cls.ddnc_10D = ddnc_10D
        cls.Eq4th = Eq4th
        cls.CovQ = CovQ
        if CovQ_inv is None:
            cls.CovQ_inv = cls.CovQ_inv
        else:
            cls.CovQ_inv = CovQ_inv

        if return_new:
            return cls
        else:
            return None


class StateDataExtLeastSquare(StateDataGeneral):
    def __init__(self, position, position_cov,
                bingham_param: BinghamParameterExtLeastSquare,
                ):
        self.relative_position = NormalDistributionData(position, position_cov, name="relative_position")

        self.jointposition_wrt_thisframe = None
        self.jointposition_wrt_childframe = None

        if bingham_param is None:
            bingham_param = BinghamParameterExtLeastSquare.empty()
        self.bingham_param = bingham_param

        self.dt = None


    def calc_Omega_mat():
        pass


    def update(self, 
               position: np.ndarray,
               position_cov: np.ndarray,
               Amat: np.ndarray,
               CovQ_inv=None,
               return_new=False):
        if return_new:
            cls = self.empty()
        else:
            cls = self
        cls.relative_position.position = position
        cls.relative_position.covariance = position_cov

        ## set previous values first then update
        cls.bingham_param = self.bingham_param.update(Amat, CovQ_inv)

        if return_new:
            return cls
        else:
            return None


    @classmethod
    def empty(cls):
        return cls(None, None, None)
    

    @classmethod
    def initialize(cls):
        return cls(np.zeros(3), np.eye(3), BinghamParameterExtLeastSquare.initialize())
    

    def is_jointposition_registered(self):
        flag = 0
        if self.jointposition_wrt_thisframe is None:
            flag += 1
        if self.jointposition_wrt_childframe is None:
            flag += 1

        ## if all is set
        if flag == 0:
            return True
        
        ## if none is set
        if flag == 2:
            return False
        
        ## invalid pattern
        raise ValueError("Both of `jointposition_wrt_thisframe` and `jointposition_wrt_childframe` must be defined or undefined.")
        

    def jointposition_registration(self, joint_position_wrt_i_array, joint_position_wrt_j_array, stddev=2e-3):
        ## helper
        def create_normaldist_from_pos_and_cov(mean_position, covariance):
            if mean_position is None:
                return None
            return NormalDistributionData(position=mean_position, covariance=covariance)
        
        self.joint_position_cov_wrt_i = np.eye(3) * stddev**2
        self.joint_position_cov_wrt_j = np.eye(3) * stddev**2

        ## set relative value information of IMU and its associated joint
        self.jointposition_wrt_thisframe = create_normaldist_from_pos_and_cov(joint_position_wrt_i_array,
                                                                              self.joint_position_cov_wrt_i)
        self.jointposition_wrt_childframe = create_normaldist_from_pos_and_cov(joint_position_wrt_j_array,
                                                                              self.joint_position_cov_wrt_j)



class BinghamHolder:
    def __init__(self):
        self.CovInv = None
        self.Amat = None
        self.eigvals = None
        self.orthomat = None
        self.mode = None
        self.empty()


    def empty(self):
        self.Amat = np.zeros((4,4))
        self.eigvals = np.zeros(4)
        self.orthomat = np.eye(4)
        self.mode = np.array([1, 0, 0, 0])


    def calc_deltaAmat(self,
                          vec3d_after_rotation: np.ndarray, 
                          vec3d_before_rotation: np.ndarray):
        Hmat = NoiseCovarianceHelper.get_Hmat(vec3d_before_rotation, vec3d_after_rotation)
        deltaA = -0.5 * Hmat.T @ self.CovInv @ Hmat
        return deltaA
    

    def update_Amat(self, Amat, CovInv=None):
        self.CovInv = CovInv
        self.eigvals, self.orthomat = np.linalg.eigh(Amat)

        amax = np.argmax(self.eigvals)
        self.Amat = Amat - np.eye(4) * self.eigvals[amax]
        self.mode = self.orthomat[:, amax]
    

    def update_Qinv(self,
                    this_state: StateDataExtLeastSquare,
                    this_obsdata: ObservationData,
                    child_obsdata: ObservationData):
        raise NotImplementedError
    

    def update(self,
            this_state: StateDataExtLeastSquare,
            this_obsdata: ObservationData,
            child_obsdata: ObservationData,
            forgetting_factor_rotation):
        raise NotImplementedError
    

    def set_Qinv_once(self,
                this_state: StateDataExtLeastSquare,
                this_obsdata: ObservationData,
                child_obsdata: ObservationData):
        if self.CovInv is None:
            self.update_Qinv(this_state, this_obsdata, child_obsdata)




class BinghamHolderGyro(BinghamHolder):
    def update_Qinv(self,
                    this_state: StateDataExtLeastSquare,
                    this_obsdata: ObservationData,
                    child_obsdata: ObservationData):
        ## CHECK: mathematics (cov order); checked: cov = diag(after, before)
        cov_gyro_i = this_obsdata.Cov_gyro
        cov_gyro_j = child_obsdata.Cov_gyro

        ## set covariance parameters
        gyro_cov = np.zeros((6,6))
        gyro_cov[:3,:3] = cov_gyro_i
        gyro_cov[3:,3:] = cov_gyro_j

        ## set inverse of Q
        N = NoiseCovarianceHelper.get_Nmat()
        Q = np.dot(np.dot(N, np.kron(gyro_cov, np.eye(4)*0.25)), N.T)
        
        ## update
        self.CovInv = np.linalg.inv(Q)


    def update(self,
               this_state: StateDataExtLeastSquare,
               this_obsdata: ObservationData,
               child_obsdata: ObservationData,
               forgetting_factor_rotation):
        w_i = this_obsdata.E_gyro
        w_j = child_obsdata.E_gyro
        deltaA_gyro  = self.calc_deltaAmat(w_i, w_j)
        self.update_Amat(self.Amat * forgetting_factor_rotation + deltaA_gyro)
    

class BinghamHolderForce(BinghamHolder):
    def update_Qinv(self,
                    this_state: StateDataExtLeastSquare,
                    this_obsdata: ObservationData,
                    child_obsdata: ObservationData):
        ## helper
        def calc_E_coeff_kron_coeff_and_E_coeff(obsdata: ObservationData):
            E_gyrocmsq_kron_gyrocmsq = NoiseCovarianceHelper.calc_E_vCMsq_kron_vCMsq(obsdata.E_gyro, obsdata.Cov_gyro)
            E_dgyrosq_kron_dgyrosq = NoiseCovarianceHelper.calc_E_vCM_kron_vCM(obsdata.E_dgyro, obsdata.Cov_dgyro)
            E_gyrocmsq = NoiseCovarianceHelper.calc_E_vCMsq(obsdata.E_gyro, obsdata.Cov_gyro)
            E_dgyrocm = NoiseCovarianceHelper.calc_E_vCM(obsdata.E_dgyro)

            ## calc E[(wcm_i^2 + dwcm_i) (X) (wcm_i^2 + dwcm_i)]
            ## = E[wcm_i^2 (X) wcm_i^2 + dwcm_i (X) dwcm_i + wcm_i^2 (X) dwcm_i + dwcm_i (X) wcm_i^2]
            ## = E[wcm_i^2 (X) wcm_i^2] + E[dwcm_i (X) dwcm_i] + E[wcm_i^2 (X) dwcm_i] + E[dwcm_i (X) wcm_i^2]
            ## = E[wcm_i^2 (X) wcm_i^2] + E[dwcm_i (X) dwcm_i] + E[wcm_i^2] (X) E[dwcm_i] + E[dwcm_i] (X) E[wcm_i^2]
            E_coeff_kron_coeff = E_gyrocmsq_kron_gyrocmsq + E_dgyrosq_kron_dgyrosq \
                               + np.kron(E_gyrocmsq, E_dgyrocm) + np.kron(E_dgyrocm, E_gyrocmsq)
            E_coeff = E_gyrocmsq + E_dgyrocm

            return E_coeff_kron_coeff, E_coeff

        E_coeff_kron_coeff_i, E_coeff_i = calc_E_coeff_kron_coeff_and_E_coeff(this_obsdata)
        E_coeff_kron_coeff_j, E_coeff_j = calc_E_coeff_kron_coeff_and_E_coeff(child_obsdata)

        joint_wrt_i = this_state.jointposition_wrt_thisframe
        joint_wrt_j = this_state.jointposition_wrt_childframe

        cov_force_i = this_obsdata.Cov_force + NoiseCovarianceHelper.calc_Cov_Ax(E_coeff_kron_coeff_i, E_coeff_i, joint_wrt_i.covariance, joint_wrt_i.position)
        cov_force_j = child_obsdata.Cov_force + NoiseCovarianceHelper.calc_Cov_Ax(E_coeff_kron_coeff_j, E_coeff_j, joint_wrt_j.covariance, joint_wrt_j.position)

        ## set covariance parameters
        force_cov = np.zeros((6,6))
        force_cov[:3,:3] = cov_force_i
        force_cov[3:,3:] = cov_force_j

        ## set inverse of Q
        N = NoiseCovarianceHelper.get_Nmat()
        Q = np.dot(np.dot(N, np.kron(force_cov, np.eye(4)*0.25)), N.T)

        ## update
        self.CovInv = np.linalg.inv(Q)


    def update(self,
              this_state: StateDataExtLeastSquare,
              this_obsdata: ObservationData,
              child_obsdata: ObservationData,
              forgetting_factor_rotation):

        f_i = this_obsdata.E_force
        f_j = child_obsdata.E_force
        joint_wrt_i = this_state.jointposition_wrt_thisframe
        joint_wrt_j = this_state.jointposition_wrt_childframe

        wcm_i  = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_gyro)
        wcm_j  = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_gyro)
        dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
        dwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_dgyro)

        joint_wrt_i = joint_wrt_i.position
        joint_wrt_j = joint_wrt_j.position
        force_i = f_i + (wcm_i@wcm_i + dwcm_i) @ joint_wrt_i
        force_j = f_j + (wcm_j@wcm_j + dwcm_j) @ joint_wrt_j
        deltaA_force = self.calc_deltaAmat(force_i, force_j)

        ## constraint about joint rotation axis
        A_neg_y = np.diag([-1e+9, 0, -1e+9, 0])
        constrainted_A = deltaA_force + A_neg_y

        self.update_Amat(constrainted_A)


class EstimateImuRelativePoseExtendedLeastSquare:
    def __init__(self,
                 joint_position_wrt_i_measured=None,
                 joint_position_wrt_j_measured=None,
                 use_child_gyro=True):
        self.N = NoiseCovarianceHelper.get_Nmat()

        ## initialize
        self.rotation_Amat = np.eye(4) * 1e-9

        ## sequential leastsq
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
    def update_rotation(this_state: StateDataExtLeastSquare,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData,
                        force_bingham: BinghamHolderForce,
                        gyro_bingham: BinghamHolderGyro,
                        forgetting_factor_rotation):
        
        ## `set_Qinv_once` updates bingham param if its CovInv is None
        ## update rotation (switch method based on whether the relative joint position is available)
        if this_state.is_jointposition_registered():
            ## NOTE: `forgetting_factor_rotation` is unused in force_bingham.update
            force_bingham.set_Qinv_once(this_state, this_obsdata, child_obsdata)
            force_bingham.update(this_state, this_obsdata, child_obsdata, forgetting_factor_rotation)
        else:
            gyro_bingham.set_Qinv_once(this_state, this_obsdata, child_obsdata)
            gyro_bingham.update(this_state, this_obsdata, child_obsdata, forgetting_factor_rotation)

    
    @staticmethod
    def determinant_of_coeffmat(this_obsdata: ObservationData):
        w  = this_obsdata.E_gyro
        dw = this_obsdata.E_dgyro
        w_cross_dw = np.cross(w, dw)
        return -1. * np.dot(w_cross_dw, w_cross_dw)


    @staticmethod
    def calc_gyro_estimate_coeffs(
                        this_state: StateDataExtLeastSquare,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData,
                        use_child_gyro=False):
        ## alias
        E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
        f_i = this_obsdata.E_force
        f_j = child_obsdata.E_force
        DeltaFi = np.dot(E_R, f_j) - f_i
        
        dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
        wcmsq_i = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_gyro, this_obsdata.Cov_gyro)
        dwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_dgyro)
        wcmsq_j = NoiseCovarianceHelper.calc_E_vCMsq(child_obsdata.E_gyro, child_obsdata.Cov_gyro)
        Omega_i = wcmsq_i + dwcm_i
        Omega_j = wcmsq_j + dwcm_j

        ## additional
        if use_child_gyro:

            ## fuse
            # s = 0.5
            # _x = Omega_i
            # _y = E_R @ Omega_j @ E_R.T
            # s = 0.5 * np.dot(DeltaFi - _y, _x - _y) / np.dot(_x - _y, _x - _y)
            extOmega = np.eye(6)
            extOmega[:3,:3] = Omega_i
            extOmega[3:,3:] = E_R @ Omega_j @ E_R.T

            extDeltaFi = np.zeros(6)
            extDeltaFi[:3] = DeltaFi
            extDeltaFi[3:] = DeltaFi
        else:
            extOmega = 0.5 * (Omega_i + E_R @ Omega_j @ E_R.T)
            extDeltaFi = DeltaFi
            
        # wcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_gyro)
        # wcmsq_i = wcm_i @ wcm_i
        ## estimate from gyro
        ## E[wwT] makes covariance zero-mean?

        return extOmega, extDeltaFi


    @classmethod
    def update_position(cls,
                        relpos_estimator: SequentialLeastSquare,
                        this_state: StateDataExtLeastSquare,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData,
                        forgetting_factor_position,
                        use_child_gyro=False):

        if this_state.is_jointposition_registered():
            ## alias
            E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
            E_R_kron_R = NoiseCovarianceHelper.calc_E_R_kron_R(this_state.bingham_param.Eq4th)
            ## estimate from force
            joint_wrt_i = this_state.jointposition_wrt_thisframe
            joint_wrt_j = this_state.jointposition_wrt_childframe
            position = joint_wrt_i.position - E_R @ joint_wrt_j.position
            cov_position = joint_wrt_i.covariance \
                        + NoiseCovarianceHelper.calc_Cov_Ax(E_R_kron_R, E_R,
                                                            joint_wrt_j.covariance,
                                                            joint_wrt_j.position)
            estimation = NormalDistributionData(position, cov_position)
        else:
            # import rospy
            # # if np.linalg.norm(this_obsdata.E_gyro) > 1.:
            # rospy.logerr(cls.determinant_of_coeffmat(this_obsdata))
            # if cls.determinant_of_coeffmat(this_obsdata) < -1.:
            if True:
                # ## alias
                # f_i = this_obsdata.E_force
                # f_j = child_obsdata.E_force
                # dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
                # wcmsq_i = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_gyro, this_obsdata.Cov_gyro)
                # ## estimate from gyro
                # Omega = wcmsq_i + dwcm_i
                # DeltaFi = np.dot(E_R, f_j) - f_i
                extOmega, DeltaFi = cls.calc_gyro_estimate_coeffs(this_state, this_obsdata, child_obsdata, use_child_gyro=use_child_gyro)
                relpos_estimator.update(extOmega, DeltaFi, forgetting_factor=forgetting_factor_position,
                                       #diff_cov = cls.calc_covariance_of_diff(relpos_estimator, this_state, this_obsdata, child_obsdata)
                                       )
            estimation = relpos_estimator.get_estimates()

        return estimation
    

    @staticmethod
    def calc_covariance_of_diff(
                        relpos_estimator: SequentialLeastSquare,
                        this_state: StateDataExtLeastSquare,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData):
        _E_r = relpos_estimator.param.position
        _cov_r = relpos_estimator.param.covariance

        _E_r_i = _E_r[:3]
        _cov_r_i = _cov_r[:3, :3]
        _E_r_j = _E_r[3:]
        _cov_r_j = _cov_r[3:, 3:]

        sum_inv = np.linalg.inv(_cov_r_i + _cov_r_j)
        cov_r = _cov_r_i @ sum_inv @ _cov_r_j
        E_r = _cov_r_j @ sum_inv @ _E_r_i + _cov_r_i @ sum_inv @ _E_r_j

        # E_r_rT = cov_r + np.outer(E_r,E_r)
        E_w = this_obsdata.E_gyro
        E_dw = this_obsdata.E_dgyro
        cov_w = this_obsdata.Cov_gyro
        cov_dw = this_obsdata.Cov_dgyro

        E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
        E_R_kron_R = NoiseCovarianceHelper.calc_E_R_kron_R(this_state.bingham_param.Eq4th)

        E_wcmsq = NoiseCovarianceHelper.calc_E_vCMsq(E_w, cov_w)
        E_dwcm = NoiseCovarianceHelper.calc_E_vCM(E_dw)
        E_a = E_R @ this_obsdata.E_force - child_obsdata.E_force
        # E_a_aT = E_R @ child_obsdata.Cov_force @ E_R.T + this_obsdata.Cov_force
        Cov_a = NoiseCovarianceHelper.calc_Cov_Ax(E_R_kron_R, E_R, child_obsdata.Cov_force, child_obsdata.E_force) + this_obsdata.Cov_force

        E_Omega = E_wcmsq + E_dwcm
        E_diff = E_a - np.dot(E_Omega, E_r)

        # E_OraT = E_Omega @ E_r @ E_a.T

        E_Omega_kron_Omega = NoiseCovarianceHelper.calc_E_vCMsq_kron_vCMsq(E_w, cov_w) + np.kron(E_wcmsq, E_dwcm) + np.kron(E_dwcm, E_wcmsq) + \
                             NoiseCovarianceHelper.calc_E_vCM_kron_vCM(E_dw, cov_dw)
        Cov_Omega_r = NoiseCovarianceHelper.calc_Cov_Ax(E_Omega_kron_Omega, E_Omega, cov_r, E_r)
        Cov_diff = Cov_a + Cov_Omega_r

        print("Cov_diff", Cov_diff)

        # return E_a_aT - E_OraT.T - E_OraT + \
        #         NextStateCovarianceHelper.vecinv_sqmat(np.dot(E_Omega_kron_Omega, NextStateCovarianceHelper.vec_sqmat(E_r_rT)))

        return np.outer(E_diff, E_diff) + Cov_diff
        # return Cov_diff
    

    def update(self,
               this_state: StateDataExtLeastSquare,
               this_obsdata: ObservationData,
               child_obsdata: ObservationData,
               forgetting_factor_rotation,
               forgetting_factor_position):
        
        self.update_rotation(this_state, this_obsdata, child_obsdata, 
                             self.force_bingham, self.gyro_bingham,
                             forgetting_factor_rotation)
        
        if this_state.is_jointposition_registered():
            this_state.bingham_param.update(self.force_bingham.Amat)
        else:
            this_state.bingham_param.update(self.gyro_bingham.Amat)

        this_state.relative_position = self.update_position(self.direct_relpos_estim, this_state,
                                                            this_obsdata, child_obsdata,
                                                            forgetting_factor_position,
                                                            use_child_gyro=self.use_child_gyro)

        return this_state
    

    def reset_estimation(self):
        self.gyro_bingham.update_Amat(np.zeros((4,4)))
        self.force_bingham.update_Amat(np.zeros((4,4)))
        self.direct_relpos_estim.initialize()
