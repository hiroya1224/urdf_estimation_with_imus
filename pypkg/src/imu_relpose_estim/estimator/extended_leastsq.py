#!/usr/bin/env python3
import numpy as np
from ..utils.noisecov_helper import NoiseCovarianceHelper
from ..utils.rotation_helper import RotationHelper
from ..utils.dataclasses import ObservationData, StateDataGeneral, BinghamParameterGeneral, NormalDistributionData
from ..utils.sequential_leastsq import SequentialLeastSquare
from ..utils.hdbingham_helper import HDBinghamHelper


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
        # self.jointposition_wrt_thisframe = NormalDistributionData(np.array([0.1, 0., 0.]), np.eye(3)*0.01)
        # self.jointposition_wrt_childframe = NormalDistributionData(np.array([-0.1, 0., 0.]), np.eye(3)*0.01)

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
                          vec3d_before_rotation: np.ndarray,
                          CovInv=None):
        if CovInv is None:
            CovInv = self.CovInv
        Hmat = NoiseCovarianceHelper.get_Hmat(vec3d_before_rotation, vec3d_after_rotation)
        deltaA = -0.5 * Hmat.T @ CovInv @ Hmat
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
    prevnorm = [np.zeros(3),  np.zeros(3)]
    prev_force_CovQ = 0.25 * np.eye(4)
    A_axiss = np.zeros((4,4))

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

        import rospy
        f_i = -1.*this_obsdata.E_force
        f_j = -1.*child_obsdata.E_force
        joint_wrt_i = this_state.jointposition_wrt_thisframe
        joint_wrt_j = this_state.jointposition_wrt_childframe

        wcm_i  = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_gyro)
        wcm_j  = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_gyro)
        dwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_dgyro)
        dwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_dgyro)

        df_i = -1.*this_obsdata.E_dforce
        df_j = -1.*child_obsdata.E_dforce
        ddwcm_i = NoiseCovarianceHelper.calc_E_vCM(this_obsdata.E_ddgyro)
        ddwcm_j = NoiseCovarianceHelper.calc_E_vCM(child_obsdata.E_ddgyro)

        N = NoiseCovarianceHelper.get_Nmat()

        ## obsv: plain
        A = wcm_i@wcm_i + dwcm_i
        b = f_i
        C = wcm_j@wcm_j + dwcm_j
        d = f_j

        EA = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_gyro, this_obsdata.Cov_gyro) + dwcm_i
        Eb = f_i
        EC = NoiseCovarianceHelper.calc_E_vCMsq(this_obsdata.E_dgyro, this_obsdata.Cov_dgyro) + dwcm_j
        Ed = f_j
        Cov_b = this_obsdata.Cov_force
        Cov_d = child_obsdata.Cov_force
        EAkronA = np.kron(EA, EA)
        ECkronC = np.kron(EC, EC)

        cov_theta = np.eye(6)
        cov_theta[:3, :3] = joint_wrt_i.covariance
        cov_theta[3:, 3:] = joint_wrt_j.covariance
        theta = np.hstack([joint_wrt_i.position, joint_wrt_j.position])

        Cov_Axb_Cyd_plain = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
                                EAkronA, ECkronC,
                                np.kron(Eb.reshape(-1,1), EA), np.kron(Ed.reshape(-1,1), EC),
                                EA, EC,
                                Eb, Ed,
                                Cov_b, Cov_d,
                                cov_theta,
                                theta)
        av = A @ theta[:3] + b
        bv = C @ theta[3:] + d
        Sigma_h = np.dot(N, np.dot(np.kron(Cov_Axb_Cyd_plain, np.eye(4) * 0.25), N.T))
        # deltaA_force = self.calc_deltaAmat(av, bv, CovInv=np.linalg.pinv(Sigma_h))

        ## obsv: derivative
        Ader = np.dot(wcm_i, A) + (dwcm_i@wcm_i + wcm_i@dwcm_i + ddwcm_i)
        bder = np.dot(wcm_i, b) + df_i
        Cder = np.dot(wcm_j, C) + (dwcm_j@wcm_j + wcm_j@dwcm_j + ddwcm_j)
        dder = np.dot(wcm_j, d) + df_j
        # Cov_b_der = HDBinghamHelper.calc_Cov_b_der(this_obsdata.E_gyro, -1.*this_obsdata.E_dforce, this_obsdata.Cov_gyro, this_obsdata.Cov_dforce)
        # Cov_d_der = HDBinghamHelper.calc_Cov_b_der(child_obsdata.E_gyro, -1.*child_obsdata.E_dforce, child_obsdata.Cov_gyro, child_obsdata.Cov_dforce)
        # EAderkronAder = np.kron(EAder, EAder)
        # ECderkronCder = np.kron(ECder, ECder)
        # EbderkronAder = np.kron(Ebder.reshape(-1,1), EAder)
        # EdderkronCder = np.kron(Edder.reshape(-1,1), ECder)
        obs0 = this_obsdata
        obs1 = child_obsdata
        EAder, Ebder, ECder, Edder = HDBinghamHelper.create_E_AbCd_der(obs0.E_gyro, obs1.E_gyro,
                                                obs0.E_dgyro, obs1.E_dgyro,
                                                obs0.E_ddgyro, obs1.E_ddgyro,
                                                -1.*obs0.E_force, -1.*obs1.E_force,
                                                -1.*obs0.E_dforce, -1.*obs1.E_dforce,
                                                obs0.Cov_gyro, obs1.Cov_gyro)
        Cov_b_der = HDBinghamHelper.calc_Cov_b_der(obs0.E_gyro, -1.*obs0.E_dforce, obs0.Cov_gyro, obs0.Cov_dforce)
        Cov_d_der = HDBinghamHelper.calc_Cov_b_der(obs1.E_gyro, -1.*obs1.E_dforce, obs1.Cov_gyro, obs1.Cov_dforce)
        
        EAderkronAder = HDBinghamHelper.calc_E_Aderiv_kron_Aderiv(obs0.E_gyro, obs0.E_dgyro, obs0.E_ddgyro, 
                                                                obs0.Cov_gyro, obs0.Cov_dgyro, obs0.Cov_ddgyro)
        ECderkronCder = HDBinghamHelper.calc_E_Aderiv_kron_Aderiv(obs1.E_gyro, obs1.E_dgyro, obs1.E_ddgyro, 
                                                                obs1.Cov_gyro, obs1.Cov_dgyro, obs1.Cov_ddgyro)
        EbderkronAder = HDBinghamHelper.calc_E_bderiv_kron_Aderiv(obs0.E_gyro, obs0.E_dgyro, obs0.E_ddgyro, 
                                                                -1.*obs0.E_force, -1.*obs0.E_dforce,
                                                                obs0.Cov_gyro, obs0.Cov_dgyro, obs0.Cov_ddgyro,
                                                                obs0.Cov_force, obs0.Cov_dforce)
        EdderkronCder = HDBinghamHelper.calc_E_bderiv_kron_Aderiv(obs1.E_gyro, obs1.E_dgyro, obs1.E_ddgyro, 
                                                                -1.*obs1.E_force, -1.*obs1.E_dforce,
                                                                obs1.Cov_gyro, obs1.Cov_dgyro, obs1.Cov_ddgyro,
                                                                obs1.Cov_force, obs1.Cov_dforce)

        Cov_Axb_Cyd_der = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
                                EAderkronAder, ECderkronCder,
                                EbderkronAder, EdderkronCder,
                                EAder, ECder,
                                Ebder, Edder,
                                Cov_b_der, Cov_d_der,
                                cov_theta,
                                theta)

        ## 微分値を見ていても，静止状態ではどのみち無力(ゼロになってしまって線形独立なペアを求められない)
        ## 別の場の情報，たとえば磁場の情報を追加するほかなさそう


        # Cov_Axb_Cyd_der = 2*Cov_Axb_Cyd_plain / this_state.dt**2

        av_der = Ader @ theta[:3] + bder
        bv_der = Cder @ theta[3:] + dder

        av_der = av_der / np.linalg.norm(av_der)
        bv_der = bv_der / np.linalg.norm(bv_der)

        # av_der = (av - self.prevnorm[0]) / this_state.dt
        # bv_der = (bv - self.prevnorm[1]) / this_state.dt

        HTH_decomp = RotationHelper.decompose_Amat(this_state.bingham_param.Amat)
        Eq4th, CovQ = HTH_decomp.E_q4th, HTH_decomp.CovQ
        # H_der = NoiseCovarianceHelper.get_Hmat(bv_der, av_der)

        # rospy.logwarn("Cov_Axb_Cyd_der = \n{}".format(Cov_Axb_Cyd_der))
        # rospy.logwarn("EAderkronAder = \n{}".format(EAderkronAder))
        # rospy.logwarn("ECderkronCder = \n{}".format(ECderkronCder))
        # rospy.logwarn("EbderkronAder = \n{}".format(EbderkronAder))
        
        # Sigma_h_der = np.dot(N, np.dot(np.kron(Cov_Axb_Cyd_der, self.prev_force_CovQ), N.T))
        Sigma_h_der = np.dot(N, np.dot(np.kron(Cov_Axb_Cyd_der, np.eye(4) * 0.25), N.T))
        # Sigma_h_der = np.dot(N, np.dot(np.kron(Cov_Axb_Cyd_der, CovQ), N.T))
        # HTH_der = - 0.5 * H_der.T @ np.linalg.pinv(Sigma_h_der) @ H_der
        
        deltaA_force = self.calc_deltaAmat(av, bv, CovInv=np.linalg.pinv(Sigma_h))
        deltaA_dforce = self.calc_deltaAmat(av_der, bv_der, CovInv=np.linalg.pinv(Sigma_h_der))


        # deltaA_force = self.calc_deltaAmat(av - 0.01*av_der, bv - 0.01*bv_der, CovInv=np.linalg.pinv(Sigma_h))
        # deltaA_dforce = self.calc_deltaAmat(0.01*av + av_der, 0.01*bv + bv_der, CovInv=np.linalg.pinv(Sigma_h_der))


        # av_der2 = (2 * wcm_i + np.eye(3)) @ theta[:3]
        # bv_der2 = -1. * (2 * wcm_j + np.eye(3)) @ theta[3:]

        # Ader2 = np.dot(wcm_i,wcm_i) + dwcm_i + 2*wcm_i + np.eye(3)
        # bder2 = f_i
        # Cder2 = np.dot(wcm_j,wcm_j) + dwcm_j
        # dder2 = f_j
        # av_der2 = Ader2 @ theta[:3] + bder2
        # bv_der2 = Cder2 @ theta[3:] + dder2
        
        # Cov_Axb_Cyd_der2 = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
        #                         np.kron(Ader2,Ader2), np.kron(Cder2,Cder2),
        #                         np.kron(bder2.reshape(-1,1), Ader2), np.kron(dder2.reshape(-1,1), Cder2),
        #                         Ader2, Cder2,
        #                         bder2, dder2,
        #                         this_obsdata.Cov_force, child_obsdata.Cov_force,
        #                         cov_theta,
        #                         theta)
        
        # Sigma_h_der2 = np.dot(N, np.dot(np.kron(Cov_Axb_Cyd_der2, np.eye(4) * 0.25), N.T))
        # deltaA_dforce2 = self.calc_deltaAmat(av_der2, bv_der2, CovInv=Sigma_h_der2)

        # dforce_i = df_i + (dwcm_i@wcm_i + wcm_i@dwcm_i + ddwcm_i) @ joint_wrt_i
        # dforce_j = df_j + (dwcm_j@wcm_j + wcm_j@dwcm_j + ddwcm_j) @ joint_wrt_j
        # # deltaA_dforce = self.calc_deltaAmat(wcm_i @ force_i + dforce_i,
        # #                                     wcm_j @ force_j + dforce_j, CovInv=np.eye(4))
        # deltaA_dforce = self.calc_deltaAmat(av_der, bv_der, CovInv=np.linalg.pinv(Sigma_h_der))
        
        ## predict
            
        rotexp_cov = NoiseCovarianceHelper.calc_rotation_prediction_noise(Eq4th,
                                                                        this_obsdata.E_gyro, child_obsdata.E_gyro,
                                                                        this_obsdata.Cov_gyro, child_obsdata.Cov_gyro,
                                                                        this_state.dt)
        rotpred_noisecov = NoiseCovarianceHelper.calc_nextq_Cov(CovQ, rotexp_cov)
        rotpred_Amat = RotationHelper.decompose_CovQ(rotpred_noisecov).Amat

        # rospy.logwarn("deltaA_force : {}".format(deltaA_force))
        # rospy.logwarn("deltaA_dforce: {}".format(deltaA_dforce))

        ## constraint about joint rotation axis
        E_R = NoiseCovarianceHelper.calc_E_R(Eq4th)
        E_R_kron_R = NoiseCovarianceHelper.calc_E_R_kron_R(Eq4th)
        A_neg_y = np.diag([-1e+9, 0, -1e+9, 0])
        diff_omega = obs1.E_gyro - E_R @ obs0.E_gyro
        
        Cov_diff_omega = obs1.Cov_gyro - NoiseCovarianceHelper.calc_Cov_Ax(E_R_kron_R, E_R, obs0.Cov_gyro, obs0.E_gyro)
        Cov_diff_omega2 = np.eye(6)
        Cov_diff_omega2[:3, :3] = Cov_diff_omega
        Cov_diff_omega2[3:, 3:] = Cov_diff_omega
        A_axis = self.calc_deltaAmat(diff_omega, diff_omega,
                                     CovInv=np.linalg.pinv(
                                         np.dot(N, np.dot(np.kron(Cov_diff_omega2, np.eye(4) / 4.), N.T)))
        )

        update_rate = 1 - np.exp(-1.*np.dot(diff_omega, diff_omega))
        self.A_axiss = A_axis * update_rate + self.A_axiss * (1 - update_rate)

        # rospy.logwarn("rotpred_Amat: {}".format(np.linalg.eigh(rotpred_Amat)[1][:,-1]))
        # constrainted_A = deltaA_force + A_neg_y 
        constrainted_A = deltaA_force + A_neg_y + rotpred_Amat
        # constrainted_A = rotpred_Amat
        # constrainted_A = deltaA_dforce + deltaA_force #+ rotpred_Amat
        # constrainted_A = deltaA_force + rotpred_Amat
        # constrainted_A = deltaA_force + rotpred_Amat
        # constrainted_A = deltaA_dforce 
        # constrainted_A = A_axis + deltaA_force + rotpred_Amat
        # constrainted_A = self.A_axiss

        self.prev_force_CovQ = RotationHelper.decompose_Amat(A_axis).CovQ


        # mode = np.linalg.eigh(deltaA_force)[1][:, -1]
        # Rmat = RotationHelper.quat_to_rotmat(*mode)

        # curr = av - Rmat @ bv

        rospy.logwarn("diff_omega: {}".format(diff_omega))

        # rospy.logwarn("theta: {}".format(theta))
        # rospy.logwarn("EA @ theta[:3] : {}".format(EA @ theta[:3]))

        # # rospy.logwarn("diff    : {}".format( (curr - self.prevnorm) / this_state.dt ))
        # rospy.logwarn("diff_ana: {}".format( av_der - Rmat @ bv_der ))

        self.prevnorm = [av, bv]

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

    
    # @staticmethod
    # def determinant_of_coeffmat(this_obsdata: ObservationData):
    #     w  = this_obsdata.E_gyro
    #     dw = this_obsdata.E_dgyro
    #     w_cross_dw = np.cross(w, dw)
    #     return -1. * np.dot(w_cross_dw, w_cross_dw)


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

        extOmega = np.eye(6)
        ## additional
        if use_child_gyro:
            ## fuse
            # s = 0.5
            # _x = Omega_i
            # _y = E_R @ Omega_j @ E_R.T
            # s = 0.5 * np.dot(DeltaFi - _y, _x - _y) / np.dot(_x - _y, _x - _y)
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
            extOmega, DeltaFi = cls.calc_gyro_estimate_coeffs(this_state, this_obsdata, child_obsdata, use_child_gyro=use_child_gyro)
            relpos_estimator.update(extOmega, DeltaFi, forgetting_factor=forgetting_factor_position)
            estimation = relpos_estimator.get_estimates()

        return estimation
    

    # @staticmethod
    # def calc_covariance_of_diff(
    #                     relpos_estimator: SequentialLeastSquare,
    #                     this_state: StateDataExtLeastSquare,
    #                     this_obsdata: ObservationData,
    #                     child_obsdata: ObservationData):
    #     _E_r = relpos_estimator.param.position
    #     _cov_r = relpos_estimator.param.covariance

    #     _E_r_i = _E_r[:3]
    #     _cov_r_i = _cov_r[:3, :3]
    #     _E_r_j = _E_r[3:]
    #     _cov_r_j = _cov_r[3:, 3:]

    #     sum_inv = np.linalg.inv(_cov_r_i + _cov_r_j)
    #     cov_r = _cov_r_i @ sum_inv @ _cov_r_j
    #     E_r = _cov_r_j @ sum_inv @ _E_r_i + _cov_r_i @ sum_inv @ _E_r_j

    #     # E_r_rT = cov_r + np.outer(E_r,E_r)
    #     E_w = this_obsdata.E_gyro
    #     E_dw = this_obsdata.E_dgyro
    #     cov_w = this_obsdata.Cov_gyro
    #     cov_dw = this_obsdata.Cov_dgyro

    #     E_R = NoiseCovarianceHelper.calc_E_R(this_state.bingham_param.Eq4th)
    #     E_R_kron_R = NoiseCovarianceHelper.calc_E_R_kron_R(this_state.bingham_param.Eq4th)

    #     E_wcmsq = NoiseCovarianceHelper.calc_E_vCMsq(E_w, cov_w)
    #     E_dwcm = NoiseCovarianceHelper.calc_E_vCM(E_dw)
    #     E_a = E_R @ this_obsdata.E_force - child_obsdata.E_force
    #     # E_a_aT = E_R @ child_obsdata.Cov_force @ E_R.T + this_obsdata.Cov_force
    #     Cov_a = NoiseCovarianceHelper.calc_Cov_Ax(E_R_kron_R, E_R, child_obsdata.Cov_force, child_obsdata.E_force) + this_obsdata.Cov_force

    #     E_Omega = E_wcmsq + E_dwcm
    #     E_diff = E_a - np.dot(E_Omega, E_r)

    #     # E_OraT = E_Omega @ E_r @ E_a.T

    #     E_Omega_kron_Omega = NoiseCovarianceHelper.calc_E_vCMsq_kron_vCMsq(E_w, cov_w) + np.kron(E_wcmsq, E_dwcm) + np.kron(E_dwcm, E_wcmsq) + \
    #                          NoiseCovarianceHelper.calc_E_vCM_kron_vCM(E_dw, cov_dw)
    #     Cov_Omega_r = NoiseCovarianceHelper.calc_Cov_Ax(E_Omega_kron_Omega, E_Omega, cov_r, E_r)
    #     Cov_diff = Cov_a + Cov_Omega_r

    #     print("Cov_diff", Cov_diff)

    #     # return E_a_aT - E_OraT.T - E_OraT + \
    #     #         NextStateCovarianceHelper.vecinv_sqmat(np.dot(E_Omega_kron_Omega, NextStateCovarianceHelper.vec_sqmat(E_r_rT)))

    #     return np.outer(E_diff, E_diff) + Cov_diff
    #     # return Cov_diff
    

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
