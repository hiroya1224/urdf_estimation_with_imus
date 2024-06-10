#!/usr/bin/env python3
import numpy as np
from ..utils.noisecov_helper import NoiseCovarianceHelper
from ..utils.rotation_helper import RotationHelper
from ..utils.dataclasses import ObservationData, StateDataGeneral, BinghamParameterGeneral, NormalDistributionData
from ..utils.sequential_leastsq import SequentialLeastSquare
from ..estimator.extended_leastsq import BinghamParameterExtLeastSquare

## NOTE: you should check the comments with "CHECK" notes.

class RotationHolder:

    _N = NoiseCovarianceHelper.get_Nmat()
    
    @classmethod
    def calc_bingham_common(cls, av, bv, Cov_Axb_Cyd, prev_rotcov):
        H = NoiseCovarianceHelper.get_Hmat(bv, av)
        Sigma_h = np.dot(cls._N, np.dot(np.kron(Cov_Axb_Cyd, prev_rotcov), cls._N.T))
        return - 0.5 * H.T @ np.linalg.pinv(Sigma_h) @ H
    

    @staticmethod
    def calc_E_AbCd_plain(this_obsdata, child_obsdata, take_expectation=False):
        obs0 = this_obsdata
        obs1 = child_obsdata
        
        if take_expectation:
            tE = 1
        else:
            tE = 0

        ## invert sign of force because raw data of force is inertia force
        EA, Eb, EC, Ed = HDBinghamHelper.create_E_AbCd(obs0.E_gyro, obs1.E_gyro,
                                                    obs0.E_dgyro, obs1.E_dgyro,
                                                    -1.*obs0.E_force, -1.*obs1.E_force,
                                                    tE*obs0.Cov_gyro, tE*obs1.Cov_gyro)
        EAkronA = HDBinghamHelper.calc_E_A_kron_A(obs0.E_gyro, obs0.E_dgyro, tE*obs0.Cov_gyro, tE*obs0.Cov_dgyro)
        ECkronC = HDBinghamHelper.calc_E_A_kron_A(obs1.E_gyro, obs1.E_dgyro, tE*obs1.Cov_gyro, tE*obs1.Cov_dgyro)
        EbkronA = np.kron(Eb.reshape(-1,1), EA)
        EdkronC = np.kron(Ed.reshape(-1,1), EC)

        return EA, Eb, EC, Ed, EAkronA, ECkronC, EbkronA, EdkronC


    @staticmethod
    def calc_E_AbCd_derivative(this_obsdata, child_obsdata, take_expectation=False):
        obs0 = this_obsdata
        obs1 = child_obsdata
        
        if take_expectation:
            tE = 1
        else:
            tE = 0

        ## invert sign of force because raw data of force is inertia force
        EAder, Ebder, ECder, Edder = HDBinghamHelper.create_E_AbCd_der(obs0.E_gyro, obs1.E_gyro,
                                                obs0.E_dgyro, obs1.E_dgyro,
                                                obs0.E_ddgyro, obs1.E_ddgyro,
                                                -1.*obs0.E_force, -1.*obs1.E_force,
                                                -1.*obs0.E_dforce, -1.*obs1.E_dforce,
                                                tE*obs0.Cov_gyro, tE*obs1.Cov_gyro)
         
        EAderkronAder = HDBinghamHelper.calc_E_Aderiv_kron_Aderiv(obs0.E_gyro, obs0.E_dgyro, obs0.E_ddgyro, 
                                                                tE*obs0.Cov_gyro, tE*obs0.Cov_dgyro, tE*obs0.Cov_ddgyro)
        ECderkronCder = HDBinghamHelper.calc_E_Aderiv_kron_Aderiv(obs1.E_gyro, obs1.E_dgyro, obs1.E_ddgyro, 
                                                                tE*obs1.Cov_gyro, tE*obs1.Cov_dgyro, tE*obs1.Cov_ddgyro)
        EbderkronAder = HDBinghamHelper.calc_E_bderiv_kron_Aderiv(obs0.E_gyro, obs0.E_dgyro, obs0.E_ddgyro, 
                                                                -1.*obs0.E_force, -1.*obs0.E_dforce,
                                                                tE*obs0.Cov_gyro, tE*obs0.Cov_dgyro, tE*obs0.Cov_ddgyro,
                                                                tE*obs0.Cov_force, tE*obs0.Cov_dforce)
        EdderkronCder = HDBinghamHelper.calc_E_bderiv_kron_Aderiv(obs1.E_gyro, obs1.E_dgyro, obs1.E_ddgyro, 
                                                                -1.*obs1.E_force, -1.*obs1.E_dforce,
                                                                tE*obs1.Cov_gyro, tE*obs1.Cov_dgyro, tE*obs1.Cov_ddgyro,
                                                                tE*obs1.Cov_force, tE*obs1.Cov_dforce)
    
        return EAder, Ebder, ECder, Edder, EAderkronAder, ECderkronCder, EbderkronAder, EdderkronCder
    


    @classmethod
    def calc_bingham_plain(cls, this_obsdata, child_obsdata, theta, cov_theta, prev_rotcov, take_expectation=False):
        ## obsv: plain
        obs0 = this_obsdata
        obs1 = child_obsdata

        EA, Eb, EC, Ed, EAkronA, ECkronC, EbkronA, EdkronC = cls.calc_E_AbCd_plain(this_obsdata, child_obsdata, take_expectation=take_expectation)
        Cov_b = obs0.Cov_force
        Cov_d = obs1.Cov_force

        Cov_Axb_Cyd_plain = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
                        EAkronA, ECkronC,
                        EbkronA, EdkronC,
                        EA, EC,
                        Eb, Ed,
                        Cov_b, Cov_d,
                        cov_theta,
                        theta)
        
        av = EA @ theta[:3] + Eb
        bv = EC @ theta[3:] + Ed
        return cls.calc_bingham_common(av, bv, Cov_Axb_Cyd_plain, prev_rotcov)


    @classmethod
    def calc_bingham_derivative(cls, this_obsdata, child_obsdata, theta, cov_theta, prev_rotcov, take_expectation=False):
        obs0 = this_obsdata
        obs1 = child_obsdata

        EAder, Ebder, ECder, Edder, EAderkronAder, ECderkronCder, EbderkronAder, EdderkronCder = cls.calc_E_AbCd_derivative(this_obsdata, child_obsdata, take_expectation=take_expectation)
        Cov_b_der = HDBinghamHelper.calc_Cov_b_der(obs0.E_gyro, -1.*obs0.E_dforce, obs0.Cov_gyro, obs0.Cov_dforce)
        Cov_d_der = HDBinghamHelper.calc_Cov_b_der(obs1.E_gyro, -1.*obs1.E_dforce, obs1.Cov_gyro, obs1.Cov_dforce)
            
        Cov_Axb_Cyd_der = HDBinghamHelper.calc_Cov_Axb_Cyd_from_AbCd(
                                    EAderkronAder, ECderkronCder,
                                    EbderkronAder, EdderkronCder,
                                    EAder, ECder,
                                    Ebder, Edder,
                                    Cov_b_der, Cov_d_der,
                                    cov_theta,
                                    theta)

        av_der = EAder @ theta[:3] + Ebder
        bv_der = ECder @ theta[3:] + Edder
        return cls.calc_bingham_common(av_der, bv_der, Cov_Axb_Cyd_der, prev_rotcov)

    
    @classmethod
    def calc_bingham_gyro(cls, this_obsdata, child_obsdata, theta, cov_theta, prev_rotcov, take_expectation=False):
        obs0 = this_obsdata
        obs1 = child_obsdata
    
        Cov_gyro = np.eye(6)
        Cov_gyro[:3,:3] = obs0.Cov_gyro
        Cov_gyro[3:,3:] = obs1.Cov_gyro

        av_omg = obs0.E_gyro
        bv_omg = obs1.E_gyro
        return cls.calc_bingham_common(av_omg, bv_omg, Cov_gyro, prev_rotcov)

        

class StateDataHighDimBingham(StateDataGeneral):
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



class EstimateImuRelativePoseHighDimBingham:
    def __init__(self,
                 take_expectation=False):
        self.N = NoiseCovarianceHelper.get_Nmat()

        ## initialize
        self.rotation_Amat = np.eye(4) * 1e-9

        RotationHolder


    @staticmethod
    def update_rotation(this_state: StateDataHighDimBingham,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData,
                        force_bingham: BinghamHolderForce,
                        gyro_bingham: BinghamHolderGyro,
                        forgetting_factor_rotation):
        obs0 = this_obsdata
        obs1 = child_obsdata
        
        EMi = HDBinghamHelper.create_denoised_Mtilde(
                    obs0.E_gyro, obs1.E_gyro,
                    obs0.E_dgyro, obs1.E_dgyro,
                    -1.*obs0.E_force, -1.*obs1.E_force,
                    tE*obs0.Cov_gyro, tE*obs1.Cov_gyro,
                    tE*obs0.Cov_dgyro, tE*obs1.Cov_dgyro,
                    tE*obs0.Cov_force, tE*obs1.Cov_force)
        _28d_EMi = HDBinghamHelper.convert_19dim_Mtilde_to_28dim_Mtilde(EMi)

        outerMi = -0.5 * np.outer(_28d_EMi, _28d_EMi)

        MiMiT = MiMiT + outerMi

    
    @staticmethod
    def determinant_of_coeffmat(this_obsdata: ObservationData):
        w  = this_obsdata.E_gyro
        dw = this_obsdata.E_dgyro
        w_cross_dw = np.cross(w, dw)
        return -1. * np.dot(w_cross_dw, w_cross_dw)


    @staticmethod
    def calc_gyro_estimate_coeffs(
                        this_state: StateDataHighDimBingham,
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
                        this_state: StateDataHighDimBingham,
                        this_obsdata: ObservationData,
                        child_obsdata: ObservationData,
                        forgetting_factor_position,
                        use_child_gyro=False):

        extOmega, DeltaFi = cls.calc_gyro_estimate_coeffs(this_state, this_obsdata, child_obsdata, use_child_gyro=use_child_gyro)
        relpos_estimator.update(extOmega, DeltaFi, forgetting_factor=forgetting_factor_position,
                                #diff_cov = cls.calc_covariance_of_diff(relpos_estimator, this_state, this_obsdata, child_obsdata)
                                )
        estimation = relpos_estimator.get_estimates()

        return estimation
    

    def update(self,
               this_state: StateDataHighDimBingham,
               this_obsdata: ObservationData,
               child_obsdata: ObservationData,
               MiMiT):
        
        obs0 = this_obsdata
        obs1 = child_obsdata
        
        EMi = HDBinghamHelper.create_denoised_Mtilde(
                    obs0.E_gyro, obs1.E_gyro,
                    obs0.E_dgyro, obs1.E_dgyro,
                    -1.*obs0.E_force, -1.*obs1.E_force,
                    tE*obs0.Cov_gyro, tE*obs1.Cov_gyro,
                    tE*obs0.Cov_dgyro, tE*obs1.Cov_dgyro,
                    tE*obs0.Cov_force, tE*obs1.Cov_force)
        _28d_EMi = HDBinghamHelper.convert_19dim_Mtilde_to_28dim_Mtilde(EMi)

        outerMi = -0.5 * np.outer(_28d_EMi, _28d_EMi)

        MiMiT = MiMiT + outerMi
    
        theta, theta_covinv = JointPositionEstimator.find_joint_position_params_28d(-1.*MiMiT, optim_eps=1e-12)
        
        try:
            _ = np.linalg.cholesky(theta_covinv)
        except np.linalg.LinAlgError:
            theta = prev_theta
            theta_covinv = prev_covinv_theta

        cov_theta = np.linalg.pinv(theta_covinv)

        liklihood_Amat = HDBinghamHelper.calc_likelihood_Amat_of_28dim_vec_modified_direct(
                                                            -1.*obs0.E_force, -1.*obs1.E_force,
                                                            obs0.E_gyro, obs1.E_gyro,
                                                            obs0.E_dgyro, obs1.E_dgyro,
                                                            obs0.Cov_gyro, obs1.Cov_gyro,
                                                            E_R)
        
        lL, _ = np.linalg.eigh(liklihood_Amat)
        liklihood_Amat_shifted = liklihood_Amat - lL[-1]*np.eye(28)
        MiMiT = MiMiT + liklihood_Amat_shifted
    

    def reset_estimation(self):
        self.gyro_bingham.update_Amat(np.zeros((4,4)))
        self.force_bingham.update_Amat(np.zeros((4,4)))
        self.direct_relpos_estim.initialize()
