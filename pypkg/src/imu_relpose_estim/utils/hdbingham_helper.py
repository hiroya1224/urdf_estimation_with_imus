from ._hdbingham_helper import (_calc_likelihood_Hij_28dim_bingham,
                                _calc_likelihood_Hij_28dim_bingham_modified,
                                _calc_E_Aderiv_kron_Aderiv,
                                _calc_E_bderiv_kron_Aderiv,
                                _calc_E_ATA,
                                _calc_E_AderTAder,
                                _calc_E_noise_of_sq_of_Ax_b)
from .noisecov_helper import NoiseCovarianceHelper
import numpy as np

class HDBinghamHelper:
                                        
    #### aliases for _hdbingham_helper ####

    # @staticmethod
    # def calc_likelihood_Hij_28dim_bingham(likelihood_A, likelihood_B, likelihood_C, likelihood_d, likelihood_e):
    #     return _calc_likelihood_Hij_28dim_bingham(likelihood_A, likelihood_B, likelihood_C,
    #                                               likelihood_d, likelihood_e)
    
    @staticmethod
    def calc_likelihood_Hij_28dim_bingham_modified(likelihood_A, likelihood_B, likelihood_C, likelihood_D, likelihood_e, likelihood_f, likelihood_I, likelihood_J):
        return _calc_likelihood_Hij_28dim_bingham_modified(likelihood_A, likelihood_B, likelihood_C, likelihood_D, likelihood_e, likelihood_f, likelihood_I, likelihood_J)

    @staticmethod
    def calc_E_noise_of_sq_of_Ax_b(E_gyro, E_dgyro, E_force,
                                   gyro_noisecov, dgyro_noisecov, force_noisecov):
        return _calc_E_noise_of_sq_of_Ax_b(E_gyro, E_dgyro, E_force,
                                           gyro_noisecov, dgyro_noisecov, force_noisecov)

    @staticmethod
    def calc_E_Aderiv_kron_Aderiv(gyro, dgyro, ddgyro,
                                  gyro_noisecov, dgyro_noisecov, ddgyro_noisecov):
        return _calc_E_Aderiv_kron_Aderiv(gyro, dgyro, ddgyro,
                                          gyro_noisecov, dgyro_noisecov, ddgyro_noisecov)
    

    @staticmethod
    def calc_E_ATA(gyro, dgyro, gyro_noisecov, dgyro_noisecov):
        return _calc_E_ATA(gyro, dgyro, gyro_noisecov, dgyro_noisecov)
    

    @staticmethod
    def calc_E_AderTAder(gyro, dgyro, ddgyro,
                         gyro_noisecov, dgyro_noisecov, ddgyro_noisecov):
        return _calc_E_AderTAder(gyro, dgyro, ddgyro,
                                 gyro_noisecov, dgyro_noisecov, ddgyro_noisecov)
    
    @staticmethod
    def calc_E_bderiv_kron_Aderiv(gyro, dgyro, ddgyro, force, dforce,
                                  gyro_noisecov, dgyro_noisecov, ddgyro_noisecov, force_noisecov, dforce_noisecov):
        return _calc_E_bderiv_kron_Aderiv(gyro, dgyro, ddgyro, force, dforce,
                                          gyro_noisecov, dgyro_noisecov, ddgyro_noisecov, force_noisecov, dforce_noisecov)

    @staticmethod
    def symmat_to_28dim_vec(A):
        def trius(M):
            return np.array([M[0,0], 2*M[0,1], 2*M[0,2], M[1,1], 2*M[1,2], M[2,2]])
        
        a = trius(A[0:3,0:3])
        b = 2*A[0:3,6]
        c = trius(A[3:6,3:6])
        d = 2*A[3:6,6]
        e = 2*A[0:3,3:6].flatten()
        f = A[6,6]
        
        return np.hstack([a,b,c,d,e,f])
    

    # @classmethod
    # def calc_likelihood_Amat_of_28dim_vec_from_ABCde(cls, A, B, C, d, e, EqqT):
    #     P0 = np.hstack([A, np.zeros((3,3)), e.reshape(-1,1)])
    #     P0TP0_vec = cls.symmat_to_28dim_vec(np.dot(P0.T, P0))
    #     Hij = cls.calc_likelihood_Hij_28dim_bingham(A, B, C, d, e)

    #     result = -0.5 * np.outer(P0TP0_vec, P0TP0_vec)
    #     for i in range(4):
    #         for j in range(4):
    #             gij = np.outer(Hij[:,i,j], P0TP0_vec)
    #             result += -0.5 * EqqT[i,j] * gij
        
    #     return result

    @classmethod
    def calc_likelihood_Amat_of_28dim_vec_from_ABCde_modified(cls, A, B, C, D, e, f, EqqT):
        G = np.dot(A.T, A) + np.dot(C.T, C)
        H = np.dot(B.T, B) + np.dot(D.T, D)
        I = A + C
        J = B + D
        k = np.dot(e,e) + np.dot(f,f)

        P0TP0_half = np.vstack([
            np.hstack([G / 2, np.zeros((3,3)), np.dot(I.T, f).reshape(-1,1)]),
            np.hstack([np.zeros((3,3)), H / 2, np.dot(J.T, e).reshape(-1,1)]),
            np.hstack([np.zeros(6), k]),
        ])
        P0TP0 = P0TP0_half + P0TP0_half.T
        P0TP0_vec = cls.symmat_to_28dim_vec(P0TP0)
        Hij = cls.calc_likelihood_Hij_28dim_bingham_modified(A, B, C, D, e, f, I, J)

        result = np.outer(P0TP0_vec, P0TP0_vec)
        for i in range(4):
            for j in range(4):
                gij = np.outer(Hij[:,i,j], P0TP0_vec)
                result += -1. * EqqT[i,j] * gij
        
        return -0.5 * result
    

    @classmethod
    def calc_likelihood_Amat_of_28dim_vec_from_ABCde_modified_direct(cls, A, B, C, D, e, f, E_R):
        G = np.dot(A.T, A) + np.dot(C.T, C)
        H = np.dot(B.T, B) + np.dot(D.T, D)
        I = A + C
        J = B + D
        k = np.dot(e,e) + np.dot(f,f)

        P01 = -np.dot(A.T, np.dot(E_R, B)) - np.dot(C.T, np.dot(E_R, D))
        P02 = np.dot(I.T, f - np.dot(E_R, e))
        P12 = np.dot(J.T, e - np.dot(E_R, f))
        P22 = 2*k - 4*np.dot(f, np.dot(E_R, e))

        PTP_half = np.vstack([
            np.hstack([G / 2, P01, P02.reshape(-1,1)]),
            np.hstack([np.zeros((3,3)), H / 2, P12.reshape(-1,1)]),
            np.hstack([np.zeros(6), P22 / 2]),
        ])
        PTP = PTP_half + PTP_half.T
        PTP_vec = cls.symmat_to_28dim_vec(PTP)

        return -0.5 * np.outer(PTP_vec, PTP_vec)

    # @classmethod
    # def calc_likelihood_Amat_of_28dim_vec(cls, imu0_acc_inertial,
    #                                            imu1_acc_inertial,
    #                                            imu0_gyr, imu1_gyr,
    #                                            imu0_dgyr, 
    #                                            imu0_gyr_cov,
    #                                            CovQ):
    #     omega_i_cmsq = NoiseCovarianceHelper.calc_E_vCMsq(imu0_gyr, imu0_gyr_cov)
    #     omega_i_cm = NoiseCovarianceHelper.calc_E_vCM(imu0_gyr)
    #     omega_j_cm = NoiseCovarianceHelper.calc_E_vCM(imu1_gyr)
    #     domega_i_cm = NoiseCovarianceHelper.calc_E_vCM(imu0_dgyr)

    #     A = omega_i_cmsq + 2*omega_i_cm + domega_i_cm + np.eye(3)
    #     B = omega_i_cmsq 
    #     C = omega_j_cm
    #     d = imu1_acc_inertial
    #     e = imu0_acc_inertial

    #     return cls.calc_likelihood_Amat_of_28dim_vec_from_ABCde(A,B,C,d,e,CovQ)

    @classmethod
    def calc_likelihood_Amat_of_28dim_vec_helper(cls, imu0_acc_inertial,
                                               imu1_acc_inertial,
                                               imu0_gyr, imu1_gyr,
                                               imu0_dgyr, imu1_dgyr,
                                               imu0_gyr_cov, imu1_gyr_cov):
        omega_i_cmsq = NoiseCovarianceHelper.calc_E_vCMsq(imu0_gyr, imu0_gyr_cov)
        omega_i_cm = NoiseCovarianceHelper.calc_E_vCM(imu0_gyr)
        domega_i_cm = NoiseCovarianceHelper.calc_E_vCM(imu0_dgyr)

        omega_j_cmsq = NoiseCovarianceHelper.calc_E_vCMsq(imu1_gyr, imu1_gyr_cov)
        omega_j_cm = NoiseCovarianceHelper.calc_E_vCM(imu1_gyr)
        domega_j_cm = NoiseCovarianceHelper.calc_E_vCM(imu1_dgyr)

        A = omega_i_cmsq + 2*omega_i_cm + domega_i_cm + np.eye(3)
        B = omega_j_cmsq + domega_j_cm
        C = omega_i_cmsq + domega_i_cm
        D = omega_j_cmsq + 2*omega_j_cm + domega_j_cm + np.eye(3)
        e = imu1_acc_inertial
        f = imu0_acc_inertial

        return A,B,C,D,e,f


    @classmethod
    def calc_likelihood_Amat_of_28dim_vec_modified(cls, imu0_acc_inertial,
                                               imu1_acc_inertial,
                                               imu0_gyr, imu1_gyr,
                                               imu0_dgyr, imu1_dgyr,
                                               imu0_gyr_cov, imu1_gyr_cov,
                                               CovQ):
        A,B,C,D,e,f = cls.calc_likelihood_Amat_of_28dim_vec_helper(imu0_acc_inertial,
                                               imu1_acc_inertial,
                                               imu0_gyr, imu1_gyr,
                                               imu0_dgyr, imu1_dgyr,
                                               imu0_gyr_cov, imu1_gyr_cov)
        return cls.calc_likelihood_Amat_of_28dim_vec_from_ABCde_modified(A,B,C,D,e,f,CovQ)
    

    @classmethod
    def calc_likelihood_Amat_of_28dim_vec_modified_direct(cls, imu0_acc_inertial,
                                               imu1_acc_inertial,
                                               imu0_gyr, imu1_gyr,
                                               imu0_dgyr, imu1_dgyr,
                                               imu0_gyr_cov, imu1_gyr_cov,
                                               E_R):
        A,B,C,D,e,f = cls.calc_likelihood_Amat_of_28dim_vec_helper(imu0_acc_inertial,
                                               imu1_acc_inertial,
                                               imu0_gyr, imu1_gyr,
                                               imu0_dgyr, imu1_dgyr,
                                               imu0_gyr_cov, imu1_gyr_cov)
        return cls.calc_likelihood_Amat_of_28dim_vec_from_ABCde_modified_direct(A,B,C,D,e,f,E_R)


    @staticmethod
    def calc_E_Ader(gyro, dgyro, ddgyro, gyro_noisecov):
        return NoiseCovarianceHelper.calc_E_vCMcb(gyro, gyro_noisecov) \
            + 2*NoiseCovarianceHelper.calc_E_vCM(gyro) @ NoiseCovarianceHelper.calc_E_vCM(dgyro) \
            + NoiseCovarianceHelper.calc_E_vCM(dgyro) @ NoiseCovarianceHelper.calc_E_vCM(gyro) \
            + NoiseCovarianceHelper.calc_E_vCM(ddgyro)
    

    @staticmethod
    def calc_E_A(gyro, dgyro, gyro_noisecov):
        return NoiseCovarianceHelper.calc_E_vCMsq(gyro, gyro_noisecov) + NoiseCovarianceHelper.calc_E_vCM(dgyro)


    @staticmethod
    def calc_E_A_kron_A(gyro, dgyro, gyro_noisecov, dgyro_noisecov):
        vCMsq_kron_vCMsq = NoiseCovarianceHelper.calc_E_vCMsq_kron_vCMsq(gyro, gyro_noisecov)
        vCM_kron_vCM = NoiseCovarianceHelper.calc_E_vCM_kron_vCM(dgyro, dgyro_noisecov)

        E_gCMsq = NoiseCovarianceHelper.calc_E_vCMsq(gyro, gyro_noisecov)
        E_dgCM = NoiseCovarianceHelper.calc_E_vCM(dgyro)

        vCMsq_kron_vCM = np.kron(E_gCMsq, E_dgCM)
        vCM_kron_vCMsq = np.kron(E_dgCM, E_gCMsq)
        return vCMsq_kron_vCMsq + vCMsq_kron_vCM + vCM_kron_vCMsq + vCM_kron_vCM
    

    @staticmethod
    def calc_Cov_Axb_Cyd_from_AbCd(E_Ai_kron_Ai, E_Ci_kron_Ci,
                                   E_bi_kron_Ai, E_di_kron_Ci,
                                   E_Ai, E_Ci,
                                   E_bi, E_di,
                                   Cov_bi, Cov_di,
                                   Cov_xy,
                                   E_xy):
        E_x = E_xy[:3]
        E_y = E_xy[3:]
        Cov_x = Cov_xy[:3,:3]
        Cov_y = Cov_xy[3:,3:]
        Cov_xyT = Cov_xy[:3,3:]

        Cov_Ax = NoiseCovarianceHelper.calc_Cov_Ax(E_Ai_kron_Ai, E_Ai, Cov_x, E_x)
        Cov_Cy = NoiseCovarianceHelper.calc_Cov_Ax(E_Ci_kron_Ci, E_Ci, Cov_y, E_y)
        Cov_Ax_Cy = np.dot(E_Ai, np.dot(Cov_xyT, E_Ci.T))

        Cov_bxTAT = np.dot(E_bi_kron_Ai, E_x).reshape(3,3) - np.outer(E_bi, np.dot(E_Ai, E_x))
        Cov_dyTCT = np.dot(E_di_kron_Ci, E_y).reshape(3,3) - np.outer(E_di, np.dot(E_Ci, E_y))
        Cov_Ax_b = Cov_Ax + Cov_bi + Cov_bxTAT + Cov_bxTAT.T
        Cov_Cy_d = Cov_Cy + Cov_di + Cov_dyTCT + Cov_dyTCT.T

        result = np.zeros((6,6))
        result[:3, :3] = Cov_Ax_b
        result[:3, 3:] = Cov_Ax_Cy
        result[3:, :3] = Cov_Ax_Cy.T
        result[3:, 3:] = Cov_Cy_d

        return result

    
    @classmethod
    def create_E_AbCd(cls,
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_acc, imu1_acc,
                    imu0_gyr_cov, imu1_gyr_cov):
        A = cls.calc_E_A(imu0_gyr, imu0_dgyr, imu0_gyr_cov)
        b = imu0_acc
        C = cls.calc_E_A(imu1_gyr, imu1_dgyr, imu1_gyr_cov)
        d = imu1_acc
        return A, b, C, d
    

    @classmethod
    def create_AbCd(cls,
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_acc, imu1_acc):
        return cls.create_E_AbCd(imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_acc, imu1_acc,
                    np.zeros((3,3)), np.zeros((3,3)))
    

    @classmethod
    def create_E_AbCd_der(cls,
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_ddgyr, imu1_ddgyr,
                    imu0_acc, imu1_acc,
                    imu0_dacc, imu1_dacc,
                    imu0_gyr_cov, imu1_gyr_cov):
        A_der = cls.calc_E_Ader(imu0_gyr, imu0_dgyr, imu0_ddgyr, imu0_gyr_cov)
        b_der = np.dot(NoiseCovarianceHelper.calc_E_vCM(imu0_gyr), imu0_acc) + imu0_dacc
        C_der = cls.calc_E_Ader(imu1_gyr, imu1_dgyr, imu1_ddgyr, imu1_gyr_cov)
        d_der = np.dot(NoiseCovarianceHelper.calc_E_vCM(imu1_gyr), imu1_acc) + imu1_dacc
        return A_der, b_der, C_der, d_der
    

    @classmethod
    def calc_Cov_b_der(cls,
                    imu0_gyr,
                    imu0_dacc,
                    imu0_gyr_cov,
                    imu0_dacc_cov):
        imu0_gyr_kron_gyr = NoiseCovarianceHelper.calc_E_vCM_kron_vCM(imu0_gyr, imu0_gyr_cov)
        return NoiseCovarianceHelper.calc_Cov_Ax(imu0_gyr_kron_gyr, imu0_gyr, imu0_dacc_cov, imu0_dacc)
    


    @classmethod
    def create_AbCd_der(cls,
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_ddgyr, imu1_ddgyr,
                    imu0_acc, imu1_acc,
                    imu0_dacc, imu1_dacc):
        return cls.create_E_AbCd_der(
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_ddgyr, imu1_ddgyr,
                    imu0_acc, imu1_acc,
                    imu0_dacc, imu1_dacc,
                    np.zeros((3,3)), np.zeros((3,3)))
    

    @classmethod
    def create_general_Mtilde(cls, A,C,d,e,f):
        """
        Assume the matrix is written in the following form:
            [[A, O, d],
        M =  [   C, e],
             [sym.  f]]

        Then this function returns Mtilde satisfying:
        x^T M x = Mtilde^T xtilde,
        where xtilde = [xvec^2, xvec, yvec^2, yvec, 1]
        """
        assert len(A.shape) == 2
        assert len(C.shape) == 2
        assert len(d.shape) == 1
        assert len(e.shape) == 1
        assert np.all(np.isclose(A, A.T))
        assert np.all(np.isclose(C, C.T))

        triu_ind = np.triu_indices(3)
        scaler = 2*np.ones((3,3)) - np.eye(3)
        
        A_elem = (A * scaler)[triu_ind[0], triu_ind[1]]
        ATb_elem = 2 * d
        C_elem = (C * scaler)[triu_ind[0], triu_ind[1]]
        CTd_elem = 2 * e
        bd_elem = f

        return np.hstack([A_elem, ATb_elem, C_elem, CTd_elem, bd_elem])


    @classmethod
    def create_general_Mtilde_28dim(cls, A,B,C,d,e,f):
        """
        Assume the matrix is written in the following form:
            [[A, O, d],
        M =  [   C, e],
             [sym.  f]]

        Then this function returns Mtilde satisfying:
        x^T M x = Mtilde^T xtilde,
        where xtilde = [xvec^2, xvec*yvec, yvec^2, xvec, yvec, 1]
        """
        assert len(A.shape) == 2
        assert len(B.shape) == 2
        assert len(C.shape) == 2
        assert len(d.shape) == 1
        assert len(e.shape) == 1
        assert np.all(np.isclose(A, A.T))
        assert np.all(np.isclose(C, C.T))

        triu_ind = np.triu_indices(3)
        scaler = 2*np.ones((3,3)) - np.eye(3)
        
        A_elem = (A * scaler)[triu_ind[0], triu_ind[1]]
        ATb_elem = 2 * d
        C_elem = (C * scaler)[triu_ind[0], triu_ind[1]]
        CTd_elem = 2 * e
        bd_elem = f
        B_elem = B.flatten()

        return np.hstack([A_elem, B_elem, C_elem, ATb_elem, CTd_elem, bd_elem])
    
    

    @classmethod
    def create_Mtilde(cls, A,b,C,d):
        return cls.create_general_Mtilde(A.T @ A,
                                         -C.T @ C,
                                         A.T @ b,
                                         -C.T @ d,
                                         np.dot(b,b) - np.dot(d,d))
    

    @classmethod
    def create_Mtilde_28dim(cls, A,b,C,d):
        return cls.create_general_Mtilde_28dim(A.T @ A,
                                               -C.T @ C,
                                               A.T @ b,
                                               -C.T @ d,
                                               np.dot(b,b) - np.dot(d,d))
            

    @classmethod
    def create_E_Mtilde(cls, E_A, E_b, E_C, E_d, E_ATA, E_CTC, Cov_b, Cov_d):
        triu_ind = np.triu_indices(3)
        scaler = 2*np.ones((3,3)) - np.eye(3)
        
        E_A_elem = (E_ATA * scaler)[triu_ind[0], triu_ind[1]]
        E_ATb_elem = 2 * E_A.T @ E_b
        E_C_elem = -(E_CTC * scaler)[triu_ind[0], triu_ind[1]]
        E_CTd_elem = -2 * E_C.T @ E_d
        E_bd_elem = np.trace(Cov_b + np.outer(E_b, E_b)) - np.trace(Cov_d + np.outer(E_d, E_d))

        return np.hstack([E_A_elem, E_ATb_elem, E_C_elem, E_CTd_elem, E_bd_elem])
    

    @staticmethod
    def convert_19dim_Mtilde_to_28dim_Mtilde(_19dim_Mtilde):
        A, f = _19dim_Mtilde[:18], _19dim_Mtilde[18]
        return np.hstack([A, np.zeros(9), f])
    

    @staticmethod
    def create_Msmall(A,b,C,d, ATA=None, CTC=None):
        M = np.zeros((6,6))
        if ATA is None:
            ATA = A.T @ A
        if CTC is None:
            CTC = C.T @ C

        M[0:3, 0:3] = ATA
        M[3:6, 3:6] = -CTC
        
        c = np.zeros(6)
        c[0:3] = A.T @ b
        c[3:6] = -C.T @ d
        
        u = np.dot(b+d, b-d)
        
        return M, c, u
    

    @classmethod
    def create_homogeneousMsmall(cls, A,b,C,d, ATA=None, CTC=None):
        M, c, u = cls.create_Msmall(A,b,C,d, ATA=ATA, CTC=CTC)
        homoM = np.zeros((7,7))
        homoM[:6,:6] = M
        homoM[6,:6] = c
        homoM[:6,6] = c
        homoM[6,6] = u
        return homoM
    
    
    @classmethod
    def create_denoised_Mtilde(cls,
                    imu0_gyr, imu1_gyr,
                    imu0_dgyr, imu1_dgyr,
                    imu0_acc, imu1_acc,
                    imu0_gyro_cov, imu1_gyro_cov,
                    imu0_dgyro_cov, imu1_dgyro_cov,
                    imu0_force_cov, imu1_force_cov):
        
        def E_noise(E_gyro, E_dgyro, E_force,
                    gyro_noisecov, dgyro_noisecov, force_noisecov):    
            homo = cls.calc_E_noise_of_sq_of_Ax_b(E_gyro, E_dgyro, E_force,
                                           gyro_noisecov, dgyro_noisecov, force_noisecov)
            M = homo[:3,:3]
            c = homo[:3,3]
            u = homo[3,3]
            return M, c, u
        
        dA, db, dci = E_noise(imu0_gyr, imu0_dgyr, imu0_acc, imu0_gyro_cov, imu0_dgyro_cov, imu0_force_cov)
        dC, dd, dcj = E_noise(imu1_gyr, imu1_dgyr, imu1_acc, imu1_gyro_cov, imu1_dgyro_cov, imu1_force_cov)

        A, b, C, d = cls.create_AbCd(imu0_gyr, imu1_gyr,
                        imu0_dgyr, imu1_dgyr,
                        imu0_acc, imu1_acc)
        
        return cls.create_general_Mtilde(A.T @ A - dA,
                                         -C.T @ C + dC,
                                         A.T @ b - db,
                                         -C.T @ d + dd,
                                         np.dot(b,b) - np.dot(d,d) - dci + dcj)
        