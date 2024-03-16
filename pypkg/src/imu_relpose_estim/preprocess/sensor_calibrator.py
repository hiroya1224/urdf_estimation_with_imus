#!/usr/bin/env python3
import numpy as np


class CalibrationContainer:
    def __init__(self, acc_size, omg_size):
        self.acc_size = acc_size
        self.omg_size = omg_size
        cov_size = omg_size

        self.calib_param = {'acc_scaler': None,
                            'acc_bias': None,
                            'gyro_bias': None}
        self.covariances = {'acc': None,
                            'gyro': None,
                            'dgyro': None,}
        
        self.acc_list_for_cov  = [None for _ in range(cov_size)]
        self.omg_list_for_cov  = [None for _ in range(cov_size)]
        self.domg_list_for_cov = [None for _ in range(cov_size)]

        self.acc_list  = [None for _ in range(acc_size)]

        self.vendor_id = ""
    
    def set_vendor_id(self, vendor_id):
        self.vendor_id = vendor_id


class ImuCalibrator:
    """
    Based on "Calibration of MEMS Triaxial Accelerometers Based on the Maximum Likelihood Estimation Method"
    Ellipsoid fittting is based on "Least squares ellipsoid specific fitting"
    """
    def __init__(self):
        pass
        
        
    @staticmethod
    def calc_ellipsoid_param(Dmat, k=4):
        C1 = np.array([
            [-1, k/2 - 1, k/2 - 1, 0, 0, 0],
            [k/2 - 1, -1, k/2 - 1, 0, 0, 0],
            [k/2 - 1, k/2 - 1, -1, 0, 0, 0],
            [0, 0, 0, -k, 0, 0],
            [0, 0, 0, 0, -k, 0],
            [0, 0, 0, 0, 0, -k],
        ])
        # C = np.zeros((10,10))
        # C[:6,:6] = C1

        DDT = Dmat @ Dmat.T
        S11 = DDT[:6, :6]
        S12 = DDT[:6, 6:]
        # S21 = DDT[6:, :6]
        S22 = DDT[6:, 6:]

        M = np.linalg.inv(C1) @ (S11 - S12 @ np.linalg.inv(S22) @ S12.T)

        eigM = np.linalg.eig(M)
        u1 = eigM[1][:, np.argmax(eigM[0])]
        u2 = -np.dot(np.linalg.inv(S22) @ S12.T, u1)
        u = np.hstack([u1,u2])
        
        return u
    
    @staticmethod
    def acc_to_Dmat(acc_dataset):
        assert acc_dataset.shape[0] == 3
        x = acc_dataset[0,:]
        y = acc_dataset[1,:]
        z = acc_dataset[2,:]
        D = np.vstack([
            x**2, y**2, z**2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, np.ones_like(x)
        ])
        return D
    
    @classmethod
    def calc_IMU_calibration_param(cls, acc_dataset):
        Dmat = cls.acc_to_Dmat(acc_dataset)
        a,b,c,f,g,h,p,q,r,d = cls.calc_ellipsoid_param(Dmat)

        E = np.array([
                [a,f,g],
                [f,b,h],
                [g,h,c]]
            )
        F = np.array([p,q,r])

        scale = 1 / (np.dot(F, np.dot(np.linalg.inv(E), F)) - d)

        E = scale * E
        F = scale * F

        ## scale and skew parameter
        R_a = np.linalg.cholesky(E)
        ## bias
        b_a = -np.dot(np.linalg.inv(E), F)

        return R_a, b_a
    

    @staticmethod
    def calc_calibrated_acc(acc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        b_a = calib_param['acc_bias']
        return np.dot(R_a, acc_raw - b_a) * gravity_magnitude
    

    @staticmethod
    def calc_calibrated_dacc(dacc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        return np.dot(R_a, dacc_raw) * gravity_magnitude
    

    @staticmethod
    def calc_calibrated_cov_acc(cov_acc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        return np.dot(np.dot(R_a, cov_acc_raw), R_a.T) * gravity_magnitude**2
    

    @classmethod
    def calc_calibrated_cov_dacc(cls, cov_dacc_raw, calib_param, gravity_magnitude=9.80665):
        return cls.calc_calibrated_cov_acc(cov_dacc_raw, calib_param, gravity_magnitude=gravity_magnitude)
