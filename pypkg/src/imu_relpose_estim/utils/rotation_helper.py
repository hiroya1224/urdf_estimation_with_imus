import numpy as np
from .noisecov_helper import NoiseCovarianceHelper
from .dataclasses import BinghamParameterKalmanFilter

class RotationHelper:

    @staticmethod
    def quat_to_rotmat(w,x,y,z):
        n11 = w**2 + x**2 - y**2 - z**2
        n21 = 2*(x*y + w*z)
        n31 = 2*(x*z - w*y)

        n12 = 2*(x*y - w*z)
        n22 = w**2 - x**2 + y**2 - z**2
        n32 = 2*(y*z + w*x)

        n13 = 2*(x*z + w*y)
        n23 = 2*(y*z - w*x)
        n33 = w**2 - x**2 - y**2 + z**2

        return np.array([[n11, n12, n13], [n21, n22, n23], [n31, n32, n33]])
    
    @staticmethod
    def rotmat_to_quat(rotmat):
        ## http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech0052.html
        r11 = rotmat[0,0]
        r12 = rotmat[0,1]
        r13 = rotmat[0,2]
        r21 = rotmat[1,0]
        r22 = rotmat[1,1]
        r23 = rotmat[1,2]
        r31 = rotmat[2,0]
        r32 = rotmat[2,1]
        r33 = rotmat[2,2]

        q0 = ( r11 + r22 + r33 + 1.0) / 4.0
        q1 = ( r11 - r22 - r33 + 1.0) / 4.0
        q2 = (-r11 + r22 - r33 + 1.0) / 4.0
        q3 = (-r11 - r22 + r33 + 1.0) / 4.0
        if q0 < 0.0: q0 = 0.0
        if q1 < 0.0: q1 = 0.0
        if q2 < 0.0: q2 = 0.0
        if q3 < 0.0: q3 = 0.0
        q0 = np.sqrt(q0)
        q1 = np.sqrt(q1)
        q2 = np.sqrt(q2)
        q3 = np.sqrt(q3)
        if q0 >= q1 and q0 >= q2 and q0 >= q3:
            q0 *= +1.0
            q1 *= np.sign(r32 - r23)
            q2 *= np.sign(r13 - r31)
            q3 *= np.sign(r21 - r12)
        elif q1 >= q0 and q1 >= q2 and q1 >= q3:
            q0 *= np.sign(r32 - r23)
            q1 *= +1.0
            q2 *= np.sign(r21 + r12)
            q3 *= np.sign(r13 + r31)
        elif q2 >= q0 and q2 >= q1 and q2 >= q3:
            q0 *= np.sign(r13 - r31)
            q1 *= np.sign(r21 + r12)
            q2 *= +1.0
            q3 *= np.sign(r32 + r23)
        elif q3 >= q0 and q3 >= q1 and q3 >= q2:
            q0 *= np.sign(r21 - r12)
            q1 *= np.sign(r31 + r13)
            q2 *= np.sign(r32 + r23)
            q3 *= +1.0
        else:
            return None
        r = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= r
        q1 /= r
        q2 /= r
        q3 /= r

        return np.array([q0, q1, q2, q3])


    @staticmethod
    def quat_to_rpy(w,x,y,z):
        ## https://qiita.com/srs/items/93d7cc671d206a07deae
        q0q0 = w * w
        q1q1 = x * x
        q2q2 = y * y
        q3q3 = z * z
        q0q1 = w * x
        q0q2 = w * y
        q0q3 = w * z
        q1q2 = x * y
        q1q3 = x * z
        q2q3 = y * z

        roll = np.arctan2((2. * (q2q3 + q0q1)), (q0q0 - q1q1 - q2q2 + q3q3))
        pitch = -np.arcsin((2. * (q1q3 - q0q2)))
        yaw = np.arctan2((2. * (q1q2 + q0q3)), (q0q0 + q1q1 - q2q2 - q3q3))

        return np.array([roll, pitch, yaw])
    

    @staticmethod
    def rpy_to_rotmat(rpy_array: np.ndarray):
        ## helpers
        def yaw_matrix(y):
            return np.array([[np.cos(y), -np.sin(y), 0],
                             [np.sin(y), np.cos(y), 0],
                             [0, 0, 1]])
        def pitch_matrix(p):
            return np.array([[np.cos(p), 0, np.sin(p)],
                             [0, 1, 0],
                             [-np.sin(p), 0, np.cos(p)]])
        def roll_matrix(r):
            return np.array([[1, 0, 0],
                             [0, np.cos(r), -np.sin(r)],
                             [0, np.sin(r), np.cos(r)]])
        def calc_rotation_matrix(r, p, y):
            # return yaw_matrix(y) @ pitch_matrix(p) @ roll_matrix(r)
            return yaw_matrix(y) @ pitch_matrix(p) @ roll_matrix(r)
        
        return calc_rotation_matrix(*rpy_array)


    @staticmethod
    def rpy_to_quat(rpy_array: np.ndarray):
        r,p,y = rpy_array
        cy = np.cos(y / 2.)
        sy = np.sin(y / 2.)
        cp = np.cos(p / 2.)
        sp = np.sin(p / 2.)
        cr = np.cos(r / 2.)
        sr = np.sin(r / 2.)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w,x,y,z])
        

    @staticmethod
    def calc_nc_from_ddnc(ddnc):
        return ddnc[0] + ddnc[4] + ddnc[7] + ddnc[9] + 2. * (ddnc[1] + ddnc[2] + ddnc[3] + ddnc[5] + ddnc[6] + ddnc[8])
    
    
    @staticmethod
    def calc_omegas_from_ddnc(ddnc):
        return np.array([ddnc[0] + ddnc[1] + ddnc[2] + ddnc[3],
                         ddnc[1] + ddnc[4] + ddnc[5] + ddnc[6],
                         ddnc[2] + ddnc[5] + ddnc[7] + ddnc[8],
                         ddnc[3] + ddnc[6] + ddnc[8] + ddnc[9]])
    

    @classmethod
    def decompose_Amat(cls, Amat, N_nc=50):
        Z, M = np.linalg.eigh(Amat)
        mode_quat = M[:, np.argmax(Z)]
        shift_Z = Z - Z.max()

        ## calc ddnc and omegas
        ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(Z, N_nc=N_nc)
        omegas = cls.calc_omegas_from_ddnc(ddnc_10D)

        ## calc 4th moments and covariance
        E_q4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(M, ddnc_10D)
        CovQ = NoiseCovarianceHelper.calc_covQ_from_4thmoment(E_q4th)

        return BinghamParameterKalmanFilter(**{
                "M": M,
                "Z": shift_Z,
                "E_q4th": E_q4th,
                "ddnc_10D": ddnc_10D,
                "omegas": omegas,
                "Amat": Amat,
                "CovQ": CovQ,
                "mode": mode_quat})
    

    @classmethod
    def decompose_CovQ(cls, CovQ, N_nc=50, calc_Eq4th=False):
        omegas, M = np.linalg.eigh(CovQ)
        mode_quat = M[:, np.argmax(omegas)]

        ## calc ddnc and omegas
        shift_Z = cls.calc_approxZ_from_omegas(omegas)
        Amat = np.dot(np.dot(M, np.diag(shift_Z)), M.T)

        E_q4th = None
        ddnc_10D = None
        if calc_Eq4th:
            ddnc_10D = NoiseCovarianceHelper.calc_10D_ddnc(shift_Z, N_nc=N_nc)
            E_q4th = NoiseCovarianceHelper.calc_Bingham_4thMoment(M, ddnc_10D)
        

        return BinghamParameterKalmanFilter(**{
                "M": M,
                "Z": shift_Z,
                "E_q4th": E_q4th,
                "ddnc_10D": ddnc_10D,
                "omegas": omegas,
                "Amat": Amat,
                "CovQ": CovQ,
                "mode": mode_quat})


    @staticmethod
    def calc_approxZ_from_omegas(omegas):
        ## initialize
        Z_init = np.zeros(4)

        ## based on `The BIAS of the maximum likelihood estimators of the von mises-fisher concentration parameters`
        x = 1 - 2*omegas[2] ## Rbar
        if x < 0.53:
            kappa = 2*x + x**3 + (5 * x**5)/6.
        elif x <= 0.85:
            kappa = -0.4 * 1.39*x + 0.43 / (1-x)
        else:
            kappa = 1 / (x**3 - 4*x**2 + 3*x)

        Z_init[0] = -0.5 / omegas[0]
        Z_init[1] = -0.5 / omegas[1]
        Z_init[2] = -2. * kappa

        return Z_init
