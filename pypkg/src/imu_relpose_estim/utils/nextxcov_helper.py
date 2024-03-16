from ._nextxcov_helper import (_calc_E_next_x,
                               _calc_E_Xcoeff_kron_Xcoeff,
                               _calc_E_Ucoeff_kron_Ucoeff,
                               _calc_E_Xcoeff_kron_Ucoeff,
                               _calc_E_Ucoeffu_duT,
                               _calc_E_Xcoeffx_duT,
                               _calc_E_du_duT,
                               _calc_E_u_uT,
                               _calc_E_Xcoeff,
                               _calc_E_Ucoeff)
from .noisecov_helper import NoiseCovarianceHelper
from .dataclasses import ObservationData, StateDataKalmanFilter
import numpy as np


def _obsvs_to_arrays(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx):
    dt = obsv.dt

    Ex = state.trans
    Efi = obsv.E_force
    Efj = child_obsv.E_force
    Edfi = obsv.E_dforce
    Edfj = child_obsv.E_dforce
    Ewi = obsv.E_gyro
    Ewj = child_obsv.E_gyro
    Edwi = obsv.E_dgyro

    Cx = state.trans_cov
    Cfi = obsv.Cov_force
    Cfj = child_obsv.Cov_force
    Cdfi = obsv.Cov_dforce
    Cdfj = child_obsv.Cov_dforce
    Cwi = obsv.Cov_gyro
    Cwj = child_obsv.Cov_gyro
    Cdwi = obsv.Cov_dgyro

    Eq4th = state.bingham_param.E_q4th

    return (dt,
            Ex, Efi, Efj, Edfi, Edfj, Ewi, Ewj, Edwi,
            Cx, Cfi, Cfj, Cdfi, Cdfj, Cwi, Cwj, Cdwi,
            Eq4th,
            first_order_approx)

    
class NextStateCovarianceHelper:
    @staticmethod
    def calc_E_next_x(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_next_x(*args)
    

    @staticmethod
    def calc_E_Xcoeff_kron_Xcoeff(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Xcoeff_kron_Xcoeff(*args)
    

    @staticmethod
    def calc_E_Xcoeff_kron_Ucoeff(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Xcoeff_kron_Ucoeff(*args)
    

    @staticmethod
    def calc_E_Ucoeff_kron_Ucoeff(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Ucoeff_kron_Ucoeff(*args)
    

    @staticmethod
    def calc_E_Ucoeffu_duT(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Ucoeffu_duT(*args)
    

    @staticmethod
    def calc_E_Xcoeffx_duT(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Xcoeffx_duT(*args)
    

    @staticmethod
    def calc_E_du_duT(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_du_duT(*args)
    

    @staticmethod
    def calc_E_u_uT(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_u_uT(*args)
    

    @staticmethod
    def calc_E_Xcoeff(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Xcoeff(*args)


    @staticmethod
    def calc_E_Ucoeff(state: StateDataKalmanFilter,
                     obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        args = _obsvs_to_arrays(state,
                                obsv, child_obsv,
                                first_order_approx)
        return _calc_E_Ucoeff(*args)
    

    @staticmethod
    def vec_sqmat(X):
        return X.transpose(0,1).reshape(-1,1)
    
    
    @staticmethod
    def vecinv_sqmat(X):
        msq = X.shape[0]
        m = int(np.sqrt(msq))
        return X.reshape(m,m).transpose(0,1)
    

    @classmethod
    def calc_nextstate_cov(cls, state: StateDataKalmanFilter,
                     this_obsv: ObservationData,
                     child_obsv: ObservationData,
                     first_order_approx=False):
        ## aliases
        args = (state, this_obsv, child_obsv, first_order_approx)
        E_next_x = cls.calc_E_next_x(*args)
        E_Xcoeff_kron_Xcoeff = cls.calc_E_Xcoeff_kron_Xcoeff(*args)
        E_Ucoeff_kron_Ucoeff = cls.calc_E_Ucoeff_kron_Ucoeff(*args)
        E_Xcoeff_kron_Ucoeff = cls.calc_E_Xcoeff_kron_Ucoeff(*args)
        E_Xcoeffx_duT = cls.calc_E_Xcoeffx_duT(*args)
        E_Ucoeffu_duT = cls.calc_E_Ucoeffu_duT(*args)
        E_du_duT = cls.calc_E_du_duT(*args)
        E_u_uT = cls.calc_E_u_uT(*args)
        E_x_xT = state.trans_cov + np.outer(state.trans, state.trans)
        # E_Xcoeff = cls.calc_E_Xcoeff(*args)
        # E_Ucoeff = cls.calc_E_Ucoeff(*args)
        E_x = state.trans
        E_u = np.append(np.zeros(3), np.dot(NoiseCovarianceHelper.calc_E_R(state.bingham_param.E_q4th), child_obsv.E_force) - this_obsv.E_force)
        E_Xcoeffx_UcoeffuT = cls.vecinv_sqmat(np.dot(E_Xcoeff_kron_Ucoeff, cls.vec_sqmat(np.outer(E_x, E_u))))

        dt_sq = this_obsv.dt**2
        if first_order_approx:
            dt_sq = 0.

        return (dt_sq / 2) * (E_Xcoeffx_duT + E_Xcoeffx_duT.T) \
             + (dt_sq / 2) * (E_Ucoeffu_duT + E_Ucoeffu_duT.T) \
             + (dt_sq / 2)**2 * E_du_duT \
             + E_Xcoeffx_UcoeffuT + E_Xcoeffx_UcoeffuT.T \
             + cls.vecinv_sqmat(np.dot(E_Xcoeff_kron_Xcoeff, cls.vec_sqmat(E_x_xT))) \
             + cls.vecinv_sqmat(np.dot(E_Ucoeff_kron_Ucoeff, cls.vec_sqmat(E_u_uT))) \
             - np.outer(E_next_x, E_next_x)