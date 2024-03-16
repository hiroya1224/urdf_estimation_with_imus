import numpy as np
from ..utils.dataclasses import NormalDistributionData
# import rospy

class SequentialLeastSquare:
    def __init__(self, param_dim, data_dim, name):
        """
        We set
        X = `coeff_matrix`
        y = `obsv_data`.
        Then we find parameter vector $a$ satisfying $Xa = y$.
        """
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.name = name

        ## initialize `param` and `param_cov`
        self.initialize()

        self.diffs = [np.zeros(6)] * 100
        # self.N = 0
        # self.cov_scaler = 1

        # self.accum_position = np.zeros(3)
        # self.accum_covariance = np.eye(3) * 1e+30
        # self.accum_covariance = np.zeros((3,3))

    def initialize(self):
        self.param = NormalDistributionData(position=np.zeros(self.param_dim),
                                            covariance=np.eye(self.param_dim) * 1e+30,
                                            name=self.name)
        self.N = 0
        
    
    def __repr__(self):
        _str = str(self.__class__).split(".")[1][:-2]
        return "{}(param_dim={}, data_dim={}, name={})".format(_str, self.param_dim, self.data_dim, self.name)
    

    def get_estimates_raw(self):
        return self.param


    def get_estimates(self):

        _E_r = self.param.position
        _cov_r = self.param.covariance

        _E_r_i = _E_r[:3]
        _cov_r_i = _cov_r[:3, :3]
        _E_r_j = _E_r[3:]
        _cov_r_j = _cov_r[3:, 3:]

        sum_inv = np.linalg.inv(_cov_r_i + _cov_r_j)
        cov_r = _cov_r_i @ sum_inv @ _cov_r_j
        E_r = _cov_r_j @ sum_inv @ _E_r_i + _cov_r_i @ sum_inv @ _E_r_j

        covariance = cov_r
        position = E_r

        param = NormalDistributionData(position=position,
                                            covariance=covariance,
                                            name=self.name)
        return param
    

    def update(self, ext_coeff_matrix, obsv_data, diff_cov=np.eye(3), forgetting_factor=None):
        ## assertion
        if not forgetting_factor is None:
            assert not forgetting_factor < 0.
            assert not forgetting_factor > 1.
        
        if type(forgetting_factor) is float:
            gamma = forgetting_factor
        else:
            ## dynamically calculate based on likelihood
            # print("np.linalg.det(P)", np.linalg.det(P))
            # gamma_param = np.exp(-0.5 * np.trace(np.linalg.pinv(XTX @ P @ XTX) @ np.outer(XTy - np.dot(XTX, theta), XTy - np.dot(XTX, theta))))
            # gamma_param = np.min([np.exp(-0.5 * np.linalg.norm(y - np.dot(X, theta))**2), 0.9]) / 0.9
            gamma_param = 0.9
            # print("y - np.dot(X, theta)", y - np.dot(X, theta))
            # print("X @ P @ X.T", X @ P @ X.T)
            # print("(y-Xt).T P (y-Xt)", np.trace(np.linalg.pinv(X @ P @ X.T) @ np.outer(y - np.dot(X, theta), y - np.dot(X, theta))))
            # gamma_param = np.exp(-0.5 * np.trace(np.linalg.pinv(X @ P @ X.T) @ np.outer(y - np.dot(X, theta), y - np.dot(X, theta))))
            print("gamma_param", gamma_param)
            gamma = 0.90 + 0.10*gamma_param
            # if forgetting_factor is None:
            #     print("gamma_param", gamma_param)
        
        gamma = 1.0
        self.param.position, self.param.covariance = self.update_ordinal(ext_coeff_matrix, obsv_data, gamma, diff_cov)
        # print("sqrt trace(self.param.covariance)", np.sqrt(np.trace(self.param.covariance)))

    
    def update_ordinal(self, ext_coeff_matrix, obsv_data, gamma, diff_cov):
        ## alias
        # _Xi = coeff_matrix_i
        # _Xj = coeff_matrix_j
        _y = obsv_data

        # X = np.eye(3)
        # y = np.linalg.pinv(_X) @ _y
        X = ext_coeff_matrix
        y = _y
       
        print("X.shape", X.shape)
        theta = self.param.position
        P = self.param.covariance
        I = np.eye(self.data_dim)

        # _xi = np.dot(_Xi, theta)
        # _xj = np.dot(_Xj, theta)

        # bumbo = np.dot(_xi - _xj, _xi - _xj)
        # if np.isclose(bumbo, 0.0):
        #     s = 0.5
        # else:
        #     s = 0.5 * np.dot(_y - _xj, _xi - _xj) / np.dot(_xi - _xj, _xi - _xj)
        # # Xtheta = s * _xi + (1-s) * _xj
        # X = s * _Xi + (1-s) * _Xj
        diff = y - np.dot(X, theta)

        # raw_gamma = np.dot(diff, diff) / (np.dot(y, y) + np.dot(np.dot(X, theta), np.dot(X, theta)))
        # gamma = max(1 - raw_gamma, 1e-9)
        # print("gamma", gamma)
        # print("raw gamma", raw_gamma)

        self.diffs.append(diff)
        self.diffs.pop(0)

        # import rospy
        # rospy.logwarn("!!!!!!!update_ordinal!!!!!!")
        # rospy.logwarn(np.outer(diff, diff))
        # rospy.logwarn(np.trace(P))
        
        # Gamma = np.eye(self.data_dim) / gamma
        # Gamma = np.eye(3) + np.outer(diff, diff) 
        # Gamma = np.outer(diff, diff) # / (np.linalg.norm(self.diffs) / 3)
        # Gamma = np.eye(3) * np.dot(diff, diff)
        # print("np.outer(diff, diff)", np.outer(diff, diff))
        # Gamma = np.eye(self.data_dim) / gamma * np.exp(-1. * np.dot(diff, diff))
        # Gamma = diff_cov
        diff_length = len(self.diffs)
        Gamma = sum([np.outer(self.diffs[i], self.diffs[i]) for i in range(diff_length)]) / (diff_length - 1)
        # print("diffs", Gamma)
        Gamma_inv = np.linalg.pinv(Gamma)

        # print("diff_cov", diff_cov)
        # print("sqrt(trace(diff_cov))", np.sqrt(np.trace(diff_cov)))
        # print("sqrt(trace(np.outer(diff, diff)))", np.sqrt(np.trace(np.outer(diff, diff))))


        # with open("diff.csv", "a") as f:
        #     f.write("{}\n".format(np.dot(diff, diff)))

        # intermediate vars
        # S = Gamma + X @ P @ X.T
        # T = np.eye(self.param_dim) - P @ X.T @ np.linalg.pinv(S) @ X
        # K = P @ X.T @ np.linalg.pinv(X @ P @ X.T + Gamma)

        # update
        # covariance = np.dot(T, P) / gamma
        # position = np.dot(T, theta) + np.dot(covariance, np.dot(X.T, y))
        # covariance = (I - K @ X) @ P

        # print("P", P)
        # print("Pinv updater", X.T @ Gamma_inv @ X)

        ## commented out lines are numerically unstable
        # S = np.dot(X, P)
        # covariance = P - P @ X.T @ np.linalg.pinv(Gamma + X @ P @ X.T) @ X @ P
        Pinv_updater = X.T @ Gamma_inv @ X

        # # print("Gamma", Gamma)
        # print("Gamma_inv", Gamma_inv)
        # # print("eigh Gamma", np.linalg.eigh(Gamma))
        # # print("eigh Gamma_inv", np.linalg.eigh(Gamma_inv))
        # print("Pinv_updater", Pinv_updater)
        # # print("norm(diff)", np.linalg.norm(diff))

        
        ## Gamma: (m/s^2)^2
        ## X: /s^2

        # self.N = self.N + 1
        # if self.N > 2:
        #     p = 1/(self.N - 1)
        #     # p = 1/(self.N - 3)
        #     self.cov_scaler = (1 - p) * self.cov_scaler + p * np.dot(diff, diff)
        
        # print(self.cov_scaler)

        _covariance = np.linalg.pinv(np.linalg.pinv(P) + Pinv_updater) #* self.cov_scaler
        _position = theta + _covariance @ X.T @ Gamma_inv @ diff

        # fg = np.exp(-np.dot(diff, diff))
        # # fg = 0.999
        # print(fg)
        # self.accum_position = (1 - fg) * _position + fg * self.accum_position
        # self.accum_covariance = (1 - fg)**2 * _covariance + fg**2 * self.accum_covariance

        # position = self.accum_position
        # covariance = self.accum_covariance
        
        # _E_r = _position
        # _cov_r = _covariance

        # _E_r_i = _E_r[:3]
        # _cov_r_i = _cov_r[:3, :3]
        # _E_r_j = _E_r[3:]
        # _cov_r_j = _cov_r[3:, 3:]

        # sum_inv = np.linalg.inv(_cov_r_i + _cov_r_j)
        # cov_r = _cov_r_i @ sum_inv @ _cov_r_j
        # E_r = _cov_r_j @ sum_inv @ _E_r_i + _cov_r_i @ sum_inv @ _E_r_j

        # covariance = cov_r
        # position = E_r

        covariance = _covariance
        position = _position

        print("sqrt trace(covariance)", np.sqrt(np.trace(covariance)))

        return position, covariance
    
    
    def update_robust(self, coeff_matrix, obsv_data, gamma, diff_cov):
        """
        Based on `An Outlier-Robust Kalman Filter` by Gabriel et al.
        """
        def calc_Gamma_from_mu_and_delta(C, y, mu, delta):
            # print("np.dot(delta, delta)", np.dot(delta, delta))
            # raw_gamma = np.dot(delta, delta) / (np.dot(y, y) + np.dot(np.dot(C.T, mu), np.dot(C.T, mu)))
            # gamma = max(1 - raw_gamma, 1e-9)
            Gamma = np.eye(self.data_dim) / gamma
            return Gamma
        
        C = coeff_matrix.T
        y = obsv_data
        P = self.param.covariance
        mu = self.param.position

        delta = obsv_data - np.dot(coeff_matrix, mu)
        # Gamma = calc_Gamma_from_mu_and_delta(C, y, mu, delta)
        Gamma = diff_cov

        R = Gamma
        I = np.eye(self.data_dim)
        s = self.data_dim

        print(np.dot(delta,delta))


        prev_mu = mu
        for _ in range(1000):
            ## this loop takes about 3 times in average
            K = np.linalg.inv(C.T @ P @ C + Gamma) @ C.T @ P
            next_mu = mu + K.T @ (y - C.T @ mu)
            next_P = K.T @ Gamma @ K + (I - K.T @ C.T) @ P @ (I - C @ K)
            delta = y - C.T @ next_mu 
            Gamma = (s*R + np.outer(delta, delta) + C.T @ next_P @ C) / (s + 1)

            if np.linalg.norm(prev_mu - next_mu) < 1e-12:
                break

            prev_mu = next_mu
        
        return next_mu, next_P
