#!/usr/bin/env python3
import numpy as np

class DataContainerForFiltering:
    class UpdatingList:
        def __init__(self, list_):
            self.list = list_
        
        def __repr__(self):
            return repr(self.list)
        
        def update(self, new_data):
            self.list.append(new_data)
            self.list.pop(0)

    def __init__(self, size=7, poly_deg=5):
        self.size = size
        self.mid_idx = int((size - 1) / 2)
        
        self.poly_deg = poly_deg

        self._t = self.UpdatingList([0 for _ in range(size)])
        self.acc = self.UpdatingList([None for _ in range(size)])
        self.omg = self.UpdatingList([None for _ in range(size)])

        ## poly_deg-th degree polynomial
        ## (deg + 1, len(omega) + len(acc))
        self.coeffs = np.zeros((self.poly_deg + 1, 6))


class ImuPreprocessor:
    def __init__(self):
        pass


    @staticmethod
    def interpolated_value(coeff, t):
        ts = np.array([t**i for i in range(len(coeff))])
        return np.dot(coeff.T, ts)
    

    @staticmethod
    def interpolated_derivative_value(coeff, t):
        dts = np.array([i * t**(max(0, i-1)) for i in range(len(coeff))])
        return np.dot(coeff.T, dts)
    

    @classmethod
    def container_update(cls, msg, container: DataContainerForFiltering):
        container._t.update(msg.header.stamp.to_sec())
        container.acc.update(np.array([msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z]))
        container.omg.update(np.array([msg.angular_velocity.x,
                                  msg.angular_velocity.y,
                                  msg.angular_velocity.z]))
        
        ## update coeffs
        container.coeffs = cls.calc_polynomial_coeffs(container)
        

    @classmethod
    def calc_polynomial_coeffs(cls, container: DataContainerForFiltering):
        ## coeffs
        ## [c_w_x^(0), c_w_y^(0), c_w_z^(0), c_a_x^(0), c_a_y^(0), c_a_z^(0)]
        ## [c_w_x^(1), c_w_y^(1), c_w_z^(1), c_a_x^(1), c_a_y^(1), c_a_z^(1)]
        ## [c_w_x^(2), c_w_y^(2), c_w_z^(2), c_a_x^(2), c_a_y^(2), c_a_z^(2)]
        ## [c_w_x^(3), c_w_y^(3), c_w_z^(3), c_a_x^(3), c_a_y^(3), c_a_z^(3)]

        focusing_t = container._t.list[container.mid_idx]
        shift_t = np.array(container._t.list) - focusing_t

        A = np.vstack([shift_t ** i for i in range(container.poly_deg + 1)]).T

        # if np.any(None in container.omg.list) or np.any(None in container.acc.list):
        #     return None
        try:
            omegas = np.vstack(container.omg.list)
            accs = np.vstack(container.acc.list)
        except ValueError:
            return None

        raw_data = np.hstack([omegas, accs])
        coeffs = np.linalg.pinv(A) @ raw_data

        return coeffs
    

    @classmethod
    def time_interpolation(cls, base_midtime, gapped_midtime, filter_coeffs, t_list):
        ## avoid extrapolation (extrapolation makes estimation results extremely unstable)
        if base_midtime < t_list[0]:
            # base_midtime = t_list[0]
            return None
        if base_midtime > t_list[-1]:
            # base_midtime = t_list[-1]
            return None

        deltaT = gapped_midtime - base_midtime

        gyroacc_at_baset0 = cls.interpolated_value(
            filter_coeffs,
            -deltaT
        )
        dgyrodacc_at_baset0 = cls.interpolated_derivative_value(
            filter_coeffs,
            -deltaT
        )

        return gyroacc_at_baset0, dgyrodacc_at_baset0