#!/usr/bin/env python3
import numpy as np
import copy

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
        self.coeffs_list_size = 10

        self.mid_idx = int((size - 1) / 2)
        
        self.poly_deg = poly_deg

        self._t = self.UpdatingList([0 for _ in range(size)])
        self.acc = self.UpdatingList([np.zeros(3) for _ in range(size)])
        self.omg = self.UpdatingList([np.zeros(3) for _ in range(size)])

        ## poly_deg-th degree polynomial
        ## (deg + 1, len(omega) + len(acc))
        # self.coeffs = np.zeros((self.poly_deg + 1, 6))
        self.coeffs_list = self.UpdatingList([{
            "coeffs": np.zeros((self.poly_deg + 1, 6)),
            "t_list": copy.copy(self._t.list),
        } for _ in range(self.coeffs_list_size)])


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
    

    @staticmethod
    def interpolated_2nd_derivative_value(coeff, t):
        dts = np.array([(i-1)*i * t**(max(0, i-2)) for i in range(len(coeff))])
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
        
        sorted_indices = sorted(range(len(container._t.list)), key=lambda i: container._t.list[i])
        container._t.list = [container._t.list[i] for i in sorted_indices]
        container.acc.list = [container.acc.list[i] for i in sorted_indices]
        container.omg.list = [container.omg.list[i] for i in sorted_indices]
        
        ## update coeffs
        # container.coeffs = cls.calc_polynomial_coeffs(container)
        container.coeffs_list.update({
            "coeffs": cls.calc_polynomial_coeffs(container),
            "t_list": copy.copy(container._t.list),
        })
    
    @staticmethod
    def find_best_match_coeffs(base_t0, container: DataContainerForFiltering):
        elapsed_times = [np.inf for _ in range(container.coeffs_list_size)]

        for i,d in enumerate(container.coeffs_list.list):
            tlist = d["t_list"]
            if tlist[0] > base_t0:
                continue
            if tlist[-1] < base_t0:
                continue
            elapsed_times[i] = (tlist[container.mid_idx] - base_t0) ** 2

        best_i = np.argmin(elapsed_times)

        # import rospy
        # rospy.logwarn(f"elapsed_times = {elapsed_times}")
        # rospy.logwarn("base_t0 = {}".format(base_t0))

        if elapsed_times[best_i] == np.inf:
            return tlist, None

        best_coeff_dict = container.coeffs_list.list[best_i]
        best_coeffs = best_coeff_dict["coeffs"]
        best_tlist = best_coeff_dict["t_list"]

        # rospy.logerr("---")
        # # rospy.logwarn("best_coeffs = {}".format(best_coeffs))
        # rospy.logwarn("best_tlist = {}".format(best_tlist))
        # rospy.logerr("---")

        return best_tlist, best_coeffs
            

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
        omegas = np.vstack(container.omg.list)
        accs = np.vstack(container.acc.list)

        raw_data = np.hstack([omegas, accs])
        coeffs = np.linalg.pinv(A) @ raw_data

        return coeffs
    

    @classmethod
    def time_interpolation(cls, base_midtime, gapped_midtime, filter_coeffs, t_list):
        ## avoid extrapolation (extrapolation makes estimation results extremely unstable)
        if filter_coeffs is None:
            return None
        
        if base_midtime < t_list[0]:
            # base_midtime = t_list[0]
            return None
        if base_midtime > t_list[-1]:
            # base_midtime = t_list[-1]
            return None

        deltaT = -(gapped_midtime - base_midtime)
        import rospy
        rospy.logwarn(f"deltaT = {deltaT}")

        gyroacc_at_baset0 = cls.interpolated_value(
            filter_coeffs,
            -deltaT
        )
        dgyrodacc_at_baset0 = cls.interpolated_derivative_value(
            filter_coeffs,
            -deltaT
        )
        ddgyroddacc_at_baset0 = cls.interpolated_2nd_derivative_value(
            filter_coeffs,
            -deltaT
        )

        return gyroacc_at_baset0, dgyrodacc_at_baset0, ddgyroddacc_at_baset0