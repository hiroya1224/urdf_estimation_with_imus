#!/usr/bin/env python3
import rospy
import numpy as np
from urdf_estimation_with_imus.msg import ImuDataFiltered, ImuDataFilteredList
from sensor_msgs.msg import Imu, MagneticField
import xml.etree.ElementTree as ET

from imu_relpose_estim.preprocess.sensor_dataproc import ImuPreprocessor, DataContainerForFiltering
from imu_relpose_estim.preprocess.urdf_parser import UrdfLinkTree

class ImuPreprocessorROS:
    def __init__(self,
                 symbolic_robot_description,
                 publish_calib=False,
                 sorted_imu_names=None,
                 covariance_param=1.0):
        queue_size = 100

        if sorted_imu_names is None:
            self.sorted_imu_names = UrdfLinkTree.parse_imu_names_from_robot_description(symbolic_robot_description)
        else:
            self.sorted_imu_names = sorted_imu_names

        self.container  = dict([(imu_name, DataContainerForFiltering(size=11, poly_deg=7)) for imu_name in self.sorted_imu_names])
        self.vendor_ids = dict([(imu_name, "") for imu_name in self.sorted_imu_names])
        
        self.topic_name = "/imu_filtered/calib" if publish_calib else "/imu_filtered/raw"
        self.publisher = rospy.Publisher(self.topic_name, ImuDataFilteredList, queue_size=queue_size)

        self.covariance_param = covariance_param

    
    @staticmethod
    def parse_frame_id(raw_frame_id):
        return raw_frame_id.split("__")


    def create_interpolated_msg(self, base_frame_id):

        base_container = self.container[base_frame_id]
        # base_t0 = base_container._t.list[base_container.mid_idx]
        base_list = base_container.coeffs_list.list

        max_candi = []
        min_candi = []
        for frame_id, container in self.container.items():
            max_candi.append(max(container._t.list))
            min_candi.append(min(container._t.list))
        
        range_min = max(min_candi)
        range_max = min(max_candi)
        if range_max < range_min:
            return None
        
        base_t0 = (range_max + range_min) / 2.

        interpolated_result = []

        for frame_id, container in self.container.items():
            # base_t0 = None
            # coeffs = None
            # t_list = None
            t_list, coeffs = ImuPreprocessor.find_best_match_coeffs(base_t0, container)

            # for i,dct in enumerate(base_list[::-1]):
            #     base_tlist = dct["t_list"]
            #     base_t0 = base_tlist[base_container.mid_idx]
            #     rospy.logerr("=== {} ===".format(i))
            #     rospy.logwarn("base_tlist = {}".format(base_tlist))

            #     t_list, coeffs = ImuPreprocessor.find_best_match_coeffs(base_t0, container)

            #     rospy.logwarn("t_list: {}".format(t_list))
            #     # rospy.logwarn("coeffs: {}".format(coeffs))
            #     rospy.logerr("=========")


            #     if coeffs is None:
            #         continue
            #     else:
            #         break
            
            # rospy.logerr("=========")
            if base_t0 is None:
                return None

            time_interpolated = ImuPreprocessor.time_interpolation(base_t0, t_list[container.mid_idx], coeffs, t_list)

            if time_interpolated is None:
                return None

            gyroacc_at_baset0, dgyrodacc_at_baset0, ddgyroddacc_at_baset0 = time_interpolated

            if np.all(np.isclose(gyroacc_at_baset0, 0.0)) or np.all(np.isclose(dgyrodacc_at_baset0, 0.0)):
                continue

            data = ImuDataFiltered()

            data.frame_id  = frame_id
            data.vendor_id = self.vendor_ids[frame_id]

            data.gyro.x = gyroacc_at_baset0[0]
            data.gyro.y = gyroacc_at_baset0[1]
            data.gyro.z = gyroacc_at_baset0[2]
            data.acc.x  = gyroacc_at_baset0[3]
            data.acc.y  = gyroacc_at_baset0[4]
            data.acc.z  = gyroacc_at_baset0[5]
            data.dgyro_dt.x = dgyrodacc_at_baset0[0]
            data.dgyro_dt.y = dgyrodacc_at_baset0[1]
            data.dgyro_dt.z = dgyrodacc_at_baset0[2]
            data.dacc_dt.x  = dgyrodacc_at_baset0[3]
            data.dacc_dt.y  = dgyrodacc_at_baset0[4]
            data.dacc_dt.z  = dgyrodacc_at_baset0[5]
            data.ddgyro_ddt.x = ddgyroddacc_at_baset0[0]
            data.ddgyro_ddt.y = ddgyroddacc_at_baset0[1]
            data.ddgyro_ddt.z = ddgyroddacc_at_baset0[2]

            data.gyro_covariance  = np.eye(3).flatten() * self.covariance_param
            data.dgyro_covariance = np.eye(3).flatten() * self.covariance_param
            data.ddgyro_covariance = np.eye(3).flatten() * self.covariance_param
            data.acc_covariance   = np.eye(3).flatten() * self.covariance_param
            data.dacc_covariance  = np.eye(3).flatten() * self.covariance_param

            interpolated_result.append(data)


        msg = ImuDataFilteredList()
        msg.header.stamp = rospy.Time.from_sec(base_t0)
        msg.header.frame_id = base_frame_id

        msg.data = interpolated_result

        return msg


    def callback(self, msg):
        frame_id, vendor_id = self.parse_frame_id(msg.header.frame_id)
        if not frame_id in self.container.keys():
            rospy.logerr("{} is not in {}".format(frame_id, self.container.keys()))
            return None
        
        self.vendor_ids[frame_id] = vendor_id
        
        container = self.container[frame_id]
        ImuPreprocessor.container_update(msg, container)

        base_frame_id = self.sorted_imu_names[0]

        if frame_id == base_frame_id:
            pub_msg = self.create_interpolated_msg(base_frame_id)
            if pub_msg is not None:
                self.publisher.publish(pub_msg)
            else:
                # rospy.logerr("pub_msg is None")
                pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish_calib", action="store_true", help="when true it directly publish /imu_filtered/calib")
    parser.add_argument("--cov", type=float, default=1.0, help="noise covariance will set to cov * eye(3)")
    parser.add_argument("--subs_imus", type=str, nargs="*", default=None, help="noise covariance will set to cov * eye(3)")
    parser.add_argument("__name", default="imu_preprocessor", nargs="?", help="for roslaunch")
    parser.add_argument("__log", default="", nargs="?", help="for roslaunch")
    args = parser.parse_args()

    rospy.init_node(args.__name, anonymous=False)

    symbolic_robot_description = None
    if args.subs_imus is None:
        symbolic_robot_description = rospy.get_param("/symbolic_robot_description")
    preproc = ImuPreprocessorROS(symbolic_robot_description, args.publish_calib, args.subs_imus, args.cov)
    imu_sub = rospy.Subscriber("/imu", Imu, preproc.callback)
    rospy.logwarn("publishing to {}".format(preproc.topic_name))
    rospy.logwarn("IMU names to be published: {}".format(preproc.sorted_imu_names))
    rospy.logwarn("filtering...")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()