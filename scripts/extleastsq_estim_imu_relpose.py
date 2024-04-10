#!/usr/bin/env python3
import rospy
import numpy as np
from urdf_estimation_with_imus.msg import ImuDataFilteredList, PoseWithCovAndBingham
from std_msgs.msg import Empty, Float64
from imu_relpose_estim.preprocess.urdf_parser import UrdfLinkTree

from imu_relpose_estim.utils.dataclasses import ObservationData
from imu_relpose_estim.estimator.extended_leastsq import EstimateImuRelativePoseExtendedLeastSquare, StateDataExtLeastSquare


class EstimateImuRelativePoseExtendedLeastSquareROS:

    def __init__(self, this_imu_name, child_imu_name, symbolic_urdf, use_child_gyro=True) -> None:
        ## initialize state data
        self.state_t = StateDataExtLeastSquare.initialize()

        ## define publisher of estimated results
        self.relative_imu_id = "{}__to__{}".format(this_imu_name, child_imu_name)
        self.pose_publisher = rospy.Publisher("/estimated_relative_pose/{}".format(self.relative_imu_id),
                                              PoseWithCovAndBingham, queue_size=1)
        
        self.this_imu_name = this_imu_name
        self.child_imu_name = child_imu_name

        ## make list of IMUs for registration of joint relative position
        all_IMU_link_pose_wrt_assoc_joint = UrdfLinkTree.get_all_IMU_link_pose_wrt_assoc_joint(symbolic_urdf)
        rospy.logwarn("{}, {}: {}".format(this_imu_name, child_imu_name, self.check_imu_name(this_imu_name, child_imu_name)))

        ## register joint relative position of IMUs are on the same module
        if self.check_imu_name(this_imu_name, child_imu_name):
            joint_position_wrt_i_measured = np.linalg.inv(all_IMU_link_pose_wrt_assoc_joint[this_imu_name])[:3, 3]
            joint_position_wrt_j_measured = np.linalg.inv(all_IMU_link_pose_wrt_assoc_joint[child_imu_name])[:3, 3]
            ## The measured values are based on CAD model, so they are (relatively) reliable.
            ## We set their standard deviation to 2mm.
            self.state_t.jointposition_registration(joint_position_wrt_i_measured, joint_position_wrt_j_measured,
                                                    stddev=2e-3)
            ## joint encoder using IMU
            self.joint_name = self.get_joint_name(this_imu_name, child_imu_name)
            self.imu_encoder_publisher = rospy.Publisher("/joint_imu_encoder/{}".format(self.joint_name),
                                                         Float64, queue_size=1)

        self.state_estimator = EstimateImuRelativePoseExtendedLeastSquare(use_child_gyro=use_child_gyro)
        self.prev_t = None

    
    @classmethod
    def check_imu_name(cls, this_imu, child_imu):
        if not ("_parent_imu" in this_imu):
            return False
        if not ("_child_imu" in child_imu):
            return False
        
        ## check if these two are on the same IMU module
        if cls.get_joint_name(this_imu, child_imu) is None:
            return False
        
        return True

    
    @staticmethod
    def get_joint_name(this_imu, child_imu):
        this_imu_prefix  = this_imu.split("_parent_imu")[0]
        child_imu_prefix = child_imu.split("_child_imu")[0]
        if this_imu_prefix == child_imu_prefix:
            return this_imu_prefix
        return None


    @staticmethod
    def rosmsg_to_obsvdata_dict(msg: ImuDataFilteredList):
        ## helper
        def msg_to_ndarray(msg):
            return np.array([msg.x, msg.y, msg.z])
        
        _dict = dict()
        for m in msg.data:
            ## initialize
            obsvdata = ObservationData.empty()

            ## expectations
            obsvdata.E_force  = msg_to_ndarray(m.acc)
            # obsvdata.E_dforce = msg_to_ndarray(m.dacc_dt)
            obsvdata.E_gyro   = msg_to_ndarray(m.gyro)
            obsvdata.E_dgyro  = msg_to_ndarray(m.dgyro_dt)

            ## covariances
            obsvdata.Cov_force  = np.array(m.acc_covariance).reshape(3,3)
            # obsvdata.Cov_dforce = np.array(m.dacc_covariance).reshape(3,3)
            obsvdata.Cov_gyro   = np.array(m.gyro_covariance).reshape(3,3)
            obsvdata.Cov_dgyro  = np.array(m.dgyro_covariance).reshape(3,3)

            _dict[m.frame_id] = obsvdata

        return _dict


    def get_observation_data(self, msg):
        msg_dict = self.rosmsg_to_obsvdata_dict(msg)

        this_obs_t  = msg_dict[self.this_imu_name]
        child_obs_t = msg_dict[self.child_imu_name]

        return this_obs_t, child_obs_t


    def ros_callback(self, msg, publish=True):
        ## measure dt
        current_t = msg.header.stamp.to_sec()
        if self.prev_t is None:
            self.prev_t = current_t
        dt = current_t - self.prev_t

        # msg_dict = self.rosmsg_to_obsvdata_dict(msg)

        # this_obs_t  = msg_dict[self.this_imu_name]
        # child_obs_t = msg_dict[self.child_imu_name]

        this_obs_t, child_obs_t = self.get_observation_data(msg)

        # print("this_obs_t.E_gyro", this_obs_t.E_gyro)
        # print("child_obs_t.E_gyro", child_obs_t.E_gyro)

        self.state_t.dt = dt
        
        self.state_t = self.state_estimator.update(self.state_t, 
                                                   this_obs_t, child_obs_t,
                                                   1., 1.)
        
        # rospy.logwarn("A = \n{}".format(self.state_t.bingham_param.Amat))
        # rospy.logwarn("relative rotation = \n{}".format(self.state_t.bingham_param.mode_quat))
        # rospy.logwarn("relative position = \n{}".format(self.state_t.relative_position))

        # rospy.logwarn("self.state_t_inv.relative_position.position: {}".format(relative_position))

        if publish:
            self.publish_states(msg.header)

        if self.state_t.is_jointposition_registered():
            enc_msg = Float64()
            rotated_qmode = np.array([[0,0,0,-1],[0,0,-1,0],[0,1,0,0],[1,0,0,0]]) @ self.state_t.bingham_param.mode_quat
            rotated_angle = -2 * np.arctan2(rotated_qmode[2], rotated_qmode[0])
            enc_msg.data = np.mod(rotated_angle + np.pi, 2*np.pi) - np.pi

            if publish:
                self.imu_encoder_publisher.publish(enc_msg)

        self.prev_t = current_t

    
    def publish_states(self, header):
        msg = PoseWithCovAndBingham()

        msg.header = header
        msg.header.frame_id = self.relative_imu_id

        msg.pose.position.x = self.state_t.relative_position.position[0]
        msg.pose.position.y = self.state_t.relative_position.position[1]
        msg.pose.position.z = self.state_t.relative_position.position[2]

        rot_wxyz = self.state_t.bingham_param.mode_quat
        msg.pose.orientation.w = rot_wxyz[0]
        msg.pose.orientation.x = rot_wxyz[1]
        msg.pose.orientation.y = rot_wxyz[2]
        msg.pose.orientation.z = rot_wxyz[3]

        msg.position_covariance = self.state_t.relative_position.covariance.flatten()
        msg.rotation_bingham_parameter = self.state_t.bingham_param.Amat.flatten()

        self.pose_publisher.publish(msg)


    def reset_estimation_callback(self, msg):
        self.state_estimator.reset_estimation()
        rospy.logwarn("{} --> {}: reset estimation".format(self.this_imu_name, self.child_imu_name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--this", type=str, help="frame_id of IMU whose pose to be estimated")
    parser.add_argument("--child", type=str, help="frame_id of IMU whose parent's pose to be estimated")
    parser.add_argument("--only_this_gyro", action="store_false")
    parser.add_argument("__name", default="", nargs="?", help="for roslaunch")
    parser.add_argument("__log", default="", nargs="?", help="for roslaunch")
    args = parser.parse_args()

    rospy.init_node('imu_relpose_estimator_leastsq', anonymous=True)

    symbolic_robot_description = rospy.get_param("/symbolic_robot_description")

    pf = EstimateImuRelativePoseExtendedLeastSquareROS(args.this, args.child, symbolic_robot_description,
                                                       use_child_gyro=args.only_this_gyro)
    imu_sub = rospy.Subscriber("/imu_filtered/calib", ImuDataFilteredList, pf.ros_callback, queue_size=1)
    reset_estimation = rospy.Subscriber("/reset_estimation", Empty, pf.reset_estimation_callback, queue_size=1)
    
    rospy.logwarn("{} --> {}: start to estimate relative pose".format(pf.this_imu_name, pf.child_imu_name))

    rospy.spin()