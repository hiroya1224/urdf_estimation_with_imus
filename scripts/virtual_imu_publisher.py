#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import xml.etree.ElementTree as ET
from sensor_msgs.msg import Imu, JointState
import pinocchio as pin

from imu_relpose_estim.preprocess.urdf_parser import UrdfLinkTree

class ImuSimulator:
    ## using pinocchio
    def __init__(self,
                 publish_rate,
                 robot_description,
                 gyro_noise_sqrt, acc_noise_sqrt):
        self.gyro_noise_sqrt = gyro_noise_sqrt
        self.acc_noise_sqrt = acc_noise_sqrt

        self.publisher = rospy.Publisher("/imu", Imu, queue_size=1)

        # self.parent_imu_names, self.child_imu_names = self.parse_robot_description(robot_description)
        # self.all_imu_names = self.parent_imu_names + self.child_imu_names + ["link0"]
        self.all_imu_names = UrdfLinkTree.parse_imu_names_from_robot_description(robot_description)
        self.all_link_names_with_depth = UrdfLinkTree.get_nodelist_with_depth_index(robot_description)
        self.robot_model = pin.buildModelFromXML(robot_description)
        self.pin_data = self.robot_model.createData()

        ## for publish rate control
        self.prev_t = None
        self.publish_dt = 1. / publish_rate


    def differentiate_using_pin(self, q, dq, ddq, gravity):
        ## compute forwardKinematics before get frame velocity and acceleration
        model = self.robot_model
        data = self.pin_data
        pin.forwardKinematics(model, data, q, dq, ddq)

        ## process for each IMU
        result_dict = dict()
        for n in self.all_imu_names:
            jointframe = UrdfLinkTree.find_parent_ordinallink(n, self.all_link_names_with_depth)
            jointframe_idx = model.getFrameId(jointframe)
            frame_idx = model.getFrameId(n)

            ## compute gyro, dgyro/dt, accel at joint frame
            gyr_wrt_local = pin.getFrameVelocity(model, data, jointframe_idx, pin.ReferenceFrame.LOCAL).angular
            dgyr_wrt_local = pin.getFrameAcceleration(model, data, jointframe_idx, pin.ReferenceFrame.LOCAL).angular
            acc_wrt_world = pin.getFrameAcceleration(model, data, jointframe_idx, pin.ReferenceFrame.WORLD).linear

            ## compute rotation and displacement to IMU frames
            O_R_A = data.oMf[jointframe_idx].rotation
            O_R_P = data.oMf[frame_idx].rotation
            A_r_P = data.oMf[jointframe_idx].actInv(data.oMf[frame_idx]).translation

            ## calc coefficient matrix for calculate centrifugal force
            ## NOTE: this process is needed because the direct accel calculation of the IMU frame
            ##       returns the same value as that of the parent joint frame.
            gyro_cm = np.cross(np.eye(3), gyr_wrt_local)
            coeff_mat = gyro_cm @ gyro_cm + np.cross(np.eye(3), dgyr_wrt_local)

            ## rotate the result after adding gravity to simulate the accel. measured at IMU frame
            O_acc_P = acc_wrt_world + np.dot(np.dot(O_R_A, coeff_mat), A_r_P) + gravity
            P_acc_P = O_R_P.T @ O_acc_P

            ## set value to resulting dictionary
            result_dict[n] = dict(acc=P_acc_P, gyr=gyr_wrt_local)

        return result_dict

    
    def callback(self, msg):
        ## get current time from called-back message
        current_stamp = msg.header.stamp
        current_t = current_stamp.to_sec()
        if self.prev_t is None:
            self.prev_t = current_t
        
        ## if elapsed time is less than publish_dt, then do nothing
        if current_t - self.prev_t < self.publish_dt:
            return None
        
        ## get joint angle, velocity, and acceleration (effort) from message
        q = np.array(msg.position)
        dq = np.array(msg.velocity)
        ddq = np.array(msg.effort)
        results = self.differentiate_using_pin(q, dq, ddq, gravity=np.array([0, 0, 9.81]))

        for frame_id, imu in results.items():
            ## get differentiation result
            total_acc = imu["acc"]
            gyro = imu["gyr"]

            ## set random noise
            gyro_noise = np.dot(self.gyro_noise_sqrt, np.random.randn(3))
            acc_noise = np.dot(self.acc_noise_sqrt, np.random.randn(3))

            ## calc noise covariance
            gyro_cov = np.dot(self.gyro_noise_sqrt, self.gyro_noise_sqrt.T)
            acc_cov = np.dot(self.acc_noise_sqrt, self.acc_noise_sqrt.T)

            ## add noise to calculated gyro and accel
            noisy_gyro = gyro + gyro_noise
            noisy_acc = total_acc + acc_noise

            ## create Imu message
            msg = Imu()
            msg.header.stamp = current_stamp
            msg.header.frame_id = frame_id
            msg.angular_velocity.x = noisy_gyro[0]
            msg.angular_velocity.y = noisy_gyro[1]
            msg.angular_velocity.z = noisy_gyro[2]
            msg.linear_acceleration.x = noisy_acc[0]
            msg.linear_acceleration.y = noisy_acc[1]
            msg.linear_acceleration.z = noisy_acc[2]
            msg.angular_velocity_covariance = gyro_cov.flatten()
            msg.linear_acceleration_covariance = acc_cov.flatten()

            ## publish
            self.publisher.publish(msg)

            ## set prev_t for publish rate
            self.prev_t = current_t


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubrate", default=60., type=float, help="publish rate [Hz]")
    args = parser.parse_args()

    rospy.init_node('virtual_imu_publisher', anonymous=False)

    robot_description = rospy.get_param("/robot_description")
    publish_rate = args.pubrate

    imu_sim = ImuSimulator(publish_rate,
                           robot_description,
                           acc_noise_sqrt=np.eye(3)*0.01,
                           gyro_noise_sqrt=np.eye(3)*0.01)
    js_sub = rospy.Subscriber("/joint_states", JointState, imu_sim.callback)

    rospy.logwarn("publishing at a rate of {} Hz...".format(publish_rate))

    rospy.spin()