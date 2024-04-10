#!/usr/bin/env python3

import numpy as np
import rospy
from urdf_estimation_with_imus.msg import PoseWithCovAndBingham
from imu_relpose_estim.preprocess.urdf_parser import UrdfLinkTree
from imu_relpose_estim.utils.rotation_helper import RotationHelper
from std_msgs.msg import Empty
import xml.etree.ElementTree as ET
import re
from typing import List
from geometry_msgs.msg import PoseStamped


class RelativePoseGetter:
    def __init__(self, 
                 this_imu, child_imu,
                 relpose_this_link_wrt_this_imu,
                 relpose_child_imu_wrt_child_link):
        ## set target IMU names
        self.this_imu_name = this_imu
        self.child_imu_name = child_imu
        ## relative pose wrt IMUs (homogeneous matrix)
        self.Hmat_this_link_to_this_imu = relpose_this_link_wrt_this_imu
        self.Hmat_child_imu_to_child_link = relpose_child_imu_wrt_child_link

        ## extract joint name
        self.joint_name = child_imu.split("_parent_imu")[0]

        ## threshold of end estimation
        if this_imu == "joint1_child_imu":
            self.thresholdDistance = 5. / 1000.
            self.thresholdAngle = 0.1 * np.pi/180.
        elif this_imu == "joint2_child_imu":
            self.thresholdDistance = 5. / 1000.
            self.thresholdAngle = 0.1 * np.pi/180.
        # else:
        #     self.thresholdDistance = 10. / 1000.
        #     self.thresholdAngle = 1.0 * np.pi/180.

        self.init_setvalue()



    def init_setvalue(self):
        self.pos = np.array([0.2, 0, 0])
        self.rpy = np.zeros(3)
        self.pos_imu = np.zeros(3)
        self.rpy_imu = np.zeros(3)
        self.set_robot_description = False

        ## flags
        self.is_complete_position_estimation = False
        self.is_complete_rotation_estimation = False
        ## AND of is_complete_position_estimation and is_complete_rotation_estimation
        self.is_complete_estimation = False


    def update_completion_flag(self):
        self.is_complete_estimation = self.is_complete_position_estimation and self.is_complete_rotation_estimation

    
    @staticmethod
    def calc_quaternion_to_rpy(w,x,y,z):
        return RotationHelper.quat_to_rpy(w,x,y,z)


    @classmethod
    def calc_geometry_quatmsg_to_rpy(cls, geometry_quat):
        w = geometry_quat.w
        x = geometry_quat.x
        y = geometry_quat.y
        z = geometry_quat.z
        return cls.calc_quaternion_to_rpy(w,x,y,z)
    

    @staticmethod
    def xyz_rpy_to_homogeneous_matrix(xyz_array: np.ndarray,
                                      rpy_array: np.ndarray):
        ## get homogeneous matrix
        Hmat = np.eye(4)
        Hmat[:3,:3] = RotationHelper.rpy_to_rotmat(rpy_array)
        Hmat[:3,3]  = xyz_array

        return Hmat
    

    @classmethod
    def homogeneous_matrix_to_xyz_rpy(cls, homogeneous: np.ndarray):
        rotation = homogeneous[:3,:3]
        position = homogeneous[:3,3]
        quaternion = RotationHelper.rotmat_to_quat(rotation)
        rpy = cls.calc_quaternion_to_rpy(*quaternion)
        return position, rpy
    

    def update_estimation(self,
                          msg: PoseWithCovAndBingham):
        
        self.update_completion_flag()
        if not self.is_complete_estimation:
            poscov = np.array(msg.position_covariance).reshape(3,3)
            bingham = np.array(msg.rotation_bingham_parameter).reshape(4,4)

            Z, _ = np.linalg.eigh(bingham)

            percentileDistance_95perc = 1.96 * np.sqrt(np.trace(poscov))
            percentileAngle_95perc    = 2.77181 / np.sqrt(Z[-1] - Z[-2])

            thresholdDistance = self.thresholdDistance
            thresholdAngle    = self.thresholdAngle

            progress_rotation = min(1, thresholdAngle / percentileAngle_95perc)
            progress_position = min(1, thresholdDistance / percentileDistance_95perc)

            rospy.logwarn("{} --> {}: Estimating... position: {:.3f}%, rotation: {:.3f}%".format(self.this_imu_name, self.child_imu_name, progress_position * 100, progress_rotation * 100))
        
            if percentileDistance_95perc < thresholdDistance:
                self.is_complete_position_estimation = True
                self.pos_imu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            
            if percentileAngle_95perc < thresholdAngle:
                self.is_complete_rotation_estimation = True
                self.rpy_imu = self.calc_geometry_quatmsg_to_rpy(msg.pose.orientation)
            
            self.update_completion_flag()
            if self.is_complete_estimation:
                Hmat_imu = self.xyz_rpy_to_homogeneous_matrix(self.pos_imu, self.rpy_imu)
                Hmat_joint = self.Hmat_this_link_to_this_imu @ Hmat_imu @ self.Hmat_child_imu_to_child_link
                
                self.pos, self.rpy = self.homogeneous_matrix_to_xyz_rpy(Hmat_joint)
                rospy.logwarn("=== Estimation Result ===")
                rospy.logwarn("IMU position:\n{}".format(msg.pose.position))
                rospy.logwarn("IMU orientation:\n{}".format(msg.pose.orientation))
                rospy.logwarn("joint position:{}".format(self.pos))
                rospy.logwarn("joint orientation:{}".format(self.rpy))


    def ros_callback(self,
                     msg: PoseWithCovAndBingham):
        if not self.is_complete_estimation:
            self.update_estimation(msg)


class RobotDescriptionSetter:
    def __init__(self, symbolic_robot_description, all_imu_topic_list, joint_relpos_baseline):
        self.symbolic_robot_description = symbolic_robot_description

        ## parse topic name and define relative pose getters
        this_and_child_imus = [s[0].split("estimated_relative_pose/")[1].split("__to__") for s in all_imu_topic_list]
        
        self.all_link_names_with_depth = UrdfLinkTree.get_nodelist_with_depth_index(symbolic_robot_description)
        IMU_relpose_list = self.get_all_IMU_link_pose_wrt_assoc_joint(symbolic_robot_description)
        self.relpose_getters = dict([(topic_name[0], self.set_relative_pose_getter(this_and_child[0],
                                                              this_and_child[1],
                                                              IMU_relpose_list))
                                for this_and_child, topic_name in zip(this_and_child_imus, all_imu_topic_list)])
        self.joint_relpos_baseline = joint_relpos_baseline

        self.final_result_publisher = rospy.Publisher("/final_estimated_kinematics", PoseStamped, queue_size=1)

        ## reset estimation
        self.reset_estimation_callback(None)

        self.is_robot_description_set = False


    @staticmethod
    def set_relative_pose_getter(this_imu, child_imu,
                                 all_IMU_link_pose_wrt_assoc_joint):
        return RelativePoseGetter(this_imu, child_imu,
                                  all_IMU_link_pose_wrt_assoc_joint[this_imu],
                                  np.linalg.inv(all_IMU_link_pose_wrt_assoc_joint[child_imu]))
    

    def get_parent_link_name(self, imu_name):
        return UrdfLinkTree.find_parent_ordinallink(imu_name, self.all_link_names_with_depth)
        

    @classmethod
    def get_all_IMU_link_pose_wrt_assoc_joint(cls, symbolic_urdf):
        return UrdfLinkTree.get_all_IMU_link_pose_wrt_assoc_joint(symbolic_urdf)
    
    
    # def ros_callback(self,
    #                  msg: PoseWithCovAndBingham):
    #     ## helper
    #     def check_all_estimated():
    #         for rpg in self.relpose_getters.values():
    #             if not rpg.is_complete_estimation:
    #                 return False
    #             return True
        
    #     if not check_all_estimated():
    #         for rpg in self.relpose_getters.values():
    #             rpg.update_estimation(msg)
    #     else:
    #         self.set_robot_description()


    @staticmethod
    def delete_symburdf_note(symbolic_urdf: str):
        note_str1 = "<!-- ================= SYMBOLIC URDF ================= -->\n"
        note_str2 = "<!-- indefinite variables are written in #|this_form|# -->\n"
        symbolic_urdf = symbolic_urdf.replace(note_str1, "")
        symbolic_urdf = symbolic_urdf.replace(note_str2, "")
        return symbolic_urdf


    def set_robot_description(self):
        ## helper
        def set_realized_value(symbolic_urdf: str, **kwargs):
            realized_urdf = self.delete_symburdf_note(symbolic_urdf)
            for name, value in kwargs.items():
                name_str = "#|{}|#".format(name)
                value_str = " ".join([str(s) for s in value])
                realized_urdf = realized_urdf.replace(name_str, value_str)
            return realized_urdf
        
        def create_substitute_dict(relpose_getters):
            result = dict()
            for rg in relpose_getters:
                result[rg.joint_name + "_xyz"] = rg.pos
                result[rg.joint_name + "_rpy"] = rg.rpy
            return result
        
        def publish_final_result(header_stamp, baseline, relpose_getters):
            ## baseline
            for k, item in baseline.items():
                msg = PoseStamped()
                msg.header.stamp = header_stamp

                msg.header.frame_id = k + "__baseline"

                w,x,y,z = RotationHelper.rpy_to_quat(np.array(item["rpy"]))
                msg.pose.orientation.w = w
                msg.pose.orientation.x = x
                msg.pose.orientation.y = y
                msg.pose.orientation.z = z

                px,py,pz = item["pos"]
                msg.pose.position.x = px
                msg.pose.position.y = py
                msg.pose.position.z = pz

                self.final_result_publisher.publish(msg)

            
            ## estimation
            for rg in relpose_getters:
                msg = PoseStamped()
                msg.header.stamp = header_stamp

                msg.header.frame_id = rg.joint_name + "__estimation"

                w,x,y,z = RotationHelper.rpy_to_quat(rg.rpy)
                msg.pose.orientation.w = w
                msg.pose.orientation.x = x
                msg.pose.orientation.y = y
                msg.pose.orientation.z = z

                px,py,pz = rg.pos
                msg.pose.position.x = px
                msg.pose.position.y = py
                msg.pose.position.z = pz

                self.final_result_publisher.publish(msg)



            

        realized_urdf = set_realized_value(self.symbolic_robot_description,
                                           **create_substitute_dict(self.relpose_getters.values()))
        
        unsubstituted_patterns = re.findall("#\|.*?\|#", realized_urdf)
        if len(unsubstituted_patterns) > 0:
            raise ValueError("Insufficiently parsed robot description. {} are/is still remained.".format(unsubstituted_patterns))
        ## set to rosparam
        rospy.set_param("/robot_description", realized_urdf)

        ## for rcb4
        out_urdfpath = '/tmp/robot_realized.urdf'
        with open(out_urdfpath, 'w') as f:
            f.write(realized_urdf)

        ## publish final result
        publish_final_result(rospy.Time.now(), self.joint_relpos_baseline, self.relpose_getters.values())
        rospy.logwarn("published to /final_estimated_kinematics.")

    
    def reset_estimation_callback(self, msg):
        self.is_robot_description_set = False
        for rpg in self.relpose_getters.values():
            rpg.init_setvalue()
            rospy.logwarn("{} --> {}: reset estimation".format(rpg.this_imu_name, rpg.child_imu_name))
        
        if rospy.has_param("/robot_description"):
            rospy.delete_param("/robot_description")


    def realize_value_setter(self):
        ## helper
        def check_all_estimated():
            for rpg in self.relpose_getters.values():
                if not rpg.is_complete_estimation:
                    return False
            return True
        
        if check_all_estimated() and (not self.is_robot_description_set):
            self.set_robot_description()
            self.is_robot_description_set = True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distance", type=float, help="distance when two IMU mode", default=None)
    parser.add_argument("__name", default="", nargs="?", help="for roslaunch")
    parser.add_argument("__log", default="", nargs="?", help="for roslaunch")
    args = parser.parse_args()

    DEBUG = args.debug
    TWOIMU = (args.distance is not None)

    rospy.init_node('robot_description_setter', anonymous=False)

    symbolic_robot_description = rospy.get_param("/symbolic_robot_description")
    joint_relpos_baseline = rospy.get_param("/joint_relpos_from_markers")

    if not DEBUG:
        all_imu_topic_list = []
        while not rospy.is_shutdown():
            rospy.logwarn("Waiting for /estimated_relative_pose...")
            all_imu_topic_list = rospy.get_published_topics('/estimated_relative_pose')
            if len(all_imu_topic_list) > 0:
                break
            rospy.sleep(3.0)
    else:
        if TWOIMU:
            all_imu_topic_list = [["/estimated_relative_pose/imu0__to__imu1", ""]]
        else:
            all_imu_topic_list = [["/estimated_relative_pose/joint{}_child_imu__to__joint{}_parent_imu".format(i,i+1), ""] for i in range(1,3)]


    rospy.logwarn(all_imu_topic_list)

    rds = RobotDescriptionSetter(symbolic_robot_description, all_imu_topic_list, joint_relpos_baseline)

    if not DEBUG:
        rospy.Subscriber("/reset_estimation", Empty, rds.reset_estimation_callback, queue_size=1)
        
        for topic_name, rpg in rds.relpose_getters.items():
            rospy.Subscriber(topic_name, PoseWithCovAndBingham, rpg.ros_callback, queue_size=1)

        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            rds.realize_value_setter()

        rospy.spin()
    
    else:
        if TWOIMU:
            ## it doesn't necessary???
            pass
        else:
            rds.set_robot_description()
    