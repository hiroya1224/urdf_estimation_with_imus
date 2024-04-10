#!/usr/bin/env python3

import numpy as np
import rospy
import roslib.packages
import roslaunch
from imu_relpose_estim.preprocess.urdf_parser import UrdfLinkTree

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_enc", action="store_true", help="publish joint encoder using IMUs")
parser.add_argument("__name", default="", nargs="?", help="for roslaunch")
parser.add_argument("__log", default="", nargs="?", help="for roslaunch")
args = parser.parse_args()


rospy.init_node('imu_relpose_estimator_registrator', anonymous=False)

symbolic_robot_description = rospy.get_param("/symbolic_robot_description")

sorted_imu_names = UrdfLinkTree.parse_imu_names_from_robot_description(symbolic_robot_description)

pairs_of_imus = []
imu_pair = []
waiting_for_pair = False
for imu in sorted_imu_names:
    if "child_imu" in imu:
        imu_pair.append(imu)
        waiting_for_pair = True
    if waiting_for_pair and ("parent_imu" in imu):
        ## register pairs of imu
        imu_pair.append(imu)
        pairs_of_imus.append(imu_pair)
        ## initialize
        imu_pair = []
        waiting_for_pair = False

print(pairs_of_imus)

launch = roslaunch.scriptapi.ROSLaunch()
nodes = [roslaunch.core.Node("urdf_estimation_with_imus", "extleastsq_estim_imu_relpose.py", args='--this {} --child {}'.format(*imu_pair)) for imu_pair in pairs_of_imus] 

launch.start()
for node in nodes:
    launch.launch(node)

while not rospy.is_shutdown():
    pass
launch.stop()
del launch

rospy.spin()