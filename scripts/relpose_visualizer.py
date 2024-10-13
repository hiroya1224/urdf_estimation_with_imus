#!/usr/bin/env python3
import rospy
import numpy as np
from imu_relpose_estim.visualize.visualizer import Visualizer

## matplotlib == 3.7.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import gridspec

from urdf_estimation_with_imus.msg import PoseWithCovAndBingham, ImuDataFilteredList

import bingham.visualize.SO3s as vSO3
from bingham.distribution import BinghamDistribution

LIM_MIN = -500
LIM_MAX = 500

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--this", type=str, help="frame_id of IMU whose pose to be estimated")
parser.add_argument("--child", type=str, help="frame_id of IMU whose parent's pose to be estimated")
args = parser.parse_args()

rospy.init_node('relpose_estimation_viewer', anonymous=False)

viewer = Visualizer()
pubname_suffix = "{}__to__{}".format(args.this, args.child)
imu_sub = rospy.Subscriber("/estimated_relative_pose/{}".format(pubname_suffix), PoseWithCovAndBingham, viewer.callback)
calib_data = rospy.Subscriber("/imu_filtered/calib", ImuDataFilteredList, lambda msg: viewer.callback_rawdata(msg, args.this, args.child))

print(pubname_suffix)

while not rospy.is_shutdown():

    # BinghamDistribution(Z=Z_INIT, M=M_INIT)
    # vSO3.draw_bingham_3d()

    for k in viewer.axes_name:
        ## initialize
        viewer.axes[k].cla()
        viewer.axes[k].set_title(viewer.plot_title[k])

        if k in ["yz-plane", "xz-plane", "xy-plane"]:
            viewer.axes[k].scatter(*viewer.position_sampled_point[viewer.plot_idx[k],:], alpha=0.5)
            viewer.axes[k].set_xlim((LIM_MIN, LIM_MAX))
            viewer.axes[k].set_ylim((LIM_MIN, LIM_MAX))

        if k in ["x", "y", "z"]:
            viewer.axes[k].hist(viewer.position_sampled_point[viewer.plot_idx[k],:], density=True)
            viewer.axes[k].set_xlim((LIM_MIN, LIM_MAX))
        
        if k in ["bingham"]:
            viewer.draw_bingham_distribution(viewer.axes[k],
                                             viewer.rotation_bingham_param,
                                             quat_gt=np.array([1,0,0,0]),
                                             num_samples=1000,
                                             distance=9, point_of_view=np.array([1,1,1]),)

        viewer.axes[k].set_xlabel(viewer.plot_label[k][0])
        viewer.axes[k].set_ylabel(viewer.plot_label[k][1])

    
    viewer.imu_this_ax_acc.cla()
    viewer.imu_this_ax_gyr.cla()
    viewer.imu_child_ax_acc.cla()
    viewer.imu_child_ax_gyr.cla()

    viewer.imu_this_ax_acc.set_title("Accel. of {}".format(args.this))
    viewer.imu_this_ax_gyr.set_title("Gryo. of {}".format(args.this))
    viewer.imu_child_ax_acc.set_title("Accel. of {}".format(args.child))
    viewer.imu_child_ax_gyr.set_title("Gyro. of {}".format(args.child))

    this_t = viewer.imu_plot_config["this"]["timestamps"]
    if len(this_t) > 0:
        ## acc
        viewer.imu_this_ax_acc.plot(this_t, np.array(viewer.imu_plot_config["this"]["acc"]).T[0], 'r-')
        viewer.imu_this_ax_acc.plot(this_t, np.array(viewer.imu_plot_config["this"]["acc"]).T[1], 'g-')
        viewer.imu_this_ax_acc.plot(this_t, np.array(viewer.imu_plot_config["this"]["acc"]).T[2], 'b-')
        ## gyr
        viewer.imu_this_ax_gyr.plot(this_t, np.array(viewer.imu_plot_config["this"]["gyr"]).T[0], 'r-')
        viewer.imu_this_ax_gyr.plot(this_t, np.array(viewer.imu_plot_config["this"]["gyr"]).T[1], 'g-')
        viewer.imu_this_ax_gyr.plot(this_t, np.array(viewer.imu_plot_config["this"]["gyr"]).T[2], 'b-')

    child_t = viewer.imu_plot_config["child"]["timestamps"]
    if len(child_t) > 0:
        ## acc
        viewer.imu_child_ax_acc.plot(child_t, np.array(viewer.imu_plot_config["child"]["acc"]).T[0], 'r-')
        viewer.imu_child_ax_acc.plot(child_t, np.array(viewer.imu_plot_config["child"]["acc"]).T[1], 'g-')
        viewer.imu_child_ax_acc.plot(child_t, np.array(viewer.imu_plot_config["child"]["acc"]).T[2], 'b-')
        ## gyr
        viewer.imu_child_ax_gyr.plot(child_t, np.array(viewer.imu_plot_config["child"]["gyr"]).T[0], 'r-')
        viewer.imu_child_ax_gyr.plot(child_t, np.array(viewer.imu_plot_config["child"]["gyr"]).T[1], 'g-')
        viewer.imu_child_ax_gyr.plot(child_t, np.array(viewer.imu_plot_config["child"]["gyr"]).T[2], 'b-')

# def draw_bingham_distribution(ax, bdistr, quat_gt, num_samples=500, probability=0.7, **kwargs_to_drawSO3):

    plt.draw()
    plt.pause(0.001)

# spin() simply keeps python from exiting until this node is stopped
rospy.spin()