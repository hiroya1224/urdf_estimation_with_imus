#!/usr/bin/env python3
import rospy
import numpy as np
from imu_relpose_estim.visualize.visualizer import Visualizer

## matplotlib == 3.7.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import gridspec

from urdf_estimation_with_imus.msg import PoseWithCovAndBingham

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


# def draw_bingham_distribution(ax, bdistr, quat_gt, num_samples=500, probability=0.7, **kwargs_to_drawSO3):

    plt.draw()
    plt.pause(0.001)

# spin() simply keeps python from exiting until this node is stopped
rospy.spin()