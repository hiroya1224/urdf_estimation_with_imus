#!/usr/bin/env python3
import rospy
import numpy as np

## matplotlib == 3.7.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import gridspec
from bingham.visualize import SO3s as vSO3
from bingham.distribution import BinghamDistribution


class Visualizer:
    def __init__(self, sample_size=1000, activate_ion=True):
        self.sample_size = sample_size

        self.axes_name = ["bingham",
                          "yz-plane", "xz-plane", "xy-plane",
                          "x", "y", "z"]
        self.axes = dict()
        self.position_sampled_point = np.zeros((3, sample_size))
        self.rotation_bingham_param = np.diag([1e-9,0,0,0])

        self.plot_idx = {
            "yz-plane": [1, 2],
            "xz-plane": [0, 2],
            "xy-plane": [0, 1],
            "x": 0,
            "y": 1,
            "z": 2,
        }
        
        self.plot_label = {
            "bingham": (None, None),
            "yz-plane": ("y", "z"),
            "xz-plane": ("x", "z"),
            "xy-plane": ("x", "y"),
            "x": ("x", "density"),
            "y": ("y", "density"),
            "z": ("z", "density"),
        }

        self.plot_title = {
            "bingham": "Rotation",
            "yz-plane": "YZ-plane",
            "xz-plane": "XZ-plane",
            "xy-plane": "XY-plane",
            "x": "X-axis",
            "y": "Y-axis",
            "z": "Z-axis",
        }


        self.init_figure(ion=activate_ion)


    def init_figure(self, ion=True):
        if ion:
            plt.ion() # interactive-mode on

        self.fig = plt.figure(figsize=(30,10))

        gs = gridspec.GridSpec(2,5, hspace=0.33)
        grids = [gs[:2, :2]] ## for bingham
        grids.extend([gs[0, i] for i in range(2,5)]) ## proj onto 2D planes
        grids.extend([gs[1, i] for i in range(2,5)]) ## proj onto 1D lines

        for k,g in zip(self.axes_name, grids):
            kw_dict = dict()
            if k == "bingham":
                kw_dict = dict(projection="3d")
            
            self.axes[k] = self.fig.add_subplot(g, **kw_dict)
        
        if ion:
            ## to close rospy nicely (when close canvas window)
            self.fig.canvas.mpl_connect('close_event', lambda e: rospy.signal_shutdown('finish'))

    
    @staticmethod
    def quat_to_rotmat(w,x,y,z):
        n11 = w**2 + x**2 - y**2 - z**2
        n21 = 2*(x*y + w*z)
        n31 = 2*(x*z - w*y)

        n12 = 2*(x*y - w*z)
        n22 = w**2 - x**2 + y**2 - z**2
        n32 = 2*(y*z + w*x)

        n13 = 2*(x*z + w*y)
        n23 = 2*(y*z - w*x)
        n33 = w**2 - x**2 - y**2 + z**2

        return np.array([[n11, n12, n13], [n21, n22, n23], [n31, n32, n33]])
    

    @staticmethod
    def draw_bingham_distribution(ax, Amat, **kwargs):
        return vSO3.draw_bingham_distribution(ax,
                                              BinghamDistribution(A=Amat),
                                              **kwargs)
    

    def callback(self, msg):
        mean = np.array([[msg.pose.position.x,
                         msg.pose.position.y,
                         msg.pose.position.z
        ]]).T
        cov = np.array(msg.position_covariance).reshape(3,3)
        
        ## position sample
        P = np.linalg.cholesky(cov)
        y_s = np.random.randn(3, self.sample_size)

        self.position_sampled_point = (P.dot(y_s) + mean) * 1000.
        self.rotation_bingham_param = np.array(msg.rotation_bingham_parameter).reshape(4,4)
