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
        self.mean = np.zeros(3)

        self.prev_max_t = None

        self.imu_plot_config = {
            "delay": 5.0,
            "this": {
                "timestamps": [],
                "acc": [],
                "gyr": [],
            },
            "child": {
                "timestamps": [],
                "acc": [],
                "gyr": [],
            },
        }

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

        self._plot_title = lambda :{
            "bingham": "Rotation",
            "yz-plane": "YZ-plane",
            "xz-plane": "XZ-plane",
            "xy-plane": "XY-plane",
            "x": "X-axis (mean = {:.6f})".format(self.mean[0]),
            "y": "Y-axis (mean = {:.6f})".format(self.mean[1]),
            "z": "Z-axis (mean = {:.6f})".format(self.mean[2]),
        }

        self.plot_title = self._plot_title()


        self.init_figure(ion=activate_ion)


    def init_figure(self, ion=True):
        if ion:
            plt.ion() # interactive-mode on

        self.fig = plt.figure(figsize=(40,10))

        gs = gridspec.GridSpec(2,7, hspace=0.33)
        grids = [gs[:2, :2]] ## for bingham
        grids.extend([gs[0, i] for i in range(2,5)]) ## proj onto 2D planes
        grids.extend([gs[1, i] for i in range(2,5)]) ## proj onto 1D lines

        for k,g in zip(self.axes_name, grids):
            kw_dict = dict()
            if k == "bingham":
                kw_dict = dict(projection="3d")
            
            self.axes[k] = self.fig.add_subplot(g, **kw_dict)
        
        ## for imu raw (but calibrated) data
        self.imu_this_ax_acc = self.fig.add_subplot(gs[0, 5])
        self.imu_child_ax_acc = self.fig.add_subplot(gs[1, 5])
        self.imu_this_ax_gyr = self.fig.add_subplot(gs[0, 6])
        self.imu_child_ax_gyr = self.fig.add_subplot(gs[1, 6])
        
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

        self.mean = mean.flatten() * 1000.
        self.position_sampled_point = (P.dot(y_s) + mean) * 1000.
        self.rotation_bingham_param = np.array(msg.rotation_bingham_parameter).reshape(4,4)
        self.plot_title = self._plot_title()

    def callback_rawdata(self, msgs, this, child):
        _t = msgs.header.stamp.secs + msgs.header.stamp.nsecs * 10**-9

        for msg in msgs.data:
            if not (msg.frame_id in [this, child]):
                return None
            
            if msg.frame_id == this:
                plot_config = self.imu_plot_config["this"]
            elif msg.frame_id == child:
                plot_config = self.imu_plot_config["child"]
            else:
                raise NotImplementedError

            plot_config["acc"].append([
                msg.acc.x,
                msg.acc.y,
                msg.acc.z,
            ])
            plot_config["gyr"].append([
                msg.gyro.x,
                msg.gyro.y,
                msg.gyro.z,
            ])
            plot_config["timestamps"].append(_t)

            popped = False
            if self.prev_max_t is None:
                self.prev_max_t = max(plot_config["timestamps"])
            elif not _t == max(plot_config["timestamps"]):
                plot_config["timestamps"].pop(-1)
                plot_config["acc"].pop(-1)
                plot_config["gyr"].pop(-1)

            for _ in range(9999):
                last_t = plot_config["timestamps"][-1]
                first_t = plot_config["timestamps"][0]
                if last_t - first_t > self.imu_plot_config["delay"]:
                    plot_config["timestamps"].pop(0)
                    plot_config["acc"].pop(0)
                    plot_config["gyr"].pop(0)
                    popped = True
                else:
                    break
            # if popped:
            #     rospy.logwarn("popped")
