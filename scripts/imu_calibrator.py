#!/usr/bin/env python3
import rospy
import roslib.packages
import numpy as np
import yaml
from urdf_estimation_with_imus.msg import ImuDataFiltered, ImuDataFilteredList
from imu_relpose_estim.preprocess.sensor_calibrator import CalibrationContainer, ImuCalibrator


class ImuCalibratorROS:

    def __init__(self, acc_size, omg_size, calib_param_yaml, force_recalib=False):
        self.container = dict()
        self.acc_size = acc_size
        self.omg_size = omg_size
        self.calib_imu_pub = rospy.Publisher("/imu_filtered/calib", ImuDataFilteredList, queue_size=1)
        self.calibration_step = True
        self.calib_param_yaml = calib_param_yaml

        with open(calib_param_yaml, "r") as yml:
            loaded_calib_config = yaml.load(yml, Loader=yaml.SafeLoader)
        self.calib_config = dict() if loaded_calib_config is None else loaded_calib_config

        self.force_recalib = force_recalib


    def add_new_container(self, frame_id):
        self.container[frame_id] = CalibrationContainer(acc_size=self.acc_size, omg_size=self.omg_size)


    @staticmethod
    def convert_rosmsg_to_3Darray(msg):
        return np.array([msg.x, msg.y, msg.z])
    

    @classmethod
    def get_3darrays_from_msgdata(cls, data):
        acc_raw  = cls.convert_rosmsg_to_3Darray(data.acc)
        # dacc_raw = self.convert_rosmsg_to_3Darray(data.dacc_dt)
        omg_raw  = cls.convert_rosmsg_to_3Darray(data.gyro)
        domg_raw = cls.convert_rosmsg_to_3Darray(data.dgyro_dt)
        return acc_raw, omg_raw, domg_raw
    

    @staticmethod
    def convert_3Darray_to_rosmsg(_3darray, msg):
        msg.x = _3darray[0]
        msg.y = _3darray[1]
        msg.z = _3darray[2]
        return msg
    

    @staticmethod
    def update_list(_list, raw_data):
        _list.append(raw_data)
        _list.pop(0)

    
    def load_precalibrated_param(self, frame_id, container):
        # helper
        def check_dict(calib_config, vendor_id):
            if vendor_id in calib_config.keys():
                loaded_calib_config = calib_config[vendor_id]
                for k in ["acc_scaler", "acc_bias", "gyro_bias"]:
                    if not k in loaded_calib_config.keys():
                        return False
            else:
                return False
            return True
        
        if (not self.force_recalib) and check_dict(self.calib_config, container.vendor_id):
            ## load from yaml
            loaded_calib_config = self.calib_config[container.vendor_id]
            ## set to container params
            container.calib_param['acc_scaler'] = np.array(loaded_calib_config['acc_scaler']).reshape(3,3)
            container.calib_param['acc_bias'] = np.array(loaded_calib_config['acc_bias'])
            container.calib_param['gyro_bias'] = np.array(loaded_calib_config['gyro_bias'])
            rospy.logwarn("{}: loaded pre-calibrated param (vendor_id = {})".format(frame_id, container.vendor_id))
        else:
            self.calib_config[container.vendor_id] = dict()

    
    def imu_calibration(self, msg_data_elem):
        data = msg_data_elem
        if not data.frame_id in self.container.keys():
            self.add_new_container(data.frame_id)

        container = self.container[data.frame_id]
        container.set_vendor_id(data.vendor_id)
        acc_raw, omg_raw, domg_raw = self.get_3darrays_from_msgdata(data)

        ## first step: detect gyro bias and calc covariances
        if container.covariances["acc"] is None:
            ## static: here we estimate covariance and bias of gyroscopes
            ## for covariance estimation
            self.update_list(container.acc_list_for_cov, acc_raw)
            # self.update_list(container.dacc_list_for_cov, dacc_raw)
            self.update_list(container.omg_list_for_cov, omg_raw)
            self.update_list(container.domg_list_for_cov, domg_raw)

            counter = [e is None for e in container.omg_list_for_cov].count(False)
            rospy.logwarn("Do not move your robot: {} / {}".format(counter, container.omg_size))

            if not container.omg_list_for_cov[0] is None:
                omega_dataset = np.vstack(container.omg_list_for_cov).T
                b_omega = np.mean(omega_dataset, axis=1)
                container.calib_param["gyro_bias"] = b_omega

                ## covariances
                container.covariances["acc"]   = np.cov(np.vstack(container.acc_list_for_cov).T)
                # container.covariances["dacc"]  = np.cov(np.vstack(container.dacc_list_for_cov).T)
                container.covariances["gyro"]  = np.cov(np.vstack(container.omg_list_for_cov).T)
                container.covariances["dgyro"] = np.cov(np.vstack(container.domg_list_for_cov).T)
            
            return 1
        
        ## second step: acc calib
        self.load_precalibrated_param(data.frame_id, container)
        if container.calib_param["acc_bias"] is None:
            acc_list = container.acc_list
            self.update_list(container.acc_list, acc_raw)

            counter = [e is None for e in acc_list].count(False)
            rospy.logwarn("Rotate your robot slowly: {} / {}".format(counter, container.acc_size))

            if not acc_list[0] is None:
                acc_dataset = np.vstack(acc_list).T
                R_a, b_a = ImuCalibrator.calc_IMU_calibration_param(acc_dataset)
                container.calib_param["acc_scaler"] = R_a
                container.calib_param["acc_bias"] = b_a
            
            return 2
        
        return 0


    def get_container(self, frame_id):
        return self.container[frame_id]
    

    def set_calibparam_to_yamldict(self, container):
        self.calib_config[container.vendor_id]['acc_scaler'] = container.calib_param['acc_scaler'].flatten().tolist()
        self.calib_config[container.vendor_id]['acc_bias'] = container.calib_param['acc_bias'].flatten().tolist()
        self.calib_config[container.vendor_id]['gyro_bias'] = container.calib_param['gyro_bias'].flatten().tolist()


    def callback(self, msg):
        # helper

        # frame_id = msg.header.frame_id

        if self.calibration_step:
            calib_flags = 0
            for data in msg.data:
                calib_flags += self.imu_calibration(data)
            
            if calib_flags == 0:
                for frame_id, container in self.container.items():
                    self.set_calibparam_to_yamldict(container)

                ## set calibration config into yaml file
                with open(self.calib_param_yaml, "w") as yml:
                    yaml.dump(self.calib_config, yml)
                self.calibration_step = False
        else:
            data_list = []
            for data in msg.data:
                calib_data = ImuDataFiltered()

                container = self.get_container(data.frame_id)
                acc_raw, omg_raw, domg_raw = self.get_3darrays_from_msgdata(data)

                calib_param = container.calib_param
                calib_data.frame_id = data.frame_id

                acc_calib  = ImuCalibrator.calc_calibrated_acc(acc_raw, calib_param)
                # dacc_calib = ImuCalibrator.calc_calibrated_dacc(dacc_raw, calib_param)
                omg_calib  = omg_raw - calib_param["gyro_bias"]
                domg_calib = domg_raw

                covs = container.covariances

                self.convert_3Darray_to_rosmsg(acc_calib,  calib_data.acc)
                # self.convert_3Darray_to_rosmsg(dacc_calib, calib_data.dacc_dt)
                self.convert_3Darray_to_rosmsg(omg_calib,  calib_data.gyro)
                self.convert_3Darray_to_rosmsg(domg_calib, calib_data.dgyro_dt)

                calib_data.acc_covariance = ImuCalibrator.calc_calibrated_cov_acc(covs["acc"], calib_param).flatten()
                # calib_data.dacc_covariance = ImuCalibrator.calc_calibrated_cov_dacc(covs["dacc"], calib_param).flatten()
                calib_data.gyro_covariance = covs["gyro"].flatten()
                calib_data.dgyro_covariance = covs["dgyro"].flatten()
                
                # rospy.logwarn("acc_calib = {}".format(acc_calib))
                # rospy.logwarn("omg_calib = {}".format(omg_calib))
                # rospy.logwarn("{}: norm(acc_calib) = {}".format(data.frame_id, np.linalg.norm(acc_calib)))
                rospy.logwarn_once("publishing...")

                data_list.append(calib_data)

            ## publish
            calib_msg = ImuDataFilteredList()
            calib_msg.header = msg.header
            calib_msg.data = data_list
            
            self.calib_imu_pub.publish(calib_msg)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--recalib', action="store_true")
    args = parser.parse_args()

    rospy.init_node('imu_calibrator', anonymous=False)

    calib = ImuCalibratorROS(acc_size=10000, omg_size=1000,
                             calib_param_yaml="{}/config/imu_calibparams.yaml".format(roslib.packages.get_pkg_dir("urdf_estimation_with_imus")),
                             force_recalib=args.recalib)
    imu_sub = rospy.Subscriber("/imu_filtered/raw", ImuDataFilteredList, calib.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()