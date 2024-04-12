# urdf_estimation_with_imus
Estimation module for URDF with multiple IMUs

## demo

```
## in the terminal window (1)
roslaunch urdf_estimation_with_imus imu_preproc.launch

## in the terminal window (2)
#### wait for a minute. you can use -r option of rosbag for quick calibration
rosbag play ./bags/sample/twoimus_200mm_for_calibration.bag

## estimation
## in the terminal window (3)
rosrun urdf_estimation_with_imus extleastsq_estim_imu_relpose.py --this imu0 --child imu1
## in the terminal window (2)
rosbag play ./bags/sample/twoimus_200mm_sample.bag

## you can check the estimation result in the terminal window (3)
## true position      : [-200., 0., 0.]
## true rotation(wxyz): [1., 0., 0., 0.] or [-1., 0., 0., 0.]

```