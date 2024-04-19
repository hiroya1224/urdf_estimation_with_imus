# urdf_estimation_with_imus
Estimation module for URDF with multiple IMUs

## Installation
For detailed information on dependent packages, it may be helpful to refer to the `docker/Dockerfile`s.

### python package
```
cd urdf_estimation_with_imus
pip3 install -r pypkg/requirements.txt
pip3 install -e pypkg/
```

### ros package
After installing our python package, then
```
cd ~/catkin_ws/src
catkin build urdf_estimation_with_imus
```


## Estimation Demo with rosbag

- On the terminal window 1,
```
roscore
```

- On the terminal window 2,
```
roslaunch urdf_estimation_with_imus twoimus.launch # you can shutdown this immediately
roslaunch urdf_estimation_with_imus imu_preproc.launch
```
**NOTE**: `twoimus.launch` supplies rosparam `/symbolic_robot_description`.


- On the terminal window 3, 
```
rosbag play -r 3.0 ./bags/sample/twoimus_200mm_for_calibration.bag
```
**NOTE**: The calibration process takes a few minutes. You can use -r option of rosbag for quick calibration.


- On the terminal window 4, 
```
rosrun urdf_estimation_with_imus extleastsq_estim_imu_relpose.py --this imu0 --child imu1
```
**NOTE**: This is the estimator of relative pose of imu1 with respect to imu0.


- On the terminal window 5,

```
rosrun urdf_estimation_with_imus relpose_visualizer.py --this imu0 --child imu1
```
**NOTE**: This is the visualizer of estimation results.


- On the terminal window 6,
```
rosbag play ./bags/sample/twoimus_200mm_sample.bag
```
**NOTE**: The groundtruth of relative pose is as follows:

- true position: `[-0.200, 0., 0.]`
- true rotation(wxyz): `[1., 0., 0., 0.]` or `[-1., 0., 0., 0.]`

When you perform these steps, you will see the following results in terminal 4:

```
position      : [-0.19706455  0.00316656 -0.00375119]
rotation(wxyz): [ 9.99877238e-01 -9.64530793e-04 -7.83922661e-03 -1.35323849e-02]
```

## Docker

### `debug` image
- mount the host package and build it in the container
- this is useful for debugging on the host

```
cd docker
docker-compose up --build --force-recreate debug
docker exec -it urdfestim_debug bash
```

### `test` image
- clone the package directly from github and build it in the container
- this is useful for validation of our paper's method.

```
cd docker
docker-compose up --build --force-recreate test
docker exec -it urdfestim_test bash
```
