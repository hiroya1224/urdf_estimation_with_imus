string frame_id
string child_frame_id

string vendor_id

geometry_msgs/Vector3 gyro
float64[9] gyro_covariance # Row major about x, y, z axes

geometry_msgs/Vector3 dgyro_dt
float64[9] dgyro_covariance # Row major about x, y, z axes

geometry_msgs/Vector3 ddgyro_ddt
float64[9] ddgyro_covariance # Row major about x, y, z axes

geometry_msgs/Vector3 acc
float64[9] acc_covariance # Row major x, y, z

geometry_msgs/Vector3 dacc_dt
float64[9] dacc_covariance # Row major x, y, z

geometry_msgs/Vector3 mag
float64[9] mag_covariance # Row major x, y, z