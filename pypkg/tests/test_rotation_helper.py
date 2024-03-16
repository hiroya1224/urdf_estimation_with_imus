from imu_relpose_estim.utils.rotation_helper import RotationHelper
import numpy as np

def test_rotation_converters():
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)

    # quat --> rotmat --> quat
    def q2R2q(q):
        R = RotationHelper.quat_to_rotmat(*q)
        req = RotationHelper.rotmat_to_quat(R)

        assert np.all(np.isclose(q, req)) or np.all(np.isclose(q, -req))
    
    # quat --> rpy --> quat
    def q2r2q(q):
        rpy = RotationHelper.quat_to_rpy(*q)
        req = RotationHelper.rpy_to_quat(rpy)

        assert np.all(np.isclose(q, req)) or np.all(np.isclose(q, -req))

    # quat --> rpy --> rotmat --> quat
    def q2r2R2q(q):
        rpy = RotationHelper.quat_to_rpy(*q)
        R = RotationHelper.rpy_to_rotmat(rpy)
        req = RotationHelper.rotmat_to_quat(R)

        assert np.all(np.isclose(q, req)) or np.all(np.isclose(q, -req))

    q2R2q(q)
    q2r2q(q)
    q2r2R2q(q)

    return True

print(test_rotation_converters())
