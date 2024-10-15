#!/usr/bin/env python3
import numpy as np
from scipy.spatial import ConvexHull

class CalibrationContainer:
    def __init__(self, acc_size, omg_size):
        self.acc_size = acc_size
        self.omg_size = omg_size
        cov_size = omg_size

        self.calib_param = {'acc_scaler': None,
                            'acc_bias': None,
                            'gyro_bias': None}
        self.covariances = {'acc': None,
                            'dacc': None,
                            'gyro': None,
                            'dgyro': None,
                            'ddgyro': None,}
        
        self.acc_list_for_cov   = [None for _ in range(cov_size)]
        self.dacc_list_for_cov  = [None for _ in range(cov_size)]
        self.omg_list_for_cov   = [None for _ in range(cov_size)]
        self.domg_list_for_cov  = [None for _ in range(cov_size)]
        self.ddomg_list_for_cov = [None for _ in range(cov_size)]

        self.acc_list  = [None for _ in range(acc_size)]

        self.vendor_id = ""
    
    def set_vendor_id(self, vendor_id):
        self.vendor_id = vendor_id


class ImuCalibrator:
    """
    Based on "Calibration of MEMS Triaxial Accelerometers Based on the Maximum Likelihood Estimation Method"
    Ellipsoid fittting is based on "Least squares ellipsoid specific fitting"
    """
    def __init__(self):
        pass

    @staticmethod
    def resample(points_3d, num_samples_per_face=100):
        assert points_3d.shape[1] == 3

        # Use the class to uniformly sample points
        resampler = SphereUniformResampler(points_3d, num_samples_per_face)
        resampled_points = resampler.resample()
        
        return resampled_points
        
        
    @staticmethod
    def calc_ellipsoid_param(Dmat, k=4):
        C1 = np.array([
            [-1, k/2 - 1, k/2 - 1, 0, 0, 0],
            [k/2 - 1, -1, k/2 - 1, 0, 0, 0],
            [k/2 - 1, k/2 - 1, -1, 0, 0, 0],
            [0, 0, 0, -k, 0, 0],
            [0, 0, 0, 0, -k, 0],
            [0, 0, 0, 0, 0, -k],
        ])
        # C = np.zeros((10,10))
        # C[:6,:6] = C1

        DDT = Dmat @ Dmat.T
        S11 = DDT[:6, :6]
        S12 = DDT[:6, 6:]
        # S21 = DDT[6:, :6]
        S22 = DDT[6:, 6:]

        M = np.linalg.inv(C1) @ (S11 - S12 @ np.linalg.inv(S22) @ S12.T)

        eigM = np.linalg.eig(M)
        u1 = eigM[1][:, np.argmax(eigM[0])]
        u2 = -np.dot(np.linalg.inv(S22) @ S12.T, u1)
        u = np.hstack([u1,u2])
        
        return u
    
    @staticmethod
    def acc_to_Dmat(acc_dataset):
        assert acc_dataset.shape[0] == 3
        x = acc_dataset[0,:]
        y = acc_dataset[1,:]
        z = acc_dataset[2,:]
        D = np.vstack([
            x**2, y**2, z**2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, np.ones_like(x)
        ])
        return D
    
    @classmethod
    def calc_IMU_calibration_param(cls, acc_dataset):
        Dmat = cls.acc_to_Dmat(acc_dataset)
        a,b,c,f,g,h,p,q,r,d = cls.calc_ellipsoid_param(Dmat)

        E = np.array([
                [a,f,g],
                [f,b,h],
                [g,h,c]]
            )
        F = np.array([p,q,r])

        scale = 1 / (np.dot(F, np.dot(np.linalg.inv(E), F)) - d)

        E = scale * E
        F = scale * F

        ## scale and skew parameter
        R_a = np.linalg.cholesky(E)
        ## bias
        b_a = -np.dot(np.linalg.inv(E), F)

        return R_a, b_a
    

    @staticmethod
    def calc_calibrated_acc(acc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        b_a = calib_param['acc_bias']
        return np.dot(R_a, acc_raw - b_a) * gravity_magnitude
    

    @staticmethod
    def calc_calibrated_dacc(dacc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        return np.dot(R_a, dacc_raw) * gravity_magnitude
    

    @staticmethod
    def calc_calibrated_cov_acc(cov_acc_raw, calib_param, gravity_magnitude=9.80665):
        R_a = calib_param['acc_scaler']
        return np.dot(np.dot(R_a, cov_acc_raw), R_a.T) * gravity_magnitude**2
    

    @classmethod
    def calc_calibrated_cov_dacc(cls, cov_dacc_raw, calib_param, gravity_magnitude=9.80665):
        return cls.calc_calibrated_cov_acc(cov_dacc_raw, calib_param, gravity_magnitude=gravity_magnitude)


class SphereUniformResampler:
    
    def __init__(self, points, num_samples_per_face):
        """
        Initialization method
        
        Parameters:
        - points: 3D sample points (numpy array of shape (N, 3))
        - num_samples_per_face: number of points to sample from each triangular region
        """
        self.original_points = points  # Original 3D points
        self.projected_points = self.project_to_unit_sphere(points)  # Points projected onto the unit sphere
        self.num_samples_per_face = num_samples_per_face
        self.vertices = self.icosahedron_vertices()
        self.faces = self.icosahedron_faces(self.vertices)
        
    @staticmethod
    def icosahedron_vertices():
        """
        Function to generate the vertices of an icosahedron and normalize them to the unit sphere
        
        Returns:
        - vertices: vertices of the icosahedron on the sphere (numpy array of shape (12, 3))
        """
        t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        vertices = np.array([
            [-1,  t,  0],
            [ 1,  t,  0],
            [-1, -t,  0],
            [ 1, -t,  0],
            [ 0, -1,  t],
            [ 0,  1,  t],
            [ 0, -1, -t],
            [ 0,  1, -t],
            [ t,  0, -1],
            [ t,  0,  1],
            [-t,  0, -1],
            [-t,  0,  1]
        ])
        
        # Normalize vertices to the unit sphere
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices

    @staticmethod
    def icosahedron_faces(vertices):
        """
        Function to generate the faces (triangles) of the icosahedron
        
        Parameters:
        - vertices: vertices of the icosahedron on the sphere (numpy array of shape (12, 3))
        
        Returns:
        - faces: triangular faces of the icosahedron (numpy array of shape (20, 3))
        """
        hull = ConvexHull(vertices)
        return hull.simplices
    
    @staticmethod
    def project_to_unit_sphere(points):
        """
        Function to project points onto the unit sphere
        
        Parameters:
        - points: 3D sample points (numpy array of shape (N, 3))
        
        Returns:
        - unit_sphere_points: points projected onto the unit sphere (numpy array of shape (N, 3))
        """
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norm

    @staticmethod
    def point_in_triangle(tri, p):
        """
        Function to determine whether a point is inside a triangle
        
        Parameters:
        - tri: vertices of the triangle (numpy array of shape (3, 3))
        - p: point to check (numpy array of shape (3,))
        
        Returns:
        - bool: True if the point is inside the triangle, False otherwise
        """
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        v2 = p - tri[0]
        
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    def classify_points(self):
        """
        Function to classify the original points into triangular regions
        
        Returns:
        - classified_points: original points classified by each triangular region (list of lists)
        """
        classified_points = [[] for _ in range(len(self.faces))]
        
        for orig_p, proj_p in zip(self.original_points, self.projected_points):
            for i, face in enumerate(self.faces):
                triangle = self.vertices[face]
                if self.point_in_triangle(triangle, proj_p):
                    classified_points[i].append(orig_p)  # Classify the original point
                    break
        
        return classified_points
    
    def sample_points_from_faces(self, classified_points):
        """
        Function to uniformly sample points from each triangular region
        
        Parameters:
        - classified_points: original points classified by each triangular region (list of lists)
        
        Returns:
        - resampled_points: resampled original points from each region (numpy array of shape (M, 3))
        """
        resampled_points = []
        for points_in_face in classified_points:
            points_in_face = np.array(points_in_face)  # Convert to array
            if len(points_in_face) >= self.num_samples_per_face:
                indices = np.random.choice(len(points_in_face), size=self.num_samples_per_face, replace=False)
                sampled_points = points_in_face[indices]
            else:
                sampled_points = points_in_face
            resampled_points.extend(sampled_points)
        return np.array(resampled_points)
    
    def resample(self):
        """
        Function to execute the entire process and return the resampled original points
        
        Returns:
        - resampled_points: uniformly resampled original points (numpy array of shape (M, 3))
        """
        classified_points = self.classify_points()
        resampled_points = self.sample_points_from_faces(classified_points)
        return resampled_points
