import numpy as np

class NormalDistributionData:
    def __init__(self, position, covariance, size=3, name=""):
        if position is None:
            ## non-informative values
            position = np.zeros(size)
            ## non-informative values
        if covariance is None:
            covariance = np.eye(size) * 1e+30
        
        self.position = position
        self.covariance = covariance
        self.name = name

    def __getitem__(self, idces):
        ## marginal distribution (based on the property of the normal distributions)
        return NormalDistributionData(position=self.position[idces],
                                      covariance=self.covariance[idces, idces])
    
    @classmethod
    def empty(cls, size=3, name=""):
        return cls(None, None, size=size, name=name)
    
    
    def __mul__(self, other):
        """
        Multiply two Gaussians
        """
        sum_of_cov_inv = np.linalg.inv(self.covariance + other.covariance)
        position = other.covariance @ sum_of_cov_inv @ self.position \
                 + self.covariance @ sum_of_cov_inv @ other.position
        covariance = self.covariance @ sum_of_cov_inv @ other.covariance
        
        return NormalDistributionData(position, covariance)
    
    def __repr__(self) -> str:
        return "NormalDistributionData(name={}, position={}, covariance=\n{})".format(self.name, self.position, self.covariance)


class BinghamParameterGeneral:
    N_nc = 50
    def __init__(self):
        pass

    def __repr__(self):
        _str = "{}({})".format(self.__class__, vars(self))
        return _str
    

    @classmethod
    def empty(cls):
        raise NotImplementedError
    

    @classmethod
    def initialize(cls):
        raise NotImplementedError
    


class StateDataGeneral:
    def __init__(self):
        pass

    def __repr__(self):
        _str = "{}({})".format(self.__class__, vars(self))
        return _str
    

    @classmethod
    def empty(cls):
        raise NotImplementedError
    

    @classmethod
    def initialize(cls):
        raise NotImplementedError
    


class BinghamParameterKalmanFilter(BinghamParameterGeneral):
    init_varlen = 8
    def __init__(self,
                M,
                Z,
                E_q4th,
                ddnc_10D,
                omegas,
                Amat,
                CovQ,
                mode):

        self.M = M
        self.Z = Z
        self.E_q4th = E_q4th
        self.ddnc_10D = ddnc_10D
        self.omegas = omegas
        self.Amat = Amat
        self.CovQ = CovQ
        self.mode = mode

    @classmethod
    def empty(cls):
        return cls(*([None]*cls.init_varlen))
    

    @classmethod
    def initialize(cls):
        new_cls = cls.empty()
        new_cls.M = np.eye(4)
        new_cls.Z = np.array([1e-8, 0, 0, 0])
        new_cls.E_q4th = np.array([3, 0, 1, 0, 2, 
                                   0, 0, 0, 0, 1, 
                                   0, 0, 0, 0, 2, 
                                   0, 0, 0, 0, 0, 
                                   0, 0, 0, 0, 0, 
                                   1, 0, 0, 0, 0, 
                                   0, 0, 0, 0, 2]) / 24.
        new_cls.ddnc_10D = np.array([2.46733938, 0.82244646, 0.82244646, 0.82244646, 
                                                 2.46733936, 0.82244645, 0.82244645, 
                                                             2.46733936, 0.82244645, 
                                                                         2.46733936])
        new_cls.omegas = np.array([0.4, 0.2, 0.2, 0.2])
        new_cls.Amat = np.diag(new_cls.Z)
        new_cls.CovQ = np.diag(new_cls.omegas)
        new_cls.mode = np.array([1, 0, 0, 0])

        return new_cls



class BinghamParameterLeastSquare(BinghamParameterGeneral):
    init_varlen = 3
    def __init__(self,
                Amat,
                CovQ_inv,
                mode):
        
        self.Amat = Amat
        self.CovQ_inv = CovQ_inv
        self.mode = mode


    @classmethod
    def empty(cls):
        return cls(*([None]*cls.init_varlen))
    

    @classmethod
    def initialize(cls):
        new_cls = cls.empty()
        new_cls.Amat = np.zeros((4,4))
        new_cls.CovQ_inv = np.diag([4, 4, 4, 4])
        new_cls.mode = np.array([1, 0, 0, 0])
        return new_cls

    
    def update(self, Amat, CovQ_inv=None):
        new_cls = self.empty()
        Z,M = np.linalg.eigh(Amat)
        new_cls.mode = M[:, np.argmax(Z)]
        new_cls.Amat = Amat
        if CovQ_inv is None:
            new_cls.CovQ_inv = self.CovQ_inv
        else:
            new_cls.CovQ_inv = CovQ_inv

        return new_cls
    

class StateDataKalmanFilter(StateDataGeneral):
    init_varlen = 5
    def __init__(self, trans, trans_cov,
                bingham_param: BinghamParameterKalmanFilter,
                jointposition_wrt_thisframe,
                jointposition_wrt_childframe
                ):
        self.trans = trans
        self.trans_cov = trans_cov

        if bingham_param is None:
            bingham_param = BinghamParameterKalmanFilter.empty()
        self.bingham_param = bingham_param

        self.jointposition_wrt_thisframe = jointposition_wrt_thisframe
        self.jointposition_wrt_childframe = jointposition_wrt_childframe


    @classmethod
    def empty(cls):
        return cls(*([None]*cls.init_varlen))
    

    def update(self, trans, trans_cov,
               bingham_param,
               jointposition_wrt_thisframe=None,
               jointposition_wrt_childframe=None):
        
        new_cls = self.empty()
        new_cls.trans = trans
        new_cls.trans_cov = trans_cov
        new_cls.bingham_param = bingham_param

        if jointposition_wrt_thisframe is None:
            new_cls.jointposition_wrt_thisframe = self.jointposition_wrt_thisframe
        if jointposition_wrt_childframe is None:
            new_cls.jointposition_wrt_childframe = self.jointposition_wrt_childframe

        return new_cls
    

    def update_only_rotation(self, bingham_param: BinghamParameterKalmanFilter):
        new_cls = self.empty()
        new_cls.bingham_param = bingham_param

        ## use previous parameters
        new_cls.trans = self.trans
        new_cls.trans_cov = self.trans_cov
        new_cls.jointposition_wrt_thisframe = self.jointposition_wrt_thisframe
        new_cls.jointposition_wrt_childframe = self.jointposition_wrt_childframe

        return new_cls
    
    def update_only_translation(self, trans, trans_cov):
        new_cls = self.empty()
        new_cls.trans = trans
        new_cls.trans_cov = trans_cov

        ## use previous parameters
        new_cls.bingham_param = self.bingham_param
        new_cls.jointposition_wrt_thisframe = self.jointposition_wrt_thisframe
        new_cls.jointposition_wrt_childframe = self.jointposition_wrt_childframe

        return new_cls
    

    @classmethod
    def initialize(cls,
                   jointposition_wrt_thisframe,
                   jointposition_wrt_childframe):
        return cls(np.zeros(6), np.eye(6), BinghamParameterKalmanFilter.initialize(),
                   jointposition_wrt_thisframe,
                   jointposition_wrt_childframe)
    

class StateDataLeastSquare(StateDataGeneral):
    init_varlen = 3
    def __init__(self, position, position_cov,
                bingham_param: BinghamParameterLeastSquare,
                ):
        self.position = position
        self.position_cov = position_cov

        if bingham_param is None:
            bingham_param = BinghamParameterLeastSquare.empty()
        self.bingham_param = bingham_param


    def update(self, 
               position: np.ndarray,
               position_cov: np.ndarray,
               Amat: np.ndarray,
               CovQ_inv=None):
        new_cls = self.empty()
        new_cls.position = position
        new_cls.position_cov = position_cov

        ## set previous values first then update
        new_cls.bingham_param = self.bingham_param.update(Amat, CovQ_inv)

        return new_cls


    @classmethod
    def empty(cls):
        return cls(*([None]*cls.init_varlen))
    

    @classmethod
    def initialize(cls):
        return cls(np.zeros(3), np.eye(3), BinghamParameterLeastSquare.initialize())


class ObservationData:
    def __init__(self,
                 dt,
                 force, dforce, 
                 gyro, dgyro, ddgyro,
                 force_cov, dforce_cov,
                 gyro_cov, dgyro_cov, ddgyro_cov):

        self.E_force = force
        self.E_dforce = dforce
        self.E_gyro = gyro
        self.E_dgyro = dgyro
        self.E_ddgyro = ddgyro

        self.Cov_force = force_cov
        self.Cov_dforce = dforce_cov
        self.Cov_gyro = gyro_cov
        self.Cov_dgyro = dgyro_cov
        self.Cov_ddgyro = ddgyro_cov

        self.dt = dt

        self.prev_t = None


    @classmethod
    def empty(cls):
        return cls(*([None]*11))
    

    def update(self,
               dt,
                force, dforce, 
                gyro, dgyro, ddgyro,
                force_cov, dforce_cov,
                gyro_cov, dgyro_cov, ddgyro_cov):

        self.E_force = force
        self.E_dforce = dforce
        self.E_gyro = gyro
        self.E_dgyro = dgyro
        self.E_ddgyro = ddgyro

        self.Cov_force = force_cov
        self.Cov_dforce = dforce_cov
        self.Cov_gyro = gyro_cov
        self.Cov_dgyro = dgyro_cov
        self.Cov_ddgyro = ddgyro_cov

        self.dt = dt


    @classmethod
    def realized_data(cls,
                    dt,
                    force, dforce, 
                    gyro, dgyro):
        zero_cov = np.zeros((3,3))
        return cls(dt, force, dforce, gyro, dgyro,
                   zero_cov, zero_cov, zero_cov, zero_cov)
    
    
    def make_realized_data(self):
        return self.realized_data(self.dt,
                           self.E_force, self.E_dforce,
                           self.E_gyro, self.E_dgyro)


class JointModule:
    def __init__(self,
                 parent_imu: ObservationData,
                 child_imu: ObservationData):
        self.parent = parent_imu
        self.child  = child_imu