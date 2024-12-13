from filterpy.kalman import KalmanFilter
import numpy as np

class KeypointSmoother:
    def __init__(self, n_keypoints):
        self.n_keypoints = n_keypoints
        self.filters = [self._create_kalman_filter() for _ in range(n_keypoints * 3)]
        
    def _create_kalman_filter(self):
        kf = KalmanFilter(dim_x=2, dim_z=1)  # State: position, velocity
        kf.x = np.zeros(2)
        kf.F = np.array([[1., 0.5], [0., 1.]])  # State transition matrix
        kf.H = np.array([[1., 0.]])  # Measurement matrix
        kf.P *= 500.  # Covariance matrix
        kf.R = 2  # Measurement noise
        kf.Q = 0.05  # Process noise
        return kf
    
    def update(self, keypoints):
        smoothed = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            for j in range(3):  # x, y, z coordinates
                idx = i * 3 + j
                self.filters[idx].predict()
                self.filters[idx].update(keypoints[i, j])
                smoothed[i, j] = self.filters[idx].x[0]
        return smoothed