from collections import deque
import numpy as np

class Smoother():
    def __init__(self, n):
        self.n = n
        # was the line detected in the last iterations?
        self.detected = deque([], maxlen=n)
        # x values of the last n fits of the line
        self.recent_x_values = deque([], maxlen=n)
        #polynomial coefficients
        self.recent_poly_coef = deque([], maxlen=n)
        #radius of curvature of the line in some units
        self.recent_radius_curvature = deque([], maxlen=n)
        #distance in meters of vehicle center from the line
        self.recent_offset = deque([], maxlen=n)
        #recent hulls
        self.recent_hulls = deque([], maxlen=n)
        #recent centroids
        self.recent_centroids = deque([], maxlen=n)
        #recent inverse matrices
        self.recent_invert_matrices = deque([], maxlen=n)

    def add_detected(self, detected):
        self.detected.append(detected)

    def add_recent_x_values(self, left_lane, right_lane):
        """
        find the centroids in the real picture view
        :param left_lane:
        :param right_lane:
        :return:
        """
        reshape = np.concatenate((left_lane, right_lane),axis=1)  # 9x4
        self.recent_x_values.append(reshape)

    def add_recent_poly_coef(self, left_coefs, right_coefs):
        """
        :param coefs: in standard format where index 0 is highest order coef
        """
        self.recent_poly_coef.append((left_coefs, right_coefs))

    def add_recent_radius_curvature(self, r_left, r_right):
        self.recent_radius_curvature.append((r_left, r_right))

    def get_recent_radius_curvature(self):
        for i in range(-1, -(len(self.detected)+1), -1):
            if self.detected[i]:
                return self.recent_radius_curvature[i]

    def add_recent_offset(self, offset):
        self.recent_offset.append(offset)

    def get_recent_offset(self):
        for i in range(-1, -(len(self.detected)+1), -1):
            return self.recent_offset[i]

    def recent_offset_change(self, offset, threshold):
        if np.all(np.logical_not(np.array(self.detected))):
            return False
        elif self.get_recent_offset()==0:
            return False
        elif len(self.detected)>0:
            return abs((self.get_recent_offset() - offset)/self.get_recent_offset())>threshold
        else:
            return False

    def add_recent_hulls(self, hull):
        self.recent_hulls.append(hull)

    def get_last_hull(self):
        return self.recent_hulls[-1]

    def high_proportion_unchanged(self):
        if sum(self.detected) > float(self.n) / 2:
            return np.zeros((1,5))

    def poly_change(self, latest_value):
        self.high_proportion_unchanged()  #if high number of last frames failed test, time for forced update
        if np.all(np.logical_not(np.array(self.detected))): #they are all false
            return np.zeros((1, 6))
        elif len(self.detected)>0:
            mean_coefs = np.mean(np.concatenate(self.recent_poly_coef).reshape((-1,6))[np.array(self.detected)], axis=0) #retreive successful entries
            return np.abs((mean_coefs - latest_value.reshape((-1,6))) / mean_coefs)
        else:
            return np.zeros((1,6))

    def get_recent_x_values(self):
        #this is only called if there is at least one True in detected
        for i in range(-1, -(len(self.detected)+1), -1):
            if self.detected[i]:
                return self.recent_x_values[i]

    def add_recent_invert_matrix(self, inverse_matrix):
        self.recent_invert_matrices.append(inverse_matrix)

    def get_recent_invert_matrix(self):
        for i in range(-1,-(len(self.detected)+1),-1):
            if self.detected[i]:
                return self.recent_invert_matrices[i]

    def add_recent_centroids(self, centroids):
        self.recent_centroids.append(centroids)

    def get_recent_centroids(self):
        for i in range(-1,-(len(self.detected)+1),-1):
            if self.detected[i]:
                return self.recent_centroids[i]
