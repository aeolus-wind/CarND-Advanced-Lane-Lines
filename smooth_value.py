from collections import deque
import numpy as np

def exponential_smoothing_update(alpha):
    def exponential_smooth_f(previous_value, latest_value):
        latest_update = (1-alpha)*previous_value.pop() + alpha*latest_value
        previous_value.append(latest_update)
    return exponential_smooth_f


def exponential_smoothing_value(previous_value):
    if len(previous_value)>0:
        return previous_value[-1]
    else:
        return None


def running_average_update(previous_values, latest_value):
    previous_values.popleft()
    previous_values.append(latest_value)


def running_average_value(previous_values):
    return np.mean(previous_values)

def transform_lane_lazy(matrix, window_height=80):
    """
    equivalent to functions in find_lanelines except it return a function parametrized by matrix
    takes centroids and transforms them into points in the original picture
    see get_perspective_transform under
    http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    :param previous_centroids: takes the previous centroids
    :return:
    """
    def transform_lane(ploty, previous_centroids)
        centroids = np.array(previous_centroids)
        left = np.column_stack((centroids[:, 0], range(720, 0, -window_height), np.ones(int(720/window_height))))

        right = np.column_stack((centroids[:, 1], range(720, 0, -window_height), np.ones(int(720/window_height))))
        left_transformed = matrix.dot(left.T)
        right_transformed = matrix.dot(right.T)
        left_transformed_norm =  left_transformed[:2,:]/left_transformed[2,:]
        right_transformed_norm = right_transformed[:2,:]/right_transformed[2,:]

        left_fit = np.polyfit(left_transformed_norm[1], left_transformed_norm[0], 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        right_fit = np.polyfit(right_transformed_norm[1], right_transformed_norm[0], 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        return left_fitx, right_fitx
    return transform_lane


def extrapolate_to_y_coordinate(ploty, left_transformed, right_transformed):
    """
    fits a polynomial to the 'real coordinates'  and finds the x
    where the polynomail occurs
    :param ploty:
    :param left_transformed:
    :param right_transformed:
    :return:
    """

    return left_fitx, right_fitx

class Smoother:
    def __init__(self, maxlen, update_function, smooth_function):
        self.previous_values = deque([], maxlen=maxlen)
        self.update_function = update_function
        self.smooth_function = smooth_function

    def update(self, latest_value):
        self.previous_values.append(self.update_function(self.previous_values, latest_value))

    def get_smoothed_value(self):
        return self.smooth_function(self.previous_values)


class ExponentialSmoother(Smoother):
    def __init__(self, alpha):
        self.previous_values = deque([], maxlen=1)
        self.update_function = exponential_smoothing_update(alpha)
        self.smooth_function = exponential_smoothing_value


class RunningAverage(Smoother):
    def __init__(self, n):
        self.previous_value = deque([], maxlen=n)
        self.update_function = running_average_update
        self.smooth_function = running_average_value

class ContinuousStartFrame(Smoother):
    """
    keeps track of the current function to take the ending point of the previous frame and
    transform it into the current frame
    """
    def __init(self,self):
        self.previous_value = deque([], maxlen=2)