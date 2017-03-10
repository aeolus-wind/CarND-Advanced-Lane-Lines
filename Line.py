from collections import deque
import numpy as np
from find_lanelines import window_height, margin, window_width

class Extrapolate_Line():

    #to be used if line is not detected, can rescue the output if the changes are not too extreme

    def __init__(self, maxlen):
        if maxlen < 2:
            raise ValueError("maxlen must be at least 2")
        self.recent_xfitted = deque([], maxlen=maxlen)
        self.velocity_xchange = deque([], maxlen=2)
        self.accel_xchange = deque([], maxlen=2)

    def update_xfitted(self, centroids):
        if not isinstance(centroids, np.ndarray):
            centroids = np.array(centroids)
        self.recent_xfitted.append(centroids)
        if len(self.recent_xfitted) >= 1:
            self.velocity_xchange.append(centroids - self.recent_xfitted[-1])  # change between recent and current
        if len(self.recent_xfitted) >= 2:  # need to have seen at least 3 values to get accel
            self.accel_xchange.append((self.velocity_xchange[-1] - self.velocity_xchange[-2]))

    def predict_xfitted(self):

        if len(self.accel_xchange) > 0:
            # in one time step, how much have the first, second order
            # changes affected the values from the last time step
            return self.recent_xfitted[-1] + (self.velocity_xchange[-1] + self.accel_xchange[-1])
        else:
            raise ValueError("accel is undefined")

    def deviation_predicted(self, centroids):
        return np.abs((centroids - self.recent_xfitted[-1]) / self.recent_xfitted[-1])


class Line():
    img_height = 720
    reinitialize = 5
    c = 0

    def __init__(self, maxlen, tolerance):
        self.extrap = Extrapolate_Line(maxlen)
        self.tolerance = tolerance
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque([], maxlen=maxlen)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = deque([], maxlen=maxlen)
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.curverad = None
        #distance in meters of vehicle center from the line
        self.offset = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = np.linspace(0, self.img_height-1, self.img_height)
        #detected centroid
        self.centroid = None
    def increment_counter(self):
        self.c += 1

    def set_centroid(self,centroids):
        self.centroid = centroids

    def new_fit_valid(self, new_fit, other_new_fit, offset):
        new_xs = self.calculate_xs(new_fit)
        other_new_xs = self.calculate_xs(other_new_fit)
        polynomial_similar = np.abs((new_fit - self.get_best_fit())/self.get_best_fit()) < 0.5  # all polynomial coefficients changed less than 20%
        if np.any(polynomial_similar == np.inf):
            polynomial_similar = True
        else:
            polynomial_similar = np.all(polynomial_similar)
        margin = window_width/2

        #parallel_test = np.std(other_new_xs - new_xs) < 5*margin #how to account for curvature?
        parallel_test = True  # need to come up with new test
        if self.offset == 0 or None:
            offset_similar = True
        else:
            offset_similar = abs((self.offset - offset)) < 1.5 # car doesn't move that much from the center
        #print('tests are', polynomial_similar, parallel_test,  offset_similar)
        return (polynomial_similar and parallel_test and offset_similar) or (self.c >= 8)

    def next_fit(self, centroids_current, centroids_other, offset):
        new_fit = self.calculate_fit(centroids_current)
        other_new_fit = self.calculate_fit(centroids_other)
        new_xs = self.calculate_xs(new_fit)

        if self.current_fit is None:
            # first iteration
            self.current_fit = new_fit
            self.best_fit.append(new_fit)
            self.allx = new_xs
            self.offset = offset
            self.centroid = centroids_current
            self.extrap.update_xfitted(centroids_current)
        else:
            if self.new_fit_valid(new_fit, other_new_fit, offset):
                self.current_fit = new_fit
                self.best_fit.append(new_fit)
                self.allx = new_xs
                self.offset = offset
                self.centroid = centroids_current
                self.extrap.update_xfitted(centroids_current)
                self.c = 0
            else:
                #no update, use average of old values
                self.increment_counter() #if this is incremented 5 times in a row, it forces a reset


    def get_best_fit(self):
        return np.mean(np.array(self.best_fit).reshape((-1, 3)), axis=0)

    def set_curverad(self, curverad):
        self.curverad = curverad

    def set_offset(self, offset):
        self.offset = offset

    def add_allx(self):
        self.recent_xfitted.append(self.allx)

    def is_parallel(self, xy1, xy2, tolerance):
        xy1.T.dot(xy2)  # the inner product is almost 1


    def calculate_fit(self, centroids):
        ys = range(720,0, -window_height)
        return np.polyfit(np.array(ys), centroids, 2)

    def calculate_xs(self, fit):
        return fit[0] * self.ally ** 2 + fit[1] * self.ally + fit[0]


    def local_search(img, left_centroid, right_centroid, predictions_left, predictions_right):
        """
        pair this with the acceleration idea and things shoudl be good
        :param left_centroid:
        :param right_centroid:
        :return:
        """
        window_centroids = []
        middle_window = window_height/2
        window = np.ones(window_width)  # Create our window template that we will use for convolutions
        margin = 150
        offset = window_width/2

        conv_bottom4th = np.convolve(window, np.sum(img[(int)(3*img.shape[0]/4):,:],axis=0))
        mean_conv = np.mean(conv_bottom4th)

        l_sum = np.sum(img[int(3 * img.shape[0] / 4):, :int(img.shape[1] / 2)], axis=0)
        l_center_4th = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        if abs(l_center_4th - left_centroid[0])> window_width:
            l_center = left_centroid[0]
        else:
            l_center = l_center_4th
        r_sum = np.sum(img[int(3 * img.shape[0] / 4):, int(img.shape[1] / 2):], axis=0)
        r_center_4th = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(img.shape[1] / 2)
        if abs(r_center_4th - right_centroid[0]) > window_width:
            r_center = right_centroid[0]
        else:
            r_center = r_center_4th
        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(img.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            l_center = left_centroid[(int)(img.shape[0] / window_height)-(level+1)]
            r_center = right_centroid[(int)(img.shape[0] / window_height)-(level+1)]
            image_layer = np.sum(
                img[int(img.shape[0] - (level + 1) * window_height):int(img.shape[0] - level * window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, img.shape[1]))

            if np.mean(conv_signal) <= mean_conv/3:  # a threshold based on the mean convolutional signal in the bottom forth
                print('accel was used')
                l_center = predictions_left[level]
            else:
                potential_l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
                if abs(potential_l_center - window_centroids[-1][0]) < window_width:
                    # only accept a new center if the line stays continuous
                    l_center = potential_l_center



            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, img.shape[1]))
            if np.sum(conv_signal) <= mean_conv/3:
                print('accel was used')
                r_center = predictions_right[level]
            else:
                potential_r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
                if abs(potential_r_center - window_centroids[-1][1]) < window_width:
                    r_center = potential_r_center
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))
        return np.array(window_centroids).reshape((int(img.shape[0]/window_height),2))











