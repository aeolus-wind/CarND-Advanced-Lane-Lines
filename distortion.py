import cv2
import glob
import numpy as np
import pickle
import os


class RemoveDistortion():
    def __init__(self):

        self.objpoints = None
        self.imgpoints = None
        self.mtx = None
        self.dist = None


    def fit_obj_img_points(self, img_files_str, xsquares, ysquares, debug = False):
        # define the coordinates of a chess board
        # a standard practice in calibrating a transform
        # which removes distortion
        objp = np.zeros((xsquares*ysquares, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        images = glob.glob(img_files_str)

        objpoints = []
        imgpoints = []

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (xsquares, ysquares), None)

            if debug:
                # show the images as they are tagged
                print('index ', idx, ' succeeded: ', ret)
                cv2.drawChessboardCorners(img, (xsquares, ysquares), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        self.objpoints = objpoints
        self.imgpoints = imgpoints

    def fit_pickle_obj_img_points(self, img_files_str='./camera_cal/calibration*.jpg', xsquares=9, ysquares=6,
                                  debug=False, picklepath='camera_cal/transform.p'):
        self.fit_obj_img_points(img_files_str, xsquares, ysquares, debug=debug)

        # get image size
        img = cv2.imread('./camera_cal/calibration1.jpg')
        imgsize = (img.shape[1], img.shape[0])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, imgsize, None, None)
        self.mtx = mtx
        self.dist = dist

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(picklepath, "wb"))

    def load_pickle(self, picklepath='camera_cal/transform.p'):
        try:
            dist_pickle = pickle.load(open(picklepath, "rb"))
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]
        except IOError:
            print("distortion effects not calibrated")

    def undistort(self, src):
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(src, self.mtx, self.dist, None, self.mtx)
        else:
            raise ValueError("mtx and dist have not been initialized")

    def load_undistort(self, src, picklepath='camera_cal/transform.p'):
        self.load_pickle(picklepath)
        return self.undistort(src)


"""
set this up so that this step is pickled
"""


if __name__ == '__main__':

    rmv_distortion = RemoveDistortion()
    # to create the pickled values run the line below
    # rmv_distortion.fit_pickle_obj_img_points(debug=True)
    rmv_distortion.load_pickle()
    """
    ### Some tests to see if the calibration is reasonable
    0, 14, 15 failed so I check those
    """
    test = cv2.imread('camera_cal/calibration1.jpg')
    test2 = cv2.imread('camera_cal/calibration15.jpg')
    test3 = cv2.imread('camera_cal/calibration16.jpg')

    undistort = rmv_distortion.undistort(test)
    undistort2 = rmv_distortion.undistort(test2)
    undistort3 = rmv_distortion.undistort(test3)

    #cv2.imshow('undistort', undistort)
    #cv2.imshow('undistort2', undistort2)
    #cv2.imshow('undistort3', undistort3)
    #cv2.waitKey()
    cv2.imwrite('undistorted_output.png', cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB))
    img = rmv_distortion.undistort(cv2.imread('test_images/test1.jpg'))
    cv2.imwrite('writeup_images/undistorted_road.jpg', img)

