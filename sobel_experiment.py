import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from test_pipeline import to_RGB

def grad_xy(img, dir='x', ksize=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if dir=='x':
        abs_sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    else:
        abs_sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    scale_factor = 255.0/np.max(abs_sobel)
    abs_sobel = np.uint8(scale_factor * abs_sobel)
    threshold_bit = np.zeros_like(abs_sobel)
    threshold_bit[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1
    return threshold_bit


def grad_theta(img, ksize=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    abs_sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    d_theta = np.arctan2(abs_sobely, abs_sobelx)
    threshold_bit = np.zeros_like(d_theta)
    threshold_bit[(d_theta >= thresh[0]) & (d_theta <= thresh[1])] = 1
    return threshold_bit


def grad_magnitude(img, ksize=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sumsq = sobelx*sobelx + sobely*sobely
    sqrt = np.sqrt(sumsq)
    scale_factor = 255 / np.max(sqrt)
    sqrt = np.uint8(scale_factor*sqrt)
    threshold_bit = np.zeros_like(sqrt)
    threshold_bit[(sqrt >= thresh[0]) & (sqrt <= thresh[1])] = 1
    return threshold_bit


def bit_and_transform(img):
    theta_grad = grad_theta(img, ksize=3, thresh=(math.pi/3-0.2, math.pi/3+0.2))
    mag_grad = grad_magnitude(img, ksize=3, thresh=(50, 160))
    return np.logical_and(mag_grad, theta_grad)

if __name__ == '__main__':
    img = mpimg.imread('test_images/test5.jpg')

    isolate_lane_lines = False

    if isolate_lane_lines:
        x_grad = grad_xy(img, dir='x',ksize=3,thresh=(50, 150))
        y_grad = grad_xy(img, dir='y', ksize=3, thresh=(50, 150))

        # test different angles
        theta_grad = grad_theta(img, ksize=3, thresh=(math.pi/4-0.2, math.pi/4+0.2))
        theta_grad2 = grad_theta(img, ksize=3, thresh=(math.pi/3-0.2, math.pi/3+0.2))
        theta_grad3 = grad_theta(img, ksize=3, thresh=(math.pi/2 - 0.2, math.pi/2))

        mag_grad = grad_magnitude(img, ksize=5, thresh=(50,160))

        bit_and = np.logical_and(mag_grad, theta_grad2) ## this is the best one
        bit_and2 = np.logical_and(mag_grad, theta_grad)

        bit_and3 = np.logical_and(bit_and, bit_and2)

        # compare two schemes
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(bit_and, cmap='gray')
        ax1.set_title('bit_and', fontsize=50)
        ax2.imshow(bit_and2, cmap='gray')
        ax2.set_title('bit and3', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        """
        http://docs.opencv.org/3.1.0/d0/d86/tutorial_py_image_arithmetics.html
        -- once lane lines found, fill in
        """
    else:
        mag_grad = grad_magnitude(img, ksize=7, thresh=(50,150))
        theta_grad = grad_theta(img, ksize=3, thresh = (math.pi/2-.1, math.pi/2))

        cv2.namedWindow('magnitude')
        cv2.imshow('magnitude', to_RGB(mag_grad))
        cv2.namedWindow('not small angle')
        cv2.imshow('not small angle', to_RGB(np.logical_not(theta_grad)))
        cv2.namedWindow('magnitude not small angle')
        cv2.imshow('magnitude not small angle', to_RGB(np.logical_and(mag_grad, np.logical_not(theta_grad))))

        cv2.waitKey()

        #x_grad = grad_xy(img, dir='x', ksize=3, thresh=(50, 100))
        #y_grad = grad_xy(img, dir='y', ksize=3, thresh=(50, 150))
        #and_xy_grad = np.logical_and(x_grad, y_grad)
        #cv2.namedWindow('xgrad')
        #cv2.imshow('xgrad', to_RGB(x_grad))
        #cv2.namedWindow('ygrad')
        #cv2.imshow('ygrad', to_RGB(y_grad))

        #cv2.namedWindow('and_xygrad')
        #cv2.imshow('and_xygrad', to_RGB(and_xy_grad))
        cv2.waitKey()