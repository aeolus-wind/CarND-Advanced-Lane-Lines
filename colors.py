import cv2
import numpy as np
from watch_video import to_RGB
import numpy as np
import math


def threshold_saturation(img, thresh=(0,255)):
    s_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    binary_img = np.zeros_like(s_img)
    binary_img[(s_img >= thresh[0]) & (s_img <= thresh[1])] = 1
    return binary_img


def threshold_hue(img, thresh=(0,179)):
    h_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 0]
    binary_img = np.zeros_like(h_img)
    binary_img[(h_img >= thresh[0]) & (h_img <= thresh[1])] = 1
    return binary_img


def threshold_lightness(img, thresh=(0,255)):
    l_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]
    binary_img = np.zeros_like(l_img)
    binary_img[(l_img >= thresh[0]) & (l_img <= thresh[1])] = 1
    return binary_img


def hls_decision_rule(img):
    # The rule works the best in conjunctionw with the features we get from sobel
    hls_rule = np.logical_and(threshold_hue(img, (0,70)), threshold_saturation(img, (170, 255)))
    return hls_rule


def or_decision_rule(img):
    # I am skeptical this works in all cases
    # will experiment with a linear combination threshold rule
    return np.logical_or(hls_decision_rule(img), bit_and_transform(img))


def click_avg_color(event, x, y, flags, params):
    global bounding_box

    if event == cv2.EVENT_LBUTTONDOWN:
        bounding_box = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        bounding_box.append((x,y))

"""
Filters relying on gradients
"""

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


def grad_theta_no_thresh(img,ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    abs_sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    d_theta = np.arctan2(abs_sobely, abs_sobelx)
    return d_theta

def grad_theta(img, ksize=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    abs_sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))
    d_theta = np.arctan2(abs_sobely, abs_sobelx)
    threshold_bit = np.zeros_like(d_theta)
    threshold_bit[(d_theta >= thresh[0]) & (d_theta <= thresh[1])] = 1
    return threshold_bit


def grad_magnitude_no_thresh(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sumsq = sobelx * sobelx + sobely * sobely
    sqrt = np.sqrt(sumsq)
    scale_factor = 255 / np.max(sqrt)
    sqrt = np.uint8(scale_factor * sqrt)
    return sqrt


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
    img = cv2.imread('test_images/test6.jpg')
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    analyze_colors = False
    test_lane_colors = True
    if analyze_colors:
        cv2.namedWindow('image')
        bounding_box = []
        cv2.setMouseCallback('image', click_avg_color)

        while True:
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF


            if key == ord('q'):
                break
        if len(bounding_box) == 2:
            print(bounding_box)

            pixels_of_interest = img[bounding_box[0][1]: bounding_box[1][1],
                                     bounding_box[0][0]: bounding_box[1][0]]
            hls_pixels_of_interest = hls_img[bounding_box[0][1]: bounding_box[1][1],
                                            bounding_box[0][0]: bounding_box[1][0]]
            print("mean BGR is: ")
            print(np.mean(pixels_of_interest.reshape((-1,3)), axis=0))

            print("mean HLS is: ")
            print(np.mean(hls_pixels_of_interest.reshape((-1,3)), axis=0))
            print(np.std(hls_pixels_of_interest.reshape((-1,3)), axis=0))

    elif test_lane_colors:

        cv2.namedWindow('original')
        cv2.imshow('original', img)

        cv2.namedWindow('image Hue')
        cv2.imshow('image Hue', to_RGB(threshold_saturation(img, (70, 180))))

        cv2.namedWindow('img saturation')
        cv2.imshow('img saturation', to_RGB(threshold_hue(img, (50, 150)))) #saturation great for yellow
                                                 # great for white lines too...

        cv2.namedWindow('combined rule')
        cv2.imshow('combined rule', to_RGB(hls_decision_rule(img)))

        cv2.namedWindow('combined and sobel')
        cv2.imshow('combined and sobel', to_RGB(np.bitwise_or(hls_decision_rule(img), bit_and_transform(img))))

        cv2.waitKey()


