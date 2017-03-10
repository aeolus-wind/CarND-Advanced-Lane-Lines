import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from distortion import RemoveDistortion
import math
from colors import bit_and_transform, grad_magnitude, grad_theta, hls_decision_rule, threshold_saturation
from normalize_process_images import to_RGB


def click_bounding_box(event, x, y, flags, params):
    """
    influenced by: http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    :param event:
    :param x:
    :param y:
    :param flags:
    :param params:
    :return:
    """
    global bounding_box
    if event == cv2.EVENT_LBUTTONDOWN:
        bounding_box.append((x, y))

def transform_perspective2(img):
    #src = np.array([23, 568, 1274, 576, 578, 435, 716, 437], np.float32).reshape((4, 2))
    #dst = np.array([23, 700, 600, 700, 23, 100, 650, 100], np.float32).reshape((4, 2))
    #src = np.array([55,  594, 1256,  556,  651,  436,  771,  438], np.float32).reshape((4,2))
    #dst = np.array([55, 700, 600, 700, 55, 100, 650, 100], np.float32).reshape((4,2))

    #src = np.array([85, 564, 1224, 544, 767, 429, 588, 440], np.float32).reshape((4,2))
    src = np.array([124.66666667, 537., 1124.33333333, 521.33333333, 758.83333333, 435.16666667, 590.16666667, 436.], np.float32).reshape((4,2))
    dst = np.array([85, 700, 600, 700, 600, 100, 85, 100], np.float32).reshape((4,2))
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def summarize_bounding_boxes(bounding_boxes):
    # takes bounding boxes recorded in clockwise fashion
    # finds average
    points = np.array(bounding_boxes)
    return points.mean(axis=0)


def find_bounding_boxes(consider_test=False, consider_straight=True, size_box=4):
    global bounding_box
    rmv_distortion = RemoveDistortion()
    rmv_distortion.load_pickle()

    test_images = glob.glob('test_images/test*.jpg')
    straight_images = glob.glob('test_images/straight_lines*.jpg')

    if consider_test and consider_straight:
        images_considered = test_images + straight_images
    else:
        if consider_test:
            images_considered = test_images
        else:
            images_considered = straight_images
    for img_path in images_considered:
        bounding_box = []  # reset bounding_box
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', click_bounding_box)
        while True:
            img = cv2.imread(img_path)
            remove_distortion = rmv_distortion.undistort(img)
            cv2.imshow('image', remove_distortion)
            key = cv2.waitKey(1) & 0xFF

            if len(bounding_box) == size_box:
                bounding_boxes.append(np.array(bounding_box).reshape((-1, size_box*2)))
                break

            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    print(bounding_boxes)
    print(summarize_bounding_boxes(bounding_boxes))



def horizontal_shift(lower_points, upper_points):
    """
    :param c: some constant to be tuned
    :param lower_points: points of lower line closer to camera
    :param upper_points: points of higher line further from camera
    :return: an approximation of geometry which describes how to shift points-- uses assumption that
    radius of curvature is much larger than distance over which approximation is applied
    """
    lower_midpoint = midpoint(lower_points)
    upper_midpoint = midpoint(upper_points)
    upper_slope = (upper_points[3] - upper_points[1])/(upper_points[2] - upper_points[0])
    upper_b = -upper_slope * upper_points[2] + upper_points[3]
    y_project = (lower_midpoint[0]*upper_slope) + upper_b  # project the lower midpoint onto the upper line, holding x const
    # can approximate the geometry by assuming that this triagle is a right triangle and getting an angle of deviation
    distance_deviation = upper_midpoint[1] - y_project  # also an approx. gives the sign of the deviation
    distance_midpoints = distance(upper_midpoint, lower_midpoint)
    angle_deviation = math.atan2(distance_deviation, distance_midpoints)
    return angle_deviation  # used to scale

def center_shift_points(lower_points, upper_points, shape, angle_deviation):
    """
    :param lower_points: points closer to camera in image
    :param upper_points: points further from camera in image
    :param shape: potentially shape of image. currently, 've hardcoded for 1280 x 720 pixel shape
    :param angle_deviation: calculated from simplified trigonometry
    :return: relatively centered box
    """
    lower_y = 700  # map to the bottom of the frame
    adjust_lower_y = np.mean((lower_points[1],
                        lower_points[3])) - lower_y  # used to maintain aspect ratio to keep curve accurate
    upper_y = 600  # as upper is pushed down, the x coordinates on x_upper need to increase
    adj_upper_x = 20
    lower_points = lower_points.copy()
    lower_points[1] = lower_y
    lower_points[3] = lower_y
    x_shift_upper = math.tan(angle_deviation) * abs(lower_y - upper_y)  # correction for curvature constant
    upper_points = (lower_points[0] + x_shift_upper - adj_upper_x,      # in current code and mostly negligible
                    upper_y + adjust_lower_y,
                    lower_points[2] + x_shift_upper + adj_upper_x,
                    upper_y + adjust_lower_y)
    return lower_points, upper_points

def transform_perspective(img, src, dst, img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def invert_perspective(img, src, dst, img_size):
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped

"""
TODO: refactor functions below
"""

def default_transform_perspective(img):
    img_size = (img.shape[1], img.shape[0])
    #src = np.array([ 297,  685,  618,  444,  719,  444, 1097,  683], np.float32)
    src = np.float32(np.array([370, 623, 981, 625, 519, 506, 772, 493]))
    phi = horizontal_shift(src[:4], src[4:8])  # this transform is at this point mostly vestigial
    transformed_lower, transformed_upper = center_shift_points(src[:4], src[4:8], img_size, phi)
    src = src.reshape((4, 2))
    dst = np.float32(np.stack((transformed_lower, transformed_upper)).reshape((4, 2)))
    return transform_perspective(img, src, dst, img_size)

def default_invert_perspective(img):
    img_size = (img.shape[1], img.shape[0])
    #src = np.array([297, 685, 618, 444, 719, 444, 1097, 683], np.float32)
    src = np.float32(np.array([370, 623, 981, 625, 519, 506, 772, 493]))
    phi = horizontal_shift(src[:4], src[4:8])  # this transform is at this point mostly vestigial
    transformed_lower, transformed_upper = center_shift_points(src[:4], src[4:8], img_size,phi)
    src = src.reshape((4, 2))
    dst = np.float32(np.stack((transformed_lower, transformed_upper)).reshape((4, 2)))
    return invert_perspective(img, src, dst, img_size)

if '__main__' == __name__:
    # http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
    # http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    rmv_distortion = RemoveDistortion()
    rmv_distortion.load_pickle()

    test_images = glob.glob('test_images/test*.jpg')
    straight_lines = glob.glob('test_images/straight_lines*.jpg')

    bounding_boxes = []

    run_find_bounding_boxes = False
    first_round = False
    second_round = False
    if run_find_bounding_boxes:
        find_bounding_boxes(True, False, 4)
    elif first_round:
        test_image = cv2.imread('test_images/straight_lines1.jpg')
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        undistort = rmv_distortion.undistort(test_image)
        #works for straight road
        #sample_bound_box = np.array([317, 640, 990, 646, 490, 522, 801, 524])
        # curved road
        sample_bound_box = np.array([390, 600, 940, 590, 560, 480, 745, 470], np.float32)


        shape = (1280,720)
        phi = horizontal_shift(sample_bound_box[:4], sample_bound_box[4:8])
        transformed_lower, transformed_upper = center_shift_points(sample_bound_box[:4], sample_bound_box[4:8], shape, phi)
        transformed = transform_perspective(to_RGB(bit_and_transform(undistort)),
                                         np.float32(sample_bound_box.reshape((4,2))),
                                         np.float32(np.stack([transformed_lower, transformed_upper]).reshape((4, 2))),
                                         shape)
        plt.imshow(transformed, cmap='gray')
        plt.show()
    elif second_round:
        test_image = mpimg.imread('test_images/test2.jpg')
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        undistort = rmv_distortion.undistort(test_image)
        transformed = transform_perspective2(undistort)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undistort, cmap='gray')
        ax1.set_title('undistort', fontsize=50)
        ax2.imshow(transformed, cmap='gray')
        ax2.set_title('perspective shift', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    else:
        img = cv2.imread('test_images/straight_lines1.jpg')
        src = np.array([390, 600, 940, 590, 560, 480, 745, 470], np.float32).reshape((4, 2))
        dst = np.array([405, 670, 600, 670, 430, 300, 635, 410], np.float32).reshape((4, 2))
        M = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        boolean = to_RGB(np.logical_or(grad_magnitude(warped, ksize=7, thresh=(50, 255)),
                                       threshold_saturation(warped, (80, 255))))


        cv2.namedWindow('test')
        cv2.imshow('test', warped)

        cv2.imshow('boolean', boolean)


        #cv2.namedWindow('filter_windows')
        #cv2.imshow('filter_windows', to_RGB(grad_magnitude(warped,ksize=11, thresh=(5,100))))


        cv2.waitKey()









