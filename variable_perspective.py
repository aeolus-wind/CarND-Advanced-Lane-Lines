import numpy as np
from colors import grad_magnitude, hls_decision_rule
import cv2
import glob
from watch_video import to_RGB
import math
import unittest
import matplotlib.pyplot as plt
from distortion import RemoveDistortion


roi = np.array([17,575, 520, 420, 900, 420, 1280, 470, 1100, 700, 230, 680], np.int32).reshape((6, 2))


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=1, β=0.5, λ=0.):
    """
    Overlaps two images lowering brightness of one over the other
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def find_regions(event, x, y, flag, params):
    global bounding_box
    if event == cv2.EVENT_LBUTTONDOWN:
        bounding_box.append((x,y))
    if event == cv2.EVENT_LBUTTONUP:
        bounding_box.append((x,y))

"""
Dictionaries below define parameters of a filter
points are (x,y) and denote top and bottom corner of a rectangle
"""
outer_region_left = [
    {'start_region': ((0, 391),(435, 720)),
     'end_region': ((0, 0),(1280, 720)),
     'slope_sign':'negative',
     'slope_bounds':(0.2, math.pi/2)},
    {'start_region':((9, 488),(314, 720)),  # picks up lines on outer left, but can keep
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'negative',
     'slope_bounds': (0.1, math.pi/2)
     },
    {'start_region':((208,606), (354,711)),  # second priority outer_region
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign': 'negative',
     'slope_bounds':(0.0, math.pi/2)}
]
inner_region_left = [
    {'start_region':((150, 552),(559, 720)),  # note the lowering of threshold of small slopes
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'negative',
     'slope_bounds': (0.15, math.pi/2)
     },
    {'start_region':((150, 512),(559, 720)),  # note the lowering of threshold of small slopes
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'negative',
     'slope_bounds': (0.1, math.pi/2)
     },
    {'start_region':((150, 452),(559, 720)),  # note again the lowering of threshold for lines of small slopes
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'negative',
     'slope_bounds': (0.0, math.pi/2)
     }
]

outer_region_right = [
    {'start_region':((814, 550),(1267, 720)),
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.15, math.pi/2)
     },
    {'start_region':((804, 501),(1276, 720)),
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.1, math.pi/2)
     },
    {'start_region':((809, 450),(1269, 720)), #liberal picks up many lines, but will need some filter/weight policy
     'end_region': ((0, 0),(1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.0, math.pi/2)
     }
]

inner_region_right = [
    {'start_region':((600, 458),(900, 655)),  # picks up lines on outer left, failes on one image
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.2, math.pi/2)
     },
    {'start_region':((500, 438),(900, 715)),  # note the lowering of threshold to include lines with smaller slopes
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.1, math.pi/2)
     },
    {'start_region':((506, 463),(900, 718)),  # note the secondary see if it causes problems
     'end_region': ((0, 0), (1280, 720)),
     'slope_sign':'positive',
     'slope_bounds': (0.0, math.pi/2)
     }
]


def filter_lines(lines, start_region=((0,0), (1280,720)),
                 end_region=((0, 0), (1280, 720)),
                 slope_sign = 'positive',slope_bounds=(0, math.pi/2)):
    lines = lines.reshape((-1, 4))
    x_restriction_start = (lines[:, 0] >= start_region[0][0]) & (lines[:, 0] <= start_region[1][0])
    y_restriction_start = (lines[:, 1] >= start_region[0][1]) & (lines[:, 1] <= start_region[1][1])

    lines_start_restriction = lines[(x_restriction_start & y_restriction_start)]
    x_restriction_end = (lines_start_restriction[:,0] >= end_region[0][0]) \
                        & (lines_start_restriction[:,0] <= end_region[1][0])
    y_restriction_end = (lines_start_restriction[:,1] >= end_region[0][1]) \
                        & (lines_start_restriction[:,1] <= end_region[1][1])
    lines_end_restriction = lines_start_restriction[x_restriction_end & y_restriction_end]
    slope = (lines_end_restriction[:, 1] - lines_end_restriction[:,3]) \
            / (lines_end_restriction[:, 0] - lines_end_restriction[:,2])
    if slope_sign == 'positive':
        restrict_slope_sign = slope > 0
    else:
        restrict_slope_sign = slope <= 0
    theta_restriction = np.arctan(np.abs(slope))

    lines_all_restrictions = lines_end_restriction[(theta_restriction >= slope_bounds[0])
                                     & (theta_restriction <= slope_bounds[1])
                                     & restrict_slope_sign]
    return lines_all_restrictions


def find_filtered_candidate_lines(img):
    """
    :param img: image of lane lines
    :return: applies all filters above. The 'sub filters' are implicitly ranked by order so they relax in strength if
    no results appear from the stronger candidate
    """
    grad_img = np.uint8(np.logical_or(grad_magnitude(img, 7, (50, 150)), hls_decision_rule(img)))

    region_interest = region_of_interest(grad_img, [roi])
    highly_probable_bounding_lines = [outer_region_left, outer_region_right,
                                      inner_region_right, inner_region_left]
    probable_names = ['outer_region_left', 'outer_region_right', 'inner_region_right', 'inner_region_left']

    best_lines = {}
    lines = cv2.HoughLinesP(region_interest, 1, math.pi / 180, 20, np.array([]),
                            minLineLength=20, maxLineGap=50)
    for bounding_line_set, name in zip(highly_probable_bounding_lines, probable_names):
        for top_priority in bounding_line_set:
            temp_holder_lines = filter_lines(lines, **top_priority).reshape((-1,1,4))
            if len(temp_holder_lines) > 0:
                best_lines[name] = temp_holder_lines
                break
    return best_lines


def get_slopes_intercepts(lines):
    slopes = ((lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])).reshape((-1, 1))
    intercepts = (lines[:, 3].reshape((-1, 1)) - slopes * lines[:, 2].reshape(-1, 1)).reshape((-1, 1))
    return slopes, intercepts


def project_y_to_x(ys, slopes, intercepts):
    return (ys - np.repeat(intercepts, len(ys), axis=1)) \
         / (np.repeat(slopes, len(ys), axis=1))  # use broadcasting to form a matrix of all xs from each line


def collect_filter_projections(img, lines, ys):
    lines = lines.reshape((-1,4))
    slopes, intercepts = get_slopes_intercepts(lines)
    xs = project_y_to_x(ys, slopes, intercepts)
    projected_lines = np.column_stack((lines[:, 0], lines[:, 1], xs, np.repeat(ys, len(xs),axis=0))).reshape((-1, 1, 4))
    valid_x = (projected_lines[:, :, 3] >= 0) & (projected_lines[:, :, 2] <= img.shape[1])
    remove_inf = (projected_lines[:,:,3] != np.inf) & (projected_lines[:, :, 2] != -np.inf)
    projected_lines = projected_lines[valid_x & remove_inf] #stay in picture

    return projected_lines


def extrapolate_to_vanishing(lines, y_region=(350, 450), threshold=5, x_region=(400, 800)):
    """
    :param lines: lines from hough lines algorithm
    :param y_region: range of y's where the vanishing point may lie
    :param threshold: number of points needed to be considered connected to vanishing point
    :param x_region: range of x's where the vanishing point may lie
    :return: the lines which likely end up at a vanishing point
    """

    slopes, intercepts = get_slopes_intercepts(lines.reshape((-1,4)))

    ys = np.linspace(y_region[0], y_region[1], 100).reshape((-1, 100))  # a row vector of y's in area of interest
    xs = project_y_to_x(ys, slopes, intercepts)\

    xs[(xs == np.inf) | (xs == -np.inf)] = 0  # shape must be preserved, so we set offending points to 0
    xs[(xs <= 0) | (xs >= 1280)] = 0
    xs[np.isnan(xs)] = 0

    bins = np.array(range(0, 1282, 2))  # large bins included to catch things outside range

    bin_region_restriction = (bins >= x_region[0]) & (bins <= x_region[1])

    bin_numbers = np.digitize(xs, bins)  # indices of appropriate bin

    bin_size = np.bincount(bin_numbers.flatten())

    max_bin_index = np.max(bin_numbers)  # bincount finds counts from 0 to max index present

    bin_index_count = np.zeros_like(bins)
    bin_index_count[0:max_bin_index+1] = bin_size  # embed into all of bin space
    bin_size_threshold_restriction = bin_index_count >= threshold

    successful_bins = bin_region_restriction & bin_size_threshold_restriction
    hit_successful_bin = successful_bins[bin_numbers]

    successful_lines = np.any(hit_successful_bin, axis=1)
    return successful_lines


def generate_simple_non_trivial_test_case(tuple_of_thetas=(5 * math.pi/4, 7*math.pi/4), r=(200,100)):
    array_of_thetas = np.array(tuple_of_thetas)
    unit_circle_points= np.column_stack((np.cos(array_of_thetas), np.sin(array_of_thetas)))
    return np.hstack((r[0]*unit_circle_points, r[1]*unit_circle_points))


class TestExtrapolate_To_Vanishing(unittest.TestCase):
    # trivial set of lines-- the same indicated by different points
    t_x1 = np.array(range(10))
    t_y1 = np.array(range(2, 22, 2))
    t_x2 = np.array(range(1, 11))
    t_y2 = np.array(range(4, 24, 2))
    t_lines = np.column_stack([t_x1, t_y1, t_x2, t_y2])

    def test_trivial_slopes(self):
        slopes, intercepts = get_slopes_intercepts(self.t_lines)
        self.assertTrue(
            (slopes == np.array(
                [[ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.]])
             ).all()
        )
        self.assertTrue(
            (intercepts == np.array(
                [[ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.], [ 2.]])
             ).all())

    def test_trivial_case(self):
        successful_lines = extrapolate_to_vanishing(self.t_lines, y_region=(350, 450), threshold=3, x_region=(348/2, 448/2))
        self.assertTrue(np.all(successful_lines))

    def test_non_trivial_case(self):
        lines = generate_simple_non_trivial_test_case() + np.array((500, 500, 500, 500))
        # based on a circle, the lines should have convergence point at 0,0
        successful_lines = extrapolate_to_vanishing(lines, y_region=(499, 501), threshold=3, x_region=(499, 501))
        self.assertTrue(np.all(successful_lines))


def weighted_avg3(x, w1, w2):
    return np.sum(x * w1 * w2) / np.sum(w1*w2)


def weighted_avg2(x,w1):
    return np.sum(x*w1)/np.sum(w1)


def find_hull(lines):
    # find a polygon which describes parallel lanes in image
    # is biased toward larger ones because I believe they result in
    # more stable transforms
    lines = lines.reshape((-1,4))
    length = np.sqrt((lines[:,0] - lines[:,2])**2 + (lines[:,1] - lines[:,3])**2)
    slopes, intercepts = get_slopes_intercepts(lines)
    slopes = slopes.flatten()
    intercepts = intercepts.flatten()

    finite_slopes = (slopes >= -50) & (slopes <= 50) & (slopes != np.inf) & (slopes != -np.inf)
    slopes = slopes[finite_slopes]  # finite slopes will have finite intercepts for this
    intercepts = intercepts[finite_slopes]
    length = length[finite_slopes]
    left_lines = (slopes < 0)

    right_lines = (slopes > 0)

    # chosen lines biased towards starting close to these points
    l_bias = (150, 720)
    r_bias = (820, 720)
    left_closest_to_point = np.sqrt((lines[:, 0] - l_bias[0]) ** 2 + (lines[:, 1] - l_bias[1]) ** 2)
    left_closest_to_point = left_closest_to_point[finite_slopes]

    left_slope = weighted_avg3(slopes[left_lines].flatten(),
                               length[left_lines].flatten(),
                               left_closest_to_point[left_lines].flatten())

    #left_slope = weighted_avg2(slopes[left_lines].flatten(), length[left_lines].flatten())

    left_intercept = weighted_avg3(intercepts[left_lines].flatten(),
                                   length[left_lines].flatten(),
                                   left_closest_to_point[left_lines].flatten())
    #left_intercept = weighted_avg2(intercepts[left_lines].flatten(),
                                   #length[left_lines].flatten())

    right_closest_to_point = np.sqrt((lines[:, 0] - r_bias[0])**2 + (lines[:, 1] - r_bias[1])**2)
    right_closest_to_point = right_closest_to_point[finite_slopes]

    right_slope = weighted_avg3(slopes[right_lines].flatten(),
                                length[right_lines].flatten(),
                                right_closest_to_point[right_lines].flatten())
    #right_slope = weighted_avg2(slopes[right_lines].flatten(),
    #                            length[right_lines].flatten())

    right_intercept = weighted_avg3(intercepts[right_lines],
                                    length[right_lines],
                                    right_closest_to_point[right_lines])
    #right_intercept = weighted_avg2(intercepts[right_lines].flatten(),
    #                                length[right_lines].flatten())

    # map points-- points in the hull which determine the perspective transform
    # wider distance in the y's indicates more faith in the detected lines
    y_lower = 700
    y_upper = 500  # this one is the one to tune-- this is the outer pixel where
                   # you believe your is like parallel lines int the real world

    # x, y coordinates are calculated using the typical method
    left_line = ((y_lower-left_intercept)/left_slope, y_lower, (y_upper - left_intercept)/left_slope, y_upper)
    right_line = ((y_lower-right_intercept)/right_slope, y_lower, (y_upper - right_intercept)/right_slope, y_upper)
    return left_line, right_line


def filter_lines_find_hull(img):
    """
    :param img: road image
    :return: performs a filtered hough line transform, then filters those based on whether they have a vanishing point
    """
    best_lines = find_filtered_candidate_lines(img)  # returns a dictionary labeled by the different filters
    best_lines_accum = []

    # put lines in default format
    for segment, lines in best_lines.items():
        best_lines_accum.append(lines)

    best_lines_accum = np.int32(np.concatenate(best_lines_accum).reshape((-1, 1, 4)))
    convergent_lines = extrapolate_to_vanishing(best_lines_accum, y_region=(350, 450),
                                                threshold=10, x_region=(550, 750))
    return find_hull(best_lines_accum[convergent_lines])


def transform_coordinates_hull(img, hull, x_bound=(100, 1300), invert=False):
    slopes, intercepts = get_slopes_intercepts(np.array(hull).reshape(-1, 4))
    xs = project_y_to_x([600], slopes, intercepts)  # want to get distance between the mid-points at the bottom
    x_left_shift = 250 - xs[0]
    x_right_shift = xs[1] - 1000
    y_upper = 480  # this also affects distortion in the picture.
                   # the closer to y_lower it is, the more the image is stretched
    y_lower = 700
    src = np.array(hull,np.float32).reshape((4,2))
    dst = np.array([[x_bound[0], y_lower], [x_bound[0], y_upper], [x_bound[1], y_lower],
                    [x_bound[1], y_upper]], np.float32)
    #dst = np.array([[x_bound[0]+x_shift,y_lower],[x_bound[0]+x_shift,y_upper], [x_bound[1]+x_shift,y_lower],[x_bound[1]+x_shift,y_upper]], np.float32)
    if invert:
        M = cv2.getPerspective(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def get_perspective_transform_matrix(img, hull, x_bound=(100, 1000)):
    slopes, intercepts = get_slopes_intercepts(np.array(hull).reshape(-1, 4))
    xs = project_y_to_x([600], slopes, intercepts)
    x_left_shift = 250 - xs[0]
    x_right_shift = xs[1] - 1000
    y_upper = 550  # this also affects distortion in the picture.
                   # the closer to y_lower it is, the more the image is stretched
    y_lower = 700
    src = np.array(hull,np.float32).reshape((4,2))
    dst = np.array([[x_bound[0]-x_left_shift,y_lower],[x_bound[0]-x_left_shift,y_upper], [x_bound[1]+x_right_shift,y_lower],[x_bound[1]+x_right_shift,y_upper]], np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    invert_matrix = cv2.getPerspectiveTransform(dst, src)
    return transform_matrix, invert_matrix

"""
todo: wrap all of these transforms into an object that holds the matrices and hull
"""

"""
if __name__ == '__main__':
    unittest.main()
"""

if __name__ == '__main__':
    straight_images = glob.glob('test_images/straight_lines*.jpg')
    curved_images = glob.glob('test_images/test*.jpg')

    find_regions_interest = False
    see_all_lines = False
    test_dictionary = False
    test_local_filters = False
    view_extrapolated_lines = False
    test_hull_on_sample = False
    if find_regions_interest:

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', find_regions)
        bounding_boxes = []

        for img_path in straight_images + curved_images:
            bounding_box = []
            img = cv2.imread(img_path)
            rmv_distortion = RemoveDistortion()
            rmv_distortion.load_pickle()
            undistort = rmv_distortion.undistort(img)
            while True:
                cv2.imshow('img', undistort)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key == ord('r'):
                    bounding_box = []  # reset

                if len(bounding_box) == 4:
                    bounding_boxes.append(bounding_box)
                    break
        print(bounding_boxes)
    else:
        for img_path in straight_images + curved_images:
            img = cv2.imread(img_path)
            rmv_distortion = RemoveDistortion()
            rmv_distortion.load_pickle()
            undistort = rmv_distortion.undistort(img)
            if see_all_lines:
                grad_mag = grad_magnitude(undistort,7, (50, 150))
                region_interest = region_of_interest(grad_mag,[roi])
                lines = hough_lines(region_interest,
                                    rho=1,
                                    theta= math.pi/180,
                                    threshold=20, min_line_len=20,
                                    max_line_gap=100)
                add_lines = weighted_img(lines, to_RGB(region_interest))
                cv2.imshow('roi', to_RGB(add_lines))
                cv2.waitKey()
            elif test_dictionary:
                grad_mag = grad_magnitude(img, 7, (50, 150))
                region_interest = region_of_interest(grad_mag, [roi])
                cv2.namedWindow('region interest')
                cv2.imshow('region interest', to_RGB(region_interest))
                lines = cv2.HoughLinesP(region_interest, 1, math.pi/180, 20, np.array([]), minLineLength=20,
                                        maxLineGap=100)
                filtered_lines = filter_lines(lines,
                                              start_region=((208,606), (354,711)),  # bottom left corner
                                              end_region=((0,0), (1280, 720)),
                                              slope_bounds=(0, math.pi/2)).reshape((-1,1,4))
                blank_lines_img = np.zeros((region_interest.shape[0], region_interest.shape[1],3))
                draw_lines(blank_lines_img, filtered_lines)
                cv2.namedWindow('lines')
                cv2.imshow('lines', blank_lines_img)
                filtered_line_image = weighted_img(np.uint8(blank_lines_img), np.uint8(img))
                cv2.namedWindow(img_path + 'filtered lines')
                cv2.imshow(img_path + 'filtered lines', filtered_line_image)
                cv2.waitKey()
                cv2.destroyWindow(img_path + 'filtered lines')
            elif test_local_filters:
                # basically, look at what each filter does on its own
                filter_priority = 0  # this is the index of the considered set of filters
                current_filter_region = outer_region_right


                grad_mag = np.uint8(grad_magnitude(undistort, 7, (50, 150)))
                region_interest = region_of_interest(grad_mag, [roi])
                cv2.namedWindow('region interest')
                cv2.imshow('region interest', to_RGB(region_interest))
                lines = cv2.HoughLinesP(region_interest, 1, math.pi / 180, 20, np.array([]), minLineLength=20,
                                        maxLineGap=100)
                filtered_lines = filter_lines(lines, **current_filter_region[filter_priority]).reshape((-1,1,4))
                blank_lines_img = np.zeros((region_interest.shape[0], region_interest.shape[1], 3))
                draw_lines(blank_lines_img, filtered_lines)

                filtered_line_image = weighted_img(np.uint8(blank_lines_img), np.uint8(img))
                """
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
                f.tight_layout()

                ax1.imshow(blank_lines_img, cmap='gray')
                ax2.imshow(cv2.cvtColor(filtered_line_image, cv2.COLOR_BGR2RGB))
                ax1.set_title('blank lines img')
                ax2.set_title('filtered_line_image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.show()
                """



                cv2.namedWindow('lines')
                cv2.imshow('lines', blank_lines_img)
                filtered_line_image = weighted_img(np.uint8(blank_lines_img), np.uint8(undistort))
                cv2.namedWindow(img_path + 'filtered lines')
                cv2.imshow(img_path + 'filtered lines', filtered_line_image)
                cv2.waitKey()
                cv2.destroyWindow(img_path + 'filtered lines')
            elif view_extrapolated_lines:
                # start region and angle filters
                best_lines = find_filtered_candidate_lines(undistort)

                best_lines_accum = []
                for segment, lines in best_lines.items():
                    best_lines_accum.append(lines)
                best_lines_accum = np.int32(np.concatenate(best_lines_accum).reshape((-1, 1, 4)))

                # convergece filter
                convergent_lines = extrapolate_to_vanishing(best_lines_accum, y_region=(350, 450),
                                                            threshold=10, x_region=(550, 750))

                # resulting lines to test by projecting outwards
                projected_lines = collect_filter_projections(img, best_lines_accum, np.array([400]))
                filtered_projected_lines = collect_filter_projections(undistort, best_lines_accum[convergent_lines],
                                                                      np.array([400]))

                blank = np.zeros_like(img)
                draw_lines(blank, np.int32(projected_lines).reshape((-1, 1, 4)))
                filtered_line_image = weighted_img(blank, np.uint8(undistort))

                blank2 = np.zeros_like(img)
                draw_lines(blank2, np.int32(filtered_projected_lines).reshape(-1,1,4))
                convergence_line_image = weighted_img(blank2, np.uint8(undistort))

                cv2.namedWindow('test')
                cv2.imshow('test', blank)
                cv2.namedWindow('img w lines')
                cv2.imshow('img w lines', filtered_line_image)
                cv2.namedWindow('convergence lines')
                cv2.imshow('convergence lines', convergence_line_image)

                cv2.waitKey()

            elif test_hull_on_sample:
                """
                test dealing with effectiveness of hull
                """

                convergent_lines_hull = filter_lines_find_hull(undistort)

                blank = np.zeros_like(undistort)
                draw_lines(blank, np.int32(convergent_lines_hull).reshape((-1, 1, 4)))

                convergent_line_image = weighted_img(np.uint8(blank), np.uint8(undistort))

                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(cv2.cvtColor(convergent_line_image, cv2.COLOR_BGR2RGB))
                ax2.imshow(cv2.cvtColor(transform_coordinates_hull(img, convergent_lines_hull), cv2.COLOR_BGR2RGB))
                ax1.set_title('unfiltered vanishing point image')
                ax2.set_title('convergent vanishing point lines')
                plt.show()
            else:
                rmv_distortion = RemoveDistortion()
                rmv_distortion.load_pickle()
                img = cv2.imread('run10/problem_image0.jpg')
                cv2.namedWindow('test')
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                undistort = rmv_distortion.undistort(img)
                grad_mag = grad_magnitude(undistort, 3, (50, 150))
                cv2.imshow('test', to_RGB(grad_mag))

                hls_added = np.logical_or(grad_mag, hls_decision_rule(undistort))

                cv2.namedWindow('view')
                cv2.imshow('view', to_RGB(hls_added))
                cv2.waitKey()




