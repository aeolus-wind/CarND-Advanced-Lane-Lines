import cv2
import numpy as np
from distortion import RemoveDistortion
from variable_perspective import filter_lines_find_hull, get_perspective_transform_matrix
from variable_perspective import extrapolate_to_vanishing, collect_filter_projections, find_filtered_candidate_lines
from variable_perspective import get_slopes_intercepts
from find_lanelines import find_window_centroids, bound_lanes, draw_window_centroids
from colors import hls_decision_rule, or_decision_rule
from find_lanelines import window_width, window_height, margin  # some constants to be used by the algorithm
from find_lanelines import distance_from_center_lane, transform_lane, radius_curvature
from variable_perspective import draw_lines
from watch_video import to_RGB
from colors import grad_theta, grad_magnitude
from variable_perspective import region_of_interest, roi
from smooth_value import ExponentialSmoother, RunningAverage
import math

problem_image_counter = 0
smooth_hull_left_slope = ExponentialSmoother(alpha=0.8)
smooth_hull_right_slope = ExponentialSmoother(alpha=0.8)

def insert_diag_into(frame, diag, x_slice, y_slice):
    # should take in upper left and lower right pixel location
    x_shape = x_slice.stop - x_slice.start
    y_shape = y_slice.stop - y_slice.start
    frame[x_slice, y_slice] = cv2.resize(to_RGB(diag), (y_shape, x_shape), interpolation=cv2.INTER_AREA)


def testing_pipeline(img):
    global problem_image_counter
    global smooth_hull_left_slope
    global smooth_hull_right_slope
    rmv_distortion = RemoveDistortion()
    rmv_distortion.load_pickle()
    undistort = rmv_distortion.undistort(img)

    # color and gradient filters
    hls_colors = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)
    grad_mag = np.uint8(grad_magnitude(undistort, 7, (50, 150)))
    theta_grad = grad_theta(img, ksize=3, thresh=(math.pi/3-0.2, math.pi/3+0.2))
    binary_img = or_decision_rule(undistort)

    try:
        hull = filter_lines_find_hull(undistort)

        hull_slopes, _ = get_slopes_intercepts(np.array(hull).reshape((2,4)))
        previous_smooth_left = smooth_hull_left_slope.get_smoothed_value()
        if previous_smooth_left is not None:
            #if abs((previous_smooth_left - hull_slopes[0]) / previous_smooth_left) > 0.1: #if change is greater than 10%
            smooth_hull_left_slope.update(hull_slopes[0])
            # new x2's with adjusted slopes
            smoothed_left_slope = smooth_hull_left_slope.get_smoothed_value()
            hull[0][2] = (hull[0][3] - hull[0][1]) / smoothed_left_slope + hull[0][0]
        previous_smooth_right = smooth_hull_right_slope.get_smoothed_value()
        if previous_smooth_right is not None:
            #if abs((previous_smooth_right - hull_slopes[1]) / previous_smooth_right) > 0.1: #10% worked in assignment 1
            smooth_hull_right_slope.update(hull_slopes[1])
            # new x2's with adjusted slopes
            smoothed_right_slope = smooth_hull_right_slope.get_smoothed_value()
            hull[1][2] = (hull[1][3] - hull[1][1]) / smoothed_right_slope + hull[1][0]

        hull_lines_img = np.zeros_like(undistort)
        draw_lines(hull_lines_img, np.int32(hull).reshape((-1, 1, 4)), thickness=30)
        hull_lines_img = cv2.addWeighted(hull_lines_img, 1.0, undistort, 1.0, 0.0)
    except ValueError:
        print("hull was ", hull, "... wrong type")
        hull_lines_img = np.zeros_like(undistort)
        cv2.imwrite('problem_image' + str(problem_image_counter) + '.jpg', img)
        problem_image_counter += 1
    except:
        print('not sure what the error is')
        cv2.imwrite('problem_image'+str(problem_image_counter)+'.jpg',img)
        problem_image_counter +=1

    transform_matrix, invert_matrix = get_perspective_transform_matrix(undistort, hull)
    # transform perspective on filtered binary image
    warped = cv2.warpPerspective(to_RGB(binary_img), transform_matrix, (binary_img.shape[1], binary_img.shape[0]),
                                 flags=cv2.INTER_LINEAR)
    centroids = find_window_centroids(warped, window_width, window_height, margin)

    bounded_lane = bound_lanes(warped[:, :, 0], centroids)
    framed_lane = cv2.addWeighted(undistort, 1,
                                  cv2.warpPerspective(to_RGB(bounded_lane),
                                                      invert_matrix,
                                                      (binary_img.shape[1], binary_img.shape[0]),
                                                      flags=cv2.INTER_LINEAR),
                                  0.9, 0)
    """
    below are lower-level functions
    """
    #see all lines before filters

    region_interest = region_of_interest(grad_mag, [roi])
    lines = cv2.HoughLinesP(region_interest, 1, math.pi / 180, 30, np.array([]),
                            minLineLength=40, maxLineGap=50)
    all_hough_lines_img = np.zeros_like(undistort)
    draw_lines(all_hough_lines_img, lines, thickness=10)

    # test filters of variable perspective
    best_lines = find_filtered_candidate_lines(undistort)
    best_lines_accum = []
    for segment, lines in best_lines.items():
        best_lines_accum.append(lines)
    best_lines_accum = np.int32(np.concatenate(best_lines_accum).reshape((-1, 1, 4)))
    # convergence filter
    convergent_lines = extrapolate_to_vanishing(best_lines_accum, y_region=(350, 450),
                                                threshold=10, x_region=(550, 750))
    # resulting lines to test by projecting outwards
    projected_lines = collect_filter_projections(img, best_lines_accum, np.array([400]))
    filtered_projected_lines = collect_filter_projections(undistort, best_lines_accum[convergent_lines],
                                                          np.array([400]))
    filter_alone_img = np.zeros_like(undistort)
    draw_lines(filter_alone_img, np.int32(projected_lines).reshape((-1, 1, 4)), thickness=20)
    filtered_line_image = cv2.addWeighted(np.uint8(undistort), 1, to_RGB(filter_alone_img), 0.9,0)

    filter_projected_img = np.zeros_like(undistort)
    draw_lines(filter_projected_img, np.int32(filtered_projected_lines).reshape(-1, 1, 4), thickness=20)
    convergence_line_image = cv2.addWeighted(np.uint8(undistort), 1, to_RGB(filter_projected_img), 0.9, 0)

    #finding centroids
    convolution = draw_window_centroids(warped[:, :, 0], centroids)

    #curverad and offset

    left_lane_trans, right_lane_trans = transform_lane(centroids, invert_matrix)
    offset = distance_from_center_lane(left_lane_trans, right_lane_trans, undistort)

    curverad = radius_curvature(img, centroids)


    processing_steps = {
        'diag1': undistort,
        'diag2': binary_img,
        'diag3': hls_colors[:, :, 0],
        'diag4': hls_colors[:, :, 2],
        'diag5': grad_mag,
        'diag6': theta_grad,
        'diag7': warped,
        'diag8': convolution,
        'diag9': all_hough_lines_img,
        'diag10': filtered_line_image,
        'diag11': convergence_line_image,
        'diag12': hull_lines_img
    }

    return curverad, offset, framed_lane, processing_steps


def compose_diag_screen(curverad=0, offset=0, main_diag=None,
                        diag1=None, diag2=None, diag3=None, diag4=None,
                        diag5=None, diag6=None, diag7=None, diag8=None,
                        diag9=None, diag10=None, diag11=None, diag12=None):
    #  middle panel text example
    #  using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(curverad), (30, 60), font, 1, (255, 0, 0), 2)
    cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(offset), (30, 90), font, 1, (255, 0, 0), 2)

    # frame that contains all altered images
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # contains slices which describe placement within screen
    # first is x_slice, second y_slice
    # both slices define the shape of the image to be placed
    placement_diag = [(slice(0, 720), slice(0, 1280), main_diag),
                      (slice(0, 240), slice(1280, 1600), diag1),
                      (slice(0, 240), slice(1600, 1920), diag2),
                      (slice(240, 480), slice(1280, 1600), diag3),
                      (slice(240, 480), slice(1600, 1920), diag4),

                      (slice(600, 840), slice(1280, 1600), diag5),
                      (slice(600, 840), slice(1600, 1920), diag6),
                      (slice(840, 1080), slice(1280, 1600), diag7),

                      (slice(840, 1080), slice(1600, 1920), diag8),

                      (slice(720, 840), slice(0, 1280), middlepanel),
                      (slice(840, 1080), slice(0, 320), diag9),
                      (slice(840, 1080), slice(320, 640), diag10),
                      (slice(840, 1080), slice(640, 960), diag11),
                      (slice(840, 1080), slice(960, 1280), diag12)]

    # place all diags within frame
    for x_slice, y_slice, diag in placement_diag:
        if diag is not None:
            insert_diag_into(frame, diag, x_slice, y_slice)

    return frame

def run_compose_diag_screen(img):
    curverad, offset, framed_lane, processing_steps = testing_pipeline(img)
    result = compose_diag_screen(curverad, offset, framed_lane, **processing_steps)
    return result

if __name__ == '__main__':
    """
    img = cv2.imread('test_images/test1.jpg')
    curverad, offset, framed_lane, processing_steps = testing_pipeline(img)
    cv2.namedWindow('test')
    cv2.imshow('test', processing_steps['diag3'])
    img = compose_diag_screen(curverad, offset, framed_lane, **processing_steps)
    cv2.namedWindow('pipeline', cv2.WINDOW_NORMAL)
    cv2.imshow('pipeline', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    from moviepy.editor import VideoFileClip

    output_path = 'test_project.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(run_compose_diag_screen)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)