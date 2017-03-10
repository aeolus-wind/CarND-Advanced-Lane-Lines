import cv2
import numpy as np
from distortion import RemoveDistortion
from find_lanelines import find_window_centroids, bound_lanes, draw_window_centroids
from find_lanelines import window_width, window_height, margin  # some constants to be used by the algorithm
from find_lanelines import distance_from_center_lane, radius_curvature
from normalize_process_images import to_RGB
from colors import hls_decision_rule, bit_and_transform

from Line import Line

left_line = Line(5, tolerance=100)
right_line = Line(5, tolerance=100)
counter = 0

def insert_diag_into(frame, diag, x_slice, y_slice):
    # should take in upper left and lower right pixel location
    x_shape = x_slice.stop - x_slice.start
    y_shape = y_slice.stop - y_slice.start
    frame[x_slice, y_slice] = cv2.resize(to_RGB(diag), (y_shape, x_shape), interpolation=cv2.INTER_AREA)

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

def pipeline(img):
    global left_line
    global right_line
    global counter
    rmv_distortion = RemoveDistortion()
    rmv_distortion.load_pickle()
    undistort = rmv_distortion.undistort(img)
    # transform into bird-eye perspective
    src = np.array([390, 600, 940, 590, 560, 480, 745, 470], np.float32).reshape((4, 2))
    dst = np.array([455, 650, 870, 670, 460, 350, 905, 360], np.float32).reshape((4, 2))

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    boolean = np.logical_or(bit_and_transform(undistort), hls_decision_rule(undistort))
    warped = cv2.warpPerspective(to_RGB(boolean), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)[: ,:, 0]

    # remove irrelevant regions
    warped[-200:, :370] = False  # lower left
    warped[:550, :300] = False  # top left
    warped[-300:, -300:] = False  # lower right
    warped[:450, -300:] = False  # top right
    warped[-20:, :] = False  # remove bottom 20 pixels

    # find centroids and transform back
    if (left_line.current_fit is None and right_line.current_fit is None) or counter > 5:
        centroids = find_window_centroids(warped, window_width, window_height, margin)
        counter = 0
    else:
        try:
            centroids = Line.local_search(warped, left_line.centroid, right_line.centroid, left_line.extrap.predict_xfitted(), right_line.extrap.predict_xfitted())
            counter += 1
        except ValueError:
            print('exception used')
            centroids = find_window_centroids(warped, window_width, window_height, margin)
            counter = 0


    # add in new values
    left_centroid = centroids[:, 0]
    right_centroid = centroids[:, 1]

    # update dependent values
    offset = distance_from_center_lane(centroids, img, Minv)  # initial attempt
    left_line.next_fit(left_centroid, right_centroid, offset)
    right_line.next_fit(right_centroid, left_centroid, offset)
    # Fit a second order polynomial to each

    bounded = bound_lanes(np.uint8(warped), left_line.get_best_fit(), right_line.get_best_fit())

    bounds_img = cv2.warpPerspective(bounded, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    bounded_lanes = cv2.addWeighted(np.uint8(img), 1, to_RGB(bounds_img), 0.8, 0)

    # find radius curvature and offset
    left_curverad, right_curverad = radius_curvature(img, centroids)
    left_line.set_curverad(left_curverad)
    right_line.set_curverad(right_curverad)
    offset = distance_from_center_lane(centroids, img, Minv)

    #drawing the convolutional process for diagnostic purposes
    convolutional_process = draw_window_centroids(np.float32(warped), centroids)

    return bounded_lanes, warped, convolutional_process, (left_curverad, right_curverad), offset

def testing_pipeline(img):
    framed_lane, bit, convolutional,curverad, offset = pipeline(img)
    hls_colors = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    processing_steps = {
        'diag1': bit,
        'diag2': convolutional,
        #'diag3': hls_colors[:, :, 0],
        #'diag4': hls_colors[:, :, 2],
        #'diag5': img[:,:,0],
        #'diag6': img[:,:,1],
        #'diag7': img[:,:,2],
        #'diag8': convolution,
        #'diag9': all_hough_lines_img,
        #'diag10': filtered_line_image,
        #'diag11': convergence_line_image,
        #'diag12': hull_lines_img
    }
    return (curverad[0]+curverad[1])/2, offset, framed_lane, processing_steps




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