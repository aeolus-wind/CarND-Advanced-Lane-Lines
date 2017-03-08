import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from distortion import RemoveDistortion
from perspective import default_transform_perspective
from perspective import default_invert_perspective
from normalize_process_images import to_RGB
from colors import or_decision_rule
from variable_perspective import filter_lines_find_hull, get_perspective_transform_matrix


# Read in a thresholded image
# window settings
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 100  # How much to slide left and right for searching


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(img, window_width, window_height, margin, tolerance=5, initial_point=None):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    if len(img.shape)==3:
        img = img[:,:,0] ## must be a binary image, so pick out one 'copy'
    l_sum = np.sum(img[int(3 * img.shape[0] / 4):, :int(img.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(img[int(3 * img.shape[0] / 4):, int(img.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(img.shape[1] / 2)
    if initial_point is not None:
        if abs((l_center - initial_point[0])/initial_point[0])>0.1:
            l_center = initial_point[0]
        if abs((r_center-initial_point[1]) / initial_point[1])>0.1:
            r_center = initial_point[1]

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(img.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            img[int(img.shape[0] - (level + 1) * window_height):int(img.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))
    #hacky method of determining unusual change amount using something like derivatives
    window_centroids = np.array(window_centroids)
    left_first_deriv = window_centroids[:,0][1:] - window_centroids[:,0][:-1]
    left_second_deriv = left_first_deriv[1:] - left_first_deriv[:-1]
    right_first_deriv = window_centroids[:,1][1:] - window_centroids[:,1][:-1]
    right_second_deriv = right_first_deriv[1:] - right_first_deriv[:-1]

    # if there is a large second derivative at any point,
    # pick the well-behaved changes and tack them on the the poorly behaved
    if np.mean(np.abs(left_second_deriv)) - np.mean(np.abs(right_second_deriv)) \
            >= tolerance: #the lines should be parallel

        if np.mean(np.abs(left_second_deriv)) < np.mean(np.abs(right_second_deriv)):
            additive_derivatives = np.hstack([0, np.cumsum(left_first_deriv)])
            initial_value = window_centroids[0][1]
            window_centroids[:, 1] = initial_value + additive_derivatives
        else:
            additive_derivatives = np.hstack([0, np.cumsum(right_first_deriv)])
            initial_value = window_centroids[0][0]
            window_centroids[:, 0] = initial_value + additive_derivatives
    return window_centroids


def draw_window_centroids(img, window_centroids):
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channle
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(to_RGB(img),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((img, img, img)), np.uint8)

    # Display the final results
    #plt.imshow(output)
    #plt.title('window fitting results')
    #plt.show()
    return output


def radius_curvature(img, centroids, window_height=window_height):
    meters_per_pixel_x = 3.7 / 700
    meters_per_pixel_y = 30 / 720
    centroids = np.array(centroids)
    ys = range(img.shape[0], 0, -window_height)
    left_poly = np.polyfit(np.array(ys)*meters_per_pixel_y, centroids[:,0]*meters_per_pixel_x, 2)
    right_poly = np.polyfit(np.array(ys)*meters_per_pixel_y, centroids[:,1]*meters_per_pixel_x, 2)
    left_poly_curve = (1+(2*left_poly[0]*700*meters_per_pixel_y + left_poly[1])**2)**(3/2) / abs(2*left_poly[0])
    right_poly_curve = (1 + (2*right_poly[0]*700*meters_per_pixel_y + right_poly[1])**2)**(3/2) / abs(2*right_poly[0])
    return left_poly_curve, right_poly_curve


def bound_lanes(img, window_centroids, window_height=window_height):
    if len(img.shape) != 2:
        raise ValueError('image must be 2-d and binary!')
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    if not isinstance(window_centroids, np.ndarray):
        window_centroids = np.array(window_centroids)
    left = window_centroids[:, 0]
    right = window_centroids[:, 1]
    ys = range(720,0,-window_height)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(np.array(ys), left, 2)

    right_fit = np.polyfit(np.array(ys), right, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    highlight_margin = 30
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-highlight_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+highlight_margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-highlight_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+highlight_margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    lane_highlight_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    lane_highlight_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    lane_highlight_pts = np.hstack((lane_highlight_left, lane_highlight_right))

    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img, cmap='gray')
    ax1.set_title('original', fontsize=50)
    ax2.plot(left_line_pts[:,:,0],  left_line_pts[:,:,1], '.')
    ax2.set_title('contour', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    """

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_(left_line_pts), (0,255,0))
    cv2.fillPoly(window_img, np.int_(right_line_pts), (0,250,0))
    cv2.fillPoly(window_img, np.int_(lane_highlight_pts), (250,0,0))
    # To do: fill in the region
    return window_img


def transform_lane(centroids, matrix, window_height=window_height, from_transformed_space=True):
    """
    takes centroids and transforms them into points in the original picture
    see get_perspective_transform under
    http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    :param centroids:
    :return:
    """
    if not isinstance(centroids, np.ndarray):
        centroids = np.array(centroids)
    if from_transformed_space:
        left = np.column_stack((centroids[:, 0], range(720, 0, -window_height), np.ones(int(720/window_height))))
        right = np.column_stack((centroids[:, 1], range(720, 0, -window_height), np.ones(int(720/window_height))))
    else:
        left = np.column_stack((centroids[:,(0,1)], np.ones(int(720/window_height))))
        right = np.column_stack((centroids[:,(2,3)], np.ones(int(720/window_height))))

    left_transformed = matrix.dot(left.T)
    right_transformed = matrix.dot(right.T)
    return left_transformed[:2,:]/left_transformed[2,:], right_transformed[:2,:]/right_transformed[2,:]


def extrapolate_to_y_coordinate(ploty, left_transformed, right_transformed):
    """
    fits a polynomial to the 'real coordinates'  and finds the x
    where the polynomail occurs
    :param ploty:
    :param left_transformed:
    :param right_transformed:
    :return:
    """
    left_fit = np.polyfit(left_transformed[1], left_transformed[0], 2)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]

    right_fit = np.polyfit(right_transformed[1], right_transformed[0], 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx


def distance_from_center_lane(left_lane, right_lane, img):
    """
    :param left_lane: coordinates of left lane: this lane must be transformed using transform_lane
    otherwise the coordinates will be in the transformed space
    :param right_lane: same as left lane
    :param img: original image mainly for coordinates
    :return: distance from center of lanes assuming the lane is 700 pixels at the bottom and a lane is 3.7 meters
    """
    meters_per_pixel_x = 3.7 / 700

    left_lane_pixel, right_lane_pixel = extrapolate_to_y_coordinate(700, left_lane, right_lane)

    screen_middle_pixel = img.shape[1] / 2
    car_middle_pixel = int((right_lane_pixel + left_lane_pixel) / 2)
    pixels_off_center = screen_middle_pixel - car_middle_pixel
    meters_off_center = round(meters_per_pixel_x * pixels_off_center, 2)
    return meters_off_center


if __name__ == '__main__':
    original = False
    if original:
        """
        first procedure I tested-- to be deprecated
        """
        test = cv2.imread('test_images/straight_lines2.jpg')

        rmv_distortion = RemoveDistortion()
        rmv_distortion.load_pickle()
        undistort = rmv_distortion.undistort(test)

        binary_img = or_decision_rule(undistort)  # a more robust version using hlv color space will be added

        cv2.namedWindow('original img')
        cv2.imshow('original img', undistort)
        cv2.namedWindow('binary img')
        cv2.imshow('binary img', to_RGB(binary_img))
        cv2.namedWindow('binary img transform')
        cv2.imshow('binary img transform', default_transform_perspective(to_RGB(binary_img)))

        shifted_perspective = default_transform_perspective(to_RGB(binary_img))

        #plt.imshow(shifted_perspective)
        #plt.show()

        centroids = find_window_centroids(shifted_perspective, window_width,
                                          window_height, margin)  # find centroids using default
        #print(centroids)
        # draw_window_centroids(shifted_perspective[:,:,0], centroids)
        corrected_centroids = np.array([[ 358.,  972.],
                                        [ 359.,  973.],
                                        [ 360.,  942.],
                                        [ 366., 941.],
                                        [ 366.,  939.],
                                        [ 360.,  957.],
                                        [ 373.,  956.],
                                        [ 378.,  952.],
                                        [ 360.,  930.]])

        bounded_lane = bound_lanes(shifted_perspective[:,:,0], centroids)
        #cv2.imshow('img',default_invert_perspective(bounded_lane))
        transform = cv2.addWeighted(undistort, 1, default_invert_perspective(bounded_lane), 0.9, 0)
        cv2.imshow('transformed_image', transform)
        cv2.waitKey()
    else:
        curve_image = glob.glob('test_images/test*.jpg')
        straight_image = glob.glob('test_images/straight_lines*.jpg')

        for img_path in curve_image + straight_image:
            img = cv2.imread(img_path)
            rmv_distortion = RemoveDistortion()
            rmv_distortion.load_pickle()
            undistort = rmv_distortion.undistort(img)

            hull = filter_lines_find_hull(undistort)

            transform_matrix, invert_matrix = get_perspective_transform_matrix(undistort, hull)

            binary_img = or_decision_rule(undistort)

            warped = cv2.warpPerspective(to_RGB(binary_img), transform_matrix, (binary_img.shape[1], binary_img.shape[0]), flags=cv2.INTER_LINEAR)
            centroids = find_window_centroids(warped, window_width, window_height, margin)
            left_lane_trans, right_lane_trans = transform_lane(centroids,invert_matrix)
            output = extrapolate_to_y_coordinate(700,left_lane_trans, right_lane_trans)
            print(output)  # to be used in lane center calculation
            print(radius_curvature(img, centroids))

            bounded_lane = bound_lanes(warped[:,:,0], centroids)
            print_lane = cv2.addWeighted(undistort, 1,
                                         cv2.warpPerspective(to_RGB(bounded_lane),
                                                             invert_matrix,
                                                             (binary_img.shape[1], binary_img.shape[0]),
                                                             flags=cv2.INTER_LINEAR), 0.9, 0)
            first_two_steps = True
            if first_two_steps:
                cv2.namedWindow('original binary image')
                cv2.imshow('original binary image', to_RGB(binary_img))
                cv2.namedWindow('transformed_perspective')
                cv2.imshow('transformed_perspective', warped)
                cv2.namedWindow('bounded_lane')
                cv2.imshow('bounded_lane', bounded_lane)

            cv2.namedWindow('print lane')
            cv2.imshow('print lane', print_lane)
            cv2.waitKey()