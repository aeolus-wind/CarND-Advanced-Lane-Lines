import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from distortion import RemoveDistortion
from sobel_experiment import bit_and_transform
from perspective import default_transform_perspective
from perspective import default_invert_perspective
from test_pipeline import to_RGB
from colors import xor_decision_rule

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


def find_window_centroids(img, window_width, window_height, margin):
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
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()


def bound_lanes(img, window_centroids):
    if len(img.shape) != 2:
        raise ValueError('image must be 2-d and binary!')
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    window_centroids = np.array(window_centroids)
    left = window_centroids[:, 0]
    right = window_centroids[:, 1]
    ys = range(720,0,-80)
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


if __name__ == '__main__':
    test = cv2.imread('test_images/straight_lines2.jpg')

    rmv_distortion = RemoveDistortion()
    rmv_distortion.load_pickle()
    undistort = rmv_distortion.undistort(test)
    binary_img = xor_decision_rule(undistort)  # a more robust version using hlv color space will be added

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