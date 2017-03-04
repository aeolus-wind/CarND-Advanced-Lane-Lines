import cv2
import numpy as np
from test_pipeline import to_RGB
from sobel_experiment import bit_and_transform

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


def xor_decision_rule(img):
    # I am sceptical this works in all cases
    # will experiment with a linear combination threshold rule
    return np.logical_xor(hls_decision_rule(img), bit_and_transform(img))




def click_avg_color(event, x, y, flags, params):
    global bounding_box

    if event == cv2.EVENT_LBUTTONDOWN:
        bounding_box = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        bounding_box.append((x,y))



if __name__ == '__main__':
    img = cv2.imread('test_images/test5.jpg')
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    analyze_colors = False
    test_lane_colors = False
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
        cv2.imshow('combined and sobel', to_RGB(np.bitwise_xor(hls_decision_rule(img), bit_and_transform(img))))

        cv2.waitKey()


