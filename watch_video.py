import numpy as np
import cv2

def normalized(img):
    return np.uint8(255.0*np.float32(img)/np.max(np.absolute(img)))


def to_RGB(img):
    if img.ndim == 2:
        img_normalized = normalized(img)
        return np.dstack((img_normalized, img_normalized, img_normalized))
    elif img.ndim == 3:
        return img
    else:
        return None


if __name__ == '__main__':
    img = cv2.imread('test_images/test1.jpg')
    img = compose_diag_screen(img)
    cv2.namedWindow('pipeline', cv2.WINDOW_NORMAL)
    cv2.imshow('pipeline', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()