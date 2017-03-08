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
