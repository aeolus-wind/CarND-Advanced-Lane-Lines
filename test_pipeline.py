import cv2
import numpy as np

def normalized(img):
    return np.uint8(255*img/np.max(np.absolute(img)))


def to_RGB(img):
    if img.ndim == 2:
        img_normalized = normalized(img)
        return np.dstack((img_normalized, img_normalized, img_normalized))
    elif img.ndim == 3:
        return img
    else:
        return None


def insert_diag_into(frame, diag, x_slice, y_slice):
    # should take in upper left and lower right pixel location
    x_shape = x_slice.stop - x_slice.start
    y_shape = y_slice.stop - y_slice.start
    frame[x_slice, y_slice] = cv2.resize(to_RGB(diag), (y_shape, x_shape), interpolation=cv2.INTER_AREA)


def compose_diag_screen(curverad=0, offset=0, main_diag=None,
                        diag1=None, diag2=None, diag3=None, diag4=None,
                        diag5=None, diag6=None, diag7=None, diag8=None, diag9=None):
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
                      (slice(600, 1080), slice(1280, 1600), diag7),
                      (slice(720, 840), slice(0, 1280), middlepanel),
                      (slice(840, 1080), slice(0, 320), diag5),
                      (slice(840, 1080), slice(320, 640), diag6),
                      (slice(840, 1080), slice(640, 960), diag9),
                      (slice(840, 1080), slice(960, 1280), diag8)]

    # place all diags within frame
    for x_slice, y_slice, diag in placement_diag:
        if diag is not None:
            insert_diag_into(frame, diag, x_slice, y_slice)
    """

    if main_diag is not None:
        frame[0:720, 0:1280] = main_diag
    if diag1 is not None:
        frame[0:240, 1280:1600] = cv2.resize(to_RGB(diag1), (320,240), interpolation=cv2.INTER_AREA)
    if diag2 is not None:
        frame[0:240, 1600:1920] = cv2.resize(to_RGB(diag2), (320,240), interpolation=cv2.INTER_AREA)
    if diag3 is not None:
        frame[240:480, 1280:1600] = cv2.resize(to_RGB(diag3), (320,240), interpolation=cv2.INTER_AREA)
    if diag4 is not None:
        frame[240:480, 1600:1920] = cv2.resize(to_RGB(diag4), (320,240), interpolation=cv2.INTER_AREA)*4
    if diag7 is not None:
        frame[600:1080, 1280:1920] = cv2.resize(to_RGB(diag7), (640,480), interpolation=cv2.INTER_AREA)*4
    frame[720:840, 0:1280] = middlepanel
    if diag5 is not None:
        frame[840:1080, 0:320] = cv2.resize(to_RGB(diag5), (320,240), interpolation=cv2.INTER_AREA)
    if diag6 is not None:
        frame[840:1080, 320:640] = cv2.resize(to_RGB(diag6), (320,240), interpolation=cv2.INTER_AREA)
    if diag9 is not None:
        frame[840:1080, 640:960] = cv2.resize(to_RGB(diag9), (320,240), interpolation=cv2.INTER_AREA)
    if diag8 is not None:
        frame[840:1080, 960:1280] = cv2.resize(to_RGB(diag8), (320,240), interpolation=cv2.INTER_AREA)
    """

    return frame

if __name__ == '__main__':
    img = compose_diag_screen()
    cv2.namedWindow('pipeline', cv2.WINDOW_NORMAL)
    cv2.imshow('pipeline', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()