
from Segmentation import img_segmentation
import cv2 as cv
import numpy as np
import argparse
import os

title_window = 'Adjust HSV range!'
img_paths = []
h_min = -1
h_max = -1
s_min = -1
v_min = -1
im_ind = 0
mouse_x = 0
mouse_y = 0

def check_image(im, im_path):
    if im is None:
        print('Error opening the image {}'.format(im_path))
        exit()


def trackbar_callback_im(im_ind_new):
    global im_ind
    im_ind = im_ind_new
    im_path = img_paths[im_ind]
    im = cv.imread(im_path, cv.IMREAD_COLOR)
    check_image(im, im_path) # check if image was sucessfully read
    im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    # Do segmentation with current values
    mask_gel = img_segmentation.get_gel_hsv(im_hsv, h_min, h_max, s_min, v_min)
    if mask_gel is not None:
        im_copy = im.copy()
        # Highlight green part in red
        alpha = 0.7
        mask_gel_red = np.zeros_like(im_copy)
        mask_gel_red[:,:,2] = mask_gel
        im_copy = cv.addWeighted(im_copy, 1.0, mask_gel_red, alpha, 0)
        # Draw current HSV value
        pt_hsv = im_hsv[mouse_y, mouse_x].tolist()
        text = "HSV = [{}, {}, {}]".format(pt_hsv[0], pt_hsv[1], pt_hsv[2])
        cv.putText(im_copy, text, (mouse_x, mouse_y), 0, 1, (0, 0, 0), 5) # Black text border
        cv.putText(im_copy, text, (mouse_x, mouse_y), 0, 1, im[mouse_y, mouse_x].tolist(), 2)
        cv.imshow(title_window, im_copy)


def trackbar_callback_h_min(h_min_new):
    global h_min
    h_min = h_min_new
    trackbar_callback_im(im_ind)


def trackbar_callback_h_max(h_max_new):
    global h_max
    h_max = h_max_new
    trackbar_callback_im(im_ind)


def trackbar_callback_s_min(s_min_new):
    global s_min
    s_min = s_min_new
    trackbar_callback_im(im_ind)


def trackbar_callback_v_min(v_min_new):
    global v_min
    v_min = v_min_new
    trackbar_callback_im(im_ind)


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y
    trackbar_callback_im(im_ind)


def improve_segmentation(config):
    global h_min, h_max, s_min, v_min
    global img_paths
    # Initialize values

    img_paths = [os.path.join(config.image_dir, f) for f in  os.listdir(config.image_dir)]
    h_min = config.h_min
    h_max = config.h_max
    s_min = config.s_min
    v_min = config.v_min

    print('All the images will share the same HSV range.')
    print('Press any [key] when finished.')

    cv.namedWindow(title_window, cv.WINDOW_NORMAL)
    cv.createTrackbar("Image", title_window , 0, len(img_paths) - 1, trackbar_callback_im)
    cv.createTrackbar("h_min", title_window , h_min, 180, trackbar_callback_h_min)
    cv.createTrackbar("h_max", title_window , h_max, 180, trackbar_callback_h_max)
    cv.createTrackbar("s_min", title_window , s_min, 255, trackbar_callback_s_min)
    cv.createTrackbar("v_min", title_window , v_min, 255, trackbar_callback_v_min)
    if len(img_paths) > 0:
        trackbar_callback_im(0)
        cv.setMouseCallback(title_window, mouse_callback)
        cv.waitKey()
        print('\nDone! Please modify these values in the `config.yaml` file:')
        print('h_min: {}'.format(h_min))
        print('h_max: {}'.format(h_max))
        print('s_min: {}'.format(s_min))
        print('v_min: {}'.format(v_min))
    else:
        print('ERROR: No images found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='D:/Bioprinting/data/sgmentation_Test')
    parser.add_argument('--h_min', type=int, default=0)
    parser.add_argument('--h_max', type=int, default=15)
    parser.add_argument('--s_min', type=int, default=212)
    parser.add_argument('--v_min', type=int, default=79)

    config = parser.parse_args()

    improve_segmentation(config)
