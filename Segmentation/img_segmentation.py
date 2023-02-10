import numpy as np
import cv2 as cv
from utils import get_largest_one_component
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt


def get_hsv_lower_and_upper(h_min, h_max, s_min, s_max, v_min, v_max):
    if h_min < 0 or h_max > 180: # TODO: check if it is inclusive or not
        print("Error: h should be between 0 and 180")
        exit()
    if s_min < 0 or s_max > 255:
        print("Error: s should be between 0 and 255")
        exit()
    if v_min < 0 or v_max > 255:
        print("Error: v should be between 0 and 255")
        exit()
    lower = [[h_min, s_min, v_min]]
    upper = [[h_max, s_max, v_max]]
    return np.array(lower, np.uint8), np.array(upper, np.uint8)

def get_gel_hsv(im_hsv, h_min, h_max, s_min, v_min):
    h_min = h_min
    h_max = h_max
    s_min = s_min
    s_max = 255
    v_min = v_min
    v_max = 255
    # Get lower and upper values
    lower, upper = get_hsv_lower_and_upper(h_min, h_max, s_min, s_max, v_min, v_max)
    mask_gel_colour = cv.inRange(im_hsv, lower, upper)
    return mask_gel_colour

    # Erode mask (remove noise, not sure if needed)
    #kernel = np.ones((3, 3), np.uint8)
    #mask_bg_colour = cv.erode(mask_bg_colour, kernel, iterations = 1)

    ## Get largest contour (to avoid returning noise as a potential marker)
    # contours, _hierarchy = cv.findContours(mask_bg_colour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # if len(contours) == 0: # No marker detected
    #     return None, None
    # c = max(contours, key = cv.contourArea)
    # mask_marker_bg = np.zeros(mask_bg_colour.shape, np.uint8)
    # cv.drawContours(mask_marker_bg, [c], -1, 255, -1)
    # marker_area = cv.contourArea(c)

    # # Erode mask (given that we already have the biggest green contour)
    # kernel = np.ones((3, 3), np.uint8)
    # mask_marker_bg = cv.erode(mask_marker_bg, kernel, iterations = 3)

    # return mask_marker_bg, marker_area