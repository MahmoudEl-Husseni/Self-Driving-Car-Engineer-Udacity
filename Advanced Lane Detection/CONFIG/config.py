import pickle
import cv2 as cv
import numpy as np

nx, ny = 9, 6
IMAGE_SHAPE = (720, 1280)

SAVING_DIR = 'CONFIG'
TEST_IMAGES_DIR = "test_images"
TEST_VIDEOS_DIR = "test_videos"
IMAGES_DIR = "calibration images"
OUTPUT_VIDEOS_DIR = "output_videos"


# Undistortion
with open("CONFIG/calibration.p", "rb") as f:
    mtx = pickle.load(f)

with open("CONFIG/distortion.p", "rb") as f:
    dist = pickle.load(f)


# Prespective Transform
prespective_src = np.float32([[20, 704], [480, 450], [750, 470], [1000, 716]])
prespective_dst = np.float32([[0, 720], [0, 0], [1280, 0], [720, 1280]])

M = cv.getPerspectiveTransform(prespective_src, prespective_dst)
Minv = cv.getPerspectiveTransform(prespective_dst, prespective_src)

# Color Space Parameters
S_THRESH = 0, 300
L_LAB_THRESH = 0, 220
B_LAB_THRESH = 50, 155
LUV_THRESH = 20, 157 

# Windowing parameters
WINDOWING_LINED_DETECTION = {
    'nwindows': 12,
    'margin': 100,
    'minpix': 50, 
}

PREVIOUS_FIT = {
    'margin': 80
}