import pickle
import cv2 as cv
import numpy as np
from glob import glob

from CONFIG.config import *


images = glob(IMAGES_DIR + '/*.jpg')

chess_board_points = np.zeros((nx*ny, 3))
chess_board_points[:, :2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)

object_points = []
image_points = []

for idx, image in enumerate(images):
    image = cv.imread(image)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        image_points.append(corners)
        object_points.append(chess_board_points)

object_points = np.float32(object_points)
image_points = np.float32(image_points)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape, None, None)

# save calibration and distortion matrices
with open(SAVING_DIR + "/calibration.p", 'wb') as f:
    pickle.dump(mtx, f)

with open(SAVING_DIR + "/distortion.p", "wb") as f:
    pickle.dump(dist, f)
