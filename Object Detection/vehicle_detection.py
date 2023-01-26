from utils import * 
from CONFIG.config import *

import pickle
import cv2 as cv
import numpy as np
from scipy.ndimage.measurements import label


class VehicleDetection: 

    def __init__(self, clf, scaler, VEHICLEDETECTION):
        self.clf = clf
        self.scaler = scaler

        self.ystart = VEHICLEDETECTION['ystart']
        self.yend = VEHICLEDETECTION['yend']
        self.orient = VEHICLEDETECTION['orient'] 
        self.pix_per_cell = VEHICLEDETECTION['pix_per_cell'] 
        self.cell_per_block = VEHICLEDETECTION['cell_per_block'] 
        self.spatial_size = VEHICLEDETECTION['spatial_size'] 
        self.hist_bins = VEHICLEDETECTION['hist_bins']
        self.thresh = VEHICLE_DETECTION['thresh']

    def find_cars(self, img):
        painting = img.copy()
        ROI = img[self.ystart:self.yend, :]

        hog = get_hog_features(ROI, self.orient, self.pix_per_cell, self.cell_per_block, 
                               feature_vec=False)
        
    def find_cars(self, img):
        

        roi = img[self.ystart:self.yend,:,:]
        roi = cv.cvtColor(roi, cv.COLOR_RGB2YCrCb)
            
        ch1 = roi[:,:,0]
        ch2 = roi[:,:,1]
        ch3 = roi[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        box_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv.resize(roi[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.clf.predict_proba(test_features)
                if test_prediction[0][1] >= self.thresh:
                    xbox_left = np.int(xleft)
                    ytop_draw = np.int(ytop)
                    win_draw = np.int(window)
                    
                    box_list.append([(xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart)])
                    
        return box_list
    
    def draw_labeled_bboxes(self, img, box_list):
        painting = img.copy()
        heat = heat_map(img, box_list, INTENSITY)
        heat = threshold_img(heat, INTENSITY)
        labeled_bboxes, labels = label(heat)
        draw_img = draw_labeled_bboxes(painting, labeled_bboxes, labels)
        return draw_img

with open(MODEL_PKL, 'rb') as f:
    clf = pickle.load(f)

with open(STD_SCALER_PKL, 'rb') as f:
    scaler = pickle.load(f)

vec_det = VehicleDetection(clf, scaler, VEHICLE_DETECTION)
cap = cv.VideoCapture(TEST_VIDEOS_DIR + '/test_video.mp4')

while(cv.waitKey(1) != 27):

    succ, img = cap.read()
    bboxes  = vec_det.find_cars(img)
    painting = vec_det.draw_labeled_bboxes(img, bboxes)
    cv.imshow("out", painting)

cap.release()
cv.destroyAllWindows()