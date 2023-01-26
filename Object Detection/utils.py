import cv2 as cv
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

from skimage.feature import hog

def data_look(car_list, notcar_list):
    '''
    : gets some information about data.
    '''
    data_dict = {}

    # 1. store length of car images
    data_dict["n_cars"] = len(car_list)

    # 2. store length of non-car images
    data_dict["n_notcars"] = len(notcar_list)

    # 3. store shape of car images
    test_im = cv.imread(car_list[0])
    data_dict["image_shape"] = test_im.shape

    # 4. store data type of car images
    data_dict["data_type"] = test_im.dtype

    return data_dict

def calculate_hog(image:np.ndarray, pix_per_cell=8):
    '''
    Calculate histogram of gradients for image.
    '''
    out = []

    image = np.float32(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    sx = cv.Sobel(image, ddepth=0, dx=1, dy=0)
    sy = cv.Sobel(image, cv.CV_64F, 0, 1)


    mag = np.sqrt(sx**2 + sy**2)
    angle = np.arctan2(sy, sx) * 180 / np.pi
    n_cellsx = image.shape[0]//pix_per_cell
    n_cellsy = image.shape[1]//pix_per_cell

    for cellx in range(n_cellsx):
        for celly in range(n_cellsy):
            cell_hist = np.zeros(9)
            for i in range(pix_per_cell):
                for j in range(pix_per_cell):
                    ang = angle[cellx*pix_per_cell+i, celly*pix_per_cell+j]
                    # print(ang)
                    mag_val = mag[cellx*pix_per_cell+i, celly*pix_per_cell+j]
                    mag_right = mag_val * (1 - (ang % 20)/20)
                    mag_left = mag_val * (ang % 20)/20
                    cell_hist[int(ang//20)] += mag_left
                    cell_hist[(int(ang//20)+1)%9] += mag_right
            out.append(cell_hist)

    return np.array(out), angle


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=True,
                        visualize=vis, feature_vector=feature_vec)
        return features
    
def bin_spatial(img, size=(32, 32)):
    '''
    resize image to fixed size (feature extractor depends on color of pixel)
    '''
    features = cv.resize(img, size).ravel()
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    return histogram of channel (ex. Reds, Greens, Blues) in image.
    '''
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, 
                        orient=9, pix_per_cell=8, cell_per_block=2):
        
    '''
    Extract Features from image: 
        1. converts image to desired color space.
        extract features: 
        a) color depending features(bin spatial, channel histogram).    
        b) edeges depending features (Histogram of Gradients). 
    '''
    if isinstance(imgs[0], str):
        imgs = [cv.imread(img) for img in imgs]
    features = []

    for image in imgs:
        file_features = []
        # image = np.float32(image)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv.cvtColor(image, cv.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)

        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

        hog_features = []
        for channel in range(feature_image.shape[2]):

            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    return np.array(features)

def slide_window(img, x_start_end=[None, None], y_start_end=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    get bounding boxes of different scale windows. bbox = ((x0, y0), (x1, y1))
    '''
    if x_start_end[0] == None:
        x_start_end = [0, img.shape[1]]
    if y_start_end[0] == None:
        y_start_end = [0, img.shape[0]]

    window = []
    for y in range(y_start_end[0], y_start_end[1], int(xy_window[1]*(1-xy_overlap[1]))):
        for x in range(x_start_end[0], x_start_end[1], int(xy_window[0]*(1-xy_overlap[0]))):
            window.append(((x, y), (int(x+xy_window[0]), int(y+xy_window[1]))))
    return window

def draw_boxes(img, bboxes, colors=[(0, 0, 255)], thick=6):
    '''
    draw boudning boxes.
    '''
    imcopy = np.copy(img)
    for i, scale in enumerate(bboxes):
        for bbox in scale:
            cv.rectangle(imcopy, bbox[0], bbox[1], colors[i], thick)
    return imcopy

def search_windows(img, windows, clf, scaler, cspace='RGB', spatial_size=(32, 32), hist_bins=32,
                    orient=9, pix_per_cell=8, cell_per_block=2):
    '''
    slide at each window and: 
        extract features for every window, classify whether its car or not.
    '''
    on_windows = []
    for scale in windows:
        for window in scale:
            try:
                test_img = cv.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            except:
                continue
            features = extract_features([test_img], cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
            
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict_proba(test_features)
            if prediction[0][1] >= 0.9:
                on_windows.append(window)
    return on_windows
    
def heat_map(img, on_windows, intensity=1):
    '''
    to find overlapping boxes.
    '''
    heat = np.zeros_like(img)
    for window in on_windows:
            heat[window[0][1]:window[1][1], window[0][0]:window[1][0]] += intensity
    
    return heat

def threshold_img(img, threshold):
    '''
    threshold image.
    '''
    out = np.ones_like(img)*255
    out[img<threshold] = 0
    return out


def draw_labeled_bboxes(img, labeled_bboxes, labels):
    '''
    draw boxes around posistive classified cars.
    '''
    # Iterate through all detected cars
    for car_number in range(1, labels+1):
        # Find pixels with each car_number label value
        nonzero = (labeled_bboxes == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# ==============================================================

if __name__=="__main__":
    car_list = glob('vehicles_smallset/*/*.jpeg')
    notcar_list = glob('non-vehicles-smallest/*/*.jpeg')

    data_info = data_look(car_list, notcar_list)
    print('Your function returned a count of',
            data_info["n_cars"], ' cars and',
            data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:',
            data_info["data_type"])
    
    car_ind = np.random.randint(0, len(car_list))
    img = cv.imread(car_list[car_ind])
    for channel in range(img.shape[2]):
        features, hog_image = get_hog_features(img[:,:,channel], orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=True)
        plt.subplot(1,3,channel+1)
        plt.imshow(hog_image, cmap='gray')
    plt.show()    

    features = extract_features([car_list[car_ind]], cspace='RGB', spatial_size=(32, 32), hist_bins=32,
                        orient=9, pix_per_cell=8, cell_per_block=2, normalize=True)