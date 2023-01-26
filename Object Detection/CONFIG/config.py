# Paths
# Pickles
MODEL_PKL = "CONFIG/model.pkl"
STD_SCALER_PKL = "CONFIG/scaler.pkl"
CAR_FEATURES_PKL = "CONFIG/car_features.pkl"
NOT_CAR_FEATURES_PKL = "CONFIG/notcar_features.pkl"

# DIR
TEST_VIDEOS_DIR = "test_videos"
TEST_IMAGES_DIR = "test_images"
CAR_IAMGES_DIR = "vehicles_smallset"
NOT_CAR_IMAGES_DIR = "non-vehicles_smallset"

# ------------------------------------------------------
# Params
EXTARCT_FEATURES = {
    "orient" : 9,
    "hist_bins" : 32,
    "cspace" : 'YCrCb',
    "pix_per_cell" : 8,
    "cell_per_block" : 2,
    "spatial_size" : (32, 32),
}

VEHICLE_DETECTION = {
    "ystart" : 400,
    "yend" : 650,
    "scale" : [64, 96, 128],
    "orient": 9, 
    "pix_per_cell" : 8, 
    "cell_per_block" : 2, 
    "spatial_size" : (32, 32), 
    "hist_bins" : 32,
    "thresh" : 0.6
}
# -------------------------------------------------------

# CONSTANTS
INTENSITY = 10
SCALES = [64, 96, 128]
y_ROI_START_END = [400, 656]