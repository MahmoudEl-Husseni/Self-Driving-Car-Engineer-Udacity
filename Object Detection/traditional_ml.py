from utils import * 
from CONFIG.config import * 

import os 
import pickle
import cv2 as cv
import seaborn as sns
from glob import glob
from scipy.ndimage.measurements import label

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. load car and non-car images
car_images = glob(CAR_IAMGES_DIR + '/*/*.jpeg')
notcar_images = glob(NOT_CAR_IMAGES_DIR + '/*/*.jpeg')

# 2. extract features
if os.path.exists(CAR_FEATURES_PKL) and os.path.exists(NOT_CAR_FEATURES_PKL):
    with open(CAR_FEATURES_PKL, 'rb') as f:
        car_features = pickle.load(f)
    with open(NOT_CAR_FEATURES_PKL, 'rb') as f:
        notcar_features = pickle.load(f)
    
    with open(STD_SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)

else:
    car_features = extract_features(car_images, **EXTARCT_FEATURES)

    notcar_features = extract_features(notcar_images, **EXTARCT_FEATURES) 

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    with open(STD_SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)

    with open(CAR_FEATURES_PKL, 'wb') as f:
        pickle.dump(car_features, f)

    with open(NOT_CAR_FEATURES_PKL, 'wb') as f:
        pickle.dump(notcar_features, f)


# 3. train a classifier
X = np.vstack((car_features, notcar_features)).astype(np.float64)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
X, y = shuffle(X, y)

if os.path.exists(MODEL_PKL):
    with open(MODEL_PKL, 'rb') as f:
        model = pickle.load(f)
else:

    model = SVC(probability=True)
    model.fit(X, y)


    # 4. save the model
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model, f)

print("Model Loaded.")


# 5. test the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print('Accuracy: ', accuracy_score(y, y_pred) * 100)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
# evaluate_model(model, X, y)


# 6. search for cars
test_img_path = os.path.join(TEST_IMAGES_DIR, 'test3.jpg')
test_img = cv.imread(test_img_path)

windows = []

for scale in SCALES:
    windows.append(slide_window(test_img, x_start_end=[None, None], y_start_end=y_ROI_START_END,
                                                        xy_window=(scale, scale), xy_overlap=(0.5, 0.5)))


on_windows = search_windows(test_img, windows, model, scaler, **EXTARCT_FEATURES)

# 7. draw bounding boxes
draw = draw_boxes(test_img, [on_windows], thick=2)
plt.subplot(121)
plt.imshow(draw)

heat = heat_map(test_img, on_windows, INTENSITY)
thresholded_heat = threshold_img(heat, INTENSITY)
labeled_bboxes, labels = label(thresholded_heat)
print(f"Cars found: {labels}")

draw_img = draw_labeled_bboxes(test_img, labeled_bboxes, labels)
plt.subplot(122)
plt.imshow(draw_img, cmap='gray')
plt.show()