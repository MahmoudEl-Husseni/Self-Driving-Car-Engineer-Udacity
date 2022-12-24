import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from config import *

from tqdm import tqdm

#=============================================#

driving_log = pd.read_csv(DRIVING_LOG_PATH)

def augment(img, label):
   '''
   augment:    it flip both image and steering angle. 

   Args: 
      img:     image in non-dimensional array representation.
      label:   steering angle of this image.

   Return: 
      flipped img, label. 
   '''
   image_flipped = np.fliplr(img)
   label *= -1
   return image_flipped, label

def preprocess(images : np.ndarray, angles : np.ndarray, augmentation_p : float =0.0):
   '''
   Preprocess batch of images -> scale images in range (-128 - 128), crop ROI only, augment some images

   Args: 
      images:           image in non-dimensional array representation.
      angles:           steering angle attached to each image.
      augmentation_p:   percentage os augmented images in total images.

   Returns: 
      images, angles after preprocessing and augmentation.
   '''

   n_images = len(images)

   images = np.float32(images)
   # scaling
   images /= 255.0
   images -= 0.5

   #cropping
   images = images[:, 50:-20, :, :]

   augmented_images_indices = np.random.choice(a=[0, 1], size=n_images, replace=True, p=[1-augmentation_p, augmentation_p])
   n_augmented = np.sum(augmented_images_indices)

   for i in range(n_augmented):
      print(i)
      images[i], angles[i] = augment(images[i], angles[i])

   images, angles = shuffle(images, angles, random_state=101)
   return images, angles



def load_from_pickle():

   if not os.path.isfile('data/images.pkl') and not os.path.isfile('data/steering.pkl'):
      images = []
      steering = []

      for i in tqdm(range(len(driving_log))):
         im_path = driving_log.iloc[i, 0].split('/')[-1]
         im_path = os.path.join("data/IMG", im_path)
         img = cv.imread(im_path)
         img = preprocess(img)
         images.append(img)
         steering.append(driving_log.iloc[i, 3])

      with open('data/images.pkl', 'wb') as f:
         pickle.dump(images, f)

      with open('data/steering.pkl', 'wb') as f:
         pickle.dump(steering, f)

   else :
      with open('data/images.pkl', 'rb') as f:
         images = pickle.load(f)
      with open('data/steering.pkl', 'rb') as f:
         steering = pickle.load(f)

   return images, steering


## Create Data Generator
class Data_Generator:

   def __init__(self, batch_size=BATCH_SIZE, samples=driving_log, image_shape=(90, 320, 3)):
      
      self.batch_size = batch_size
      self.samples = samples
      self.n_samples = len(samples)
      self.steps_per_epoch = self.n_samples // batch_size
      self.image_shape = image_shape
       

   def data_generator(self, augmentation_p : int = 0.0):
      shuffle(self.samples)
      while True:

         for batch in range(0, len(self.samples), self.batch_size):
            records = self.samples.iloc[batch:batch+self.batch_size]
            images = records.iloc[:, 0]
            angles = records.iloc[:, 3]

            images = np.array([*map(lambda x : cv.imread('data/IMG/' + x.split('/')[-1]), images)])
            images, angles = preprocess(images, angles, augmentation_p)
            
            yield images, angles.values