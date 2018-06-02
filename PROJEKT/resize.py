import glob
import os
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import cv2

# nvidia-smi
# tensorboard --logdir=Graph
batch_size = 128
num_classes = 5
epochs = 20
img_rows = 128
img_cols = 128
histogram_freq = 0


def image_to_feature_vector(image, size=(img_rows, img_cols)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size)


# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = []
imagePaths += glob.glob('./data/daisy/*.jpg')
imagePaths += glob.glob('./data/sunflower/*.jpg')
imagePaths += glob.glob('./data/rose/*.jpg')
imagePaths += glob.glob('./data/tulip/*.jpg')
imagePaths += glob.glob('./data/dandelion/*.jpg')

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    features = image_to_feature_vector(image)
    cv2.imwrite(imagePath,features)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
