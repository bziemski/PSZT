import glob
import os
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model, np_utils
from keras import backend as K
import numpy as np
import cv2

# nvidia-smi
# tensorboard --logdir=Graph
batch_size = 128
num_classes = 2
epochs = 20
img_rows = 32
img_cols = 32
histogram_freq=0


def image_to_feature_vector(image, size=(img_rows, img_cols)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = glob.glob('./data/*.jpg')

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.25)

trainData = trainData.reshape(trainData.shape[0], img_rows, img_cols, 3)
testData = testData.reshape(testData.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

# define the architecture of the network
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu', padding="same",
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# train the model using SGD
print("[INFO] compiling model...")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# tbCallBackTrain = keras.callbacks.TensorBoard(log_dir='./Graph/train'+datetime.now().strftime("%Y-%m-%d %H:%M:%S"), histogram_freq=0, write_graph=True, write_images=True)
log_path = './Graph/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tb_callback = keras.callbacks.TensorBoard(
    log_dir=log_path,
    histogram_freq=histogram_freq,
    write_graph=True
)

tb_callback.set_model(model)

# Train net:
history = model.fit(trainData, trainLabels, epochs=epochs, batch_size=batch_size,
                    verbose=1, callbacks=[tb_callback],validation_data=([testData], [testLabels])).history

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=batch_size, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                     accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save("./output/model")
