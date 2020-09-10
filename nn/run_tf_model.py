import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from PIL import Image
from scipy import ndimage
#from utils.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict, explore_data_shape, show_image_example, flatten_images, normalise_images, create_placeholders
import utils.tf_utils as tfu
from utils.ml_functions import create_folds

## global variables
FNAMES = ['train.csv', 'test.csv', 'sample_submission.csv']
TARGET = 'label'
NUM_CLASSES = 10
N_SPLITS = 5
TEST_SIZE = 0.1
SEED = 42

# Loading the dataset
# csv:
X_train, Y_train, X_test, submission = tfu.load_dataset(FNAMES, TARGET)
# image:
# fname = ('./data/raw/{}'.format(FNAMES[0]))
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64))
# # my_image = my_image.reshape((1, 64*64*3)).T
# my_image = flatten_images(my_image)
# image = normalise_image(my_image)


# Explore data shape
# tfu.explore_data_shape(X_train, Y_train, X_test)

# Show example of a picture
# tfu.show_image_example(X_train, Y_train, index=10)

# Flatten images
# X_train, Y_train = flatten_images(X_train, Y_train)

# Normalise images
X_train, X_test = tfu.normalise_images(X_train, X_test)

# create folds
# folds = create_folds(N_SPLITS, SEED)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = TEST_SIZE, random_state=SEED)

# Convert training and test labels to one hot matrices
Y_train = tfu.convert_to_one_hot(Y_train, NUM_CLASSES)
Y_val = tfu.convert_to_one_hot(Y_val, NUM_CLASSES)

# Transpose train arrays - we require that the training exampled be stacked horizontally
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T

# train model
parameters = tfu.model(X_train, Y_train, X_val, Y_val)

# make predictions
predictions = tfu.predict(X_test, parameters)
submission['Label'] = pd.Series(predictions,name="Label")
submission.to_csv("./data/output/final_submission.csv",index=False)
submission.head()


