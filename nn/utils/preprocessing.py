import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset(FNAMES, TARGET):
    train = [x for x in FNAMES if "train" in x]
    if len(train) == 1:
        train_df = pd.read_csv('./data/raw/{}'.format(train[0]))
        X_train = train_df.drop(columns = [TARGET]).values # your train set features
        Y_train = train_df[[TARGET]].values # your train set labels

    test = [x for x in FNAMES if "test" in x]
    if len(test) == 1:
        test_df = pd.read_csv('./data/raw/{}'.format(test[0]))
        X_test = test_df.values # your test set features
        
        # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        # Y_train = Y_train.reshape((1, Y_train.shape[0]))
        # Y_test = test_set_y.reshape((1, test_set_y.shape[0]))
    
    submission = [x for x in FNAMES if "submission" in x]
    if len(submission) == 1:
        submission_df = pd.read_csv('./data/raw/{}'.format(submission[0]))
       
    return X_train, Y_train, X_test, submission_df


def show_image_example(X_train, Y_train, index=0):
    """
    Requires data to be in a numpy array, in a grid shape (i.e. pixels cannot be unraveled)
    """
    plt.imshow(X_train[index].reshape(int(np.sqrt(X_train.shape[1])),int(np.sqrt(X_train.shape[1]))))
    plt.show()
    print ("y = " + str(np.squeeze(Y_train[:, index])))

def flatten_images(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    
    print ("train_x's shape: " + str(X_train.shape))
    print ("test_x's shape: " + str(X_test.shape))
    
    return X_train, X_test

def unflatten_images(X_train, X_test):

    X_train = X_train.reshape(-1, int(np.sqrt(X_train.shape[1])), int(np.sqrt(X_train.shape[1])), 1)
    X_test = X_test.reshape(-1, int(np.sqrt(X_test.shape[1])), int(np.sqrt(X_test.shape[1])), 1)
    print("x_train shape: ",X_train.shape)
    print("x_test shape: ", X_test.shape)

    return X_train, X_test

def normalise_images(X_train, X_test):
    X_train = X_train/255
    X_test = X_test/255

    return X_train, X_test

def explore_data_shape(X_train, Y_train, X_test):
    m_train = X_train.shape[0]
    num_px = np.sqrt(X_train.shape[1])
    m_test = X_test.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 1)")
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    # print ("Y_test shape: " + str(Y_test.shape))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y