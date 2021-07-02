import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage



# def initialize_parameters(NN_SIZE):
#     """
#     Initializes parameters to build a neural network with tensorflow. 
#     Returns:
#     parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
#     """
    
#     tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
#     ### START CODE HERE ### (approx. 6 lines of code)
#     W1 = tf.get_variable("W1", NN_SIZE['W1'], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#     b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
#     W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#     b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
#     W3 = tf.get_variable("W3", NN_SIZE['W3'], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#     b3 = tf.get_variable("b3", NN_SIZE['b3'], initializer = tf.zeros_initializer())
#     ### END CODE HERE ###

#     parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2,
#                   "W3": W3,
#                   "b3": b3}
    
#     return parameters

def initialize_parameters(NN_SIZE):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Note that we will hard code the shape values in the function to make the grading simpler.
    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    
    ### START CODE HERE ### (approx. 2 lines of code)
    # W1 = tf.get_variable("W1", NN_SIZE['W1'], initializer = tf.contrib.layers.xavier_initializer(seed = 0) )
    # W2 = tf.get_variable("W2", NN_SIZE['W2'], initializer = tf.contrib.layers.xavier_initializer(seed = 0) )
    ### END CODE HERE ###

    parameters = {}
    for key, value in NN_SIZE.items():
        parameters[key] = tf.get_variable(key, NN_SIZE[key],
                          initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    # parameters = {"W1": W1,
    #               "W2": W2}
    
    return parameters


def forward_propagation(X, parameters, NUM_CLASSES):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
                                                                 # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X), b1)                            # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                         # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)                           # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                          # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)                           # Z3 = np.dot(W3, A2) + b3

    return Z3

def forward_propagation_cnn(X, parameters, NUM_CLASSES):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Note that for simplicity and grading purposes, we'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(F, NUM_CLASSES, activation_fn=None,
                             weights_initializer=tf.contrib.layers.xavier_initializer())
    ### END CODE HERE ###

    return Z3


# def compute_cost_cnn(Z3, Y):
#     """
#     Computes the cost
    
#     Arguments:
#     Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, num classes)
#     Y -- "true" labels vector placeholder, same shape as Z3
    
#     Returns:
#     cost - Tensor of the cost function
#     """
    
#     ### START CODE HERE ### (1 line of code)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
#     ### END CODE HERE ###
    
#     return cost

# def predict_cnn(X, parameters, NUM_CLASSES):
    
#     tf.reset_default_graph()
    
#     W1 = tf.convert_to_tensor(parameters["W1"])
#     W2 = tf.convert_to_tensor(parameters["W2"])
#     params = {"W1": W1, "W2": W2}
    
#     # x = tf.placeholder("float", shape=(None, 64, 64, 3))
#     (m, n_H0, n_W0, n_C0) = X.shape   
#     x = tf.placeholder("float", shape=[None, n_H0, n_W0, n_C0] )
#     z3 = forward_propagation_cnn(x, params, NUM_CLASSES)
#     a3 = tf.nn.softmax(z3, axis=1)
#     p = tf.argmax(a3, axis=1)
    
#     with tf.Session() as sess: 
#         tf.train.Saver().restore(sess, "./data/output/model.ckpt") # Note: The variables to restore do not have to have been initialized, as restoring is itself a way to initialize variables.
#         prediction = sess.run(p, feed_dict = {x: X})
    
#     return prediction
