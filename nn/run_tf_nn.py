import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from PIL import Image
from scipy import ndimage

from sklearn.model_selection import train_test_split
from utils.tf_utils import *
from utils.ml_functions import create_folds, random_mini_batches, random_mini_batches_cnn
from utils.preprocessing import *

## global variables
FNAMES = ['train.csv', 'test.csv', 'sample_submission.csv']
TARGET = 'label'
NUM_CLASSES = 10
N_SPLITS = 5
TEST_SIZE = 0.1
SEED = 42
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
MINIBATCH_SIZE = 64
MODEL_TYPE = 'cnn'

NN_SIZE = {"W1":[25,784],
           "b1": [25,1],
           "W2": [12,25], 
           "b2": [12,1],
           "W3": [NUM_CLASSES,12],
           "b3": [NUM_CLASSES,1]
            }
NN_SIZE = {"W1": [4,4,1,8],
           "W2": [2,2,8,16]
           }


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    """

    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32,[n_y, None])
    
    return X, Y

def create_placeholders_nn(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder("float", shape=[None, n_H0, n_W0, n_C0] )
    Y = tf.placeholder("float", shape=[None, n_y])

    return X, Y

def compute_cost(Z3, Y):
    """
    Computes the cost
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    if MODEL_TYPE == 'nn':
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Z3), 
                                                                      labels=tf.transpose(Y)))
    elif MODEL_TYPE == 'cnn':
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, 
                                                                      labels = Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.01,
          num_epochs = 100, minibatch_size = 32, MODEL_TYPE = 'nn', print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Implements a three-layer ConvNet in Tensorflow: CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs



    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    
    if MODEL_TYPE == 'nn':
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []                                        # To keep track of the cost
        X, Y = create_placeholders(n_x, n_y)                                   
    
    elif MODEL_TYPE == 'cnn':                                       
        (m, n_H0, n_W0, n_C0) = X_train.shape             
        n_y = Y_train.shape[1]                            
        costs = []
        X, Y = create_placeholders_nn(n_H0, n_W0, n_C0, n_y)                                        

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(NN_SIZE)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    if MODEL_TYPE == 'nn':
        Z3 = forward_propagation(X, parameters, NUM_CLASSES)
    elif MODEL_TYPE == 'cnn':
        Z3 = forward_propagation_cnn(X, parameters, NUM_CLASSES)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            # minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # minibatch_cost = 0.
            # num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            # seed = seed + 1
            if MODEL_TYPE == 'nn':
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            elif MODEL_TYPE == 'cnn':
                minibatches = random_mini_batches_cnn(X_train, Y_train, minibatch_size, seed)
           
            for minibatch in minibatches:
                
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run(fetches=[optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                # _ , temp_cost = sess.run(fetches=[optimizer, cost],
                #                                 feed_dict={X: minibatch_X,
                #                                            Y: minibatch_Y})
                ### END CODE HERE ###
                epoch_cost += minibatch_cost / minibatch_size
                # minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
            # # Print the cost every epoch
            # if print_cost == True and epoch % 5 == 0:
            #     print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            # if print_cost == True and epoch % 1 == 0:
            #     costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # # lets save the parameters in a variable
        # parameters = sess.run(parameters)
        # print ("Parameters have been trained!")

        # Calculate the correct predictions
        if MODEL_TYPE == 'nn':
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        elif MODEL_TYPE == 'cnn':
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        # # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # # plt.show()

        # # Calculate the correct predictions
        # predict_op = tf.argmax(Z3, 1)
        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        # Save the variables to disk
        save_path = tf.train.Saver().save(sess, "./data/output/model.ckpt")
        print (f"Variables saved in path: {save_path}")
        
        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")


        return parameters

def predict(X, parameters, NUM_CLASSES):

    tf.reset_default_graph()

    params = {}
    for key, value in parameters.items():
        params[key] = tf.convert_to_tensor(parameters[key])
    
    if MODEL_TYPE == 'nn':
        (n_x, m) = X.shape 
        x = tf.placeholder("float", [X.shape[0], None])
        z3 = forward_propagation(x, params, NUM_CLASSES)
        p = tf.argmax(z3)

    elif MODEL_TYPE == 'cnn':
        (m, n_H0, n_W0, n_C0) = X.shape   
        x = tf.placeholder("float", shape=[None, n_H0, n_W0, n_C0] )
        z3 = forward_propagation_cnn(x, params, NUM_CLASSES)
        a3 = tf.nn.softmax(z3, axis=1)
        p = tf.argmax(a3, axis=1)

    with tf.Session() as sess: 
        try:
            tf.train.Saver().restore(sess, "./data/output/model.ckpt") # Note: The variables to restore do not have to have been initialized, as restoring is itself a way to initialize variables.
        except:
            pass
        prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

# Loading the dataset
# csv:
X_train, Y_train, X_test, submission = load_dataset(FNAMES, TARGET)
# image:
# fname = ('./data/raw/{}'.format(FNAMES[0]))
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64))
# # my_image = my_image.reshape((1, 64*64*3)).T
# my_image = flatten_images(my_image)
# image = normalise_image(my_image)


# Explore data shape
explore_data_shape(X_train, Y_train, X_test)

# Show example of a picture
# show_image_example(X_train, Y_train, index=10)

# Flatten / reshape images
# X_train, X_test = flatten_images(X_train, X_test)
X_train, X_test = unflatten_images(X_train, X_test)
# print image dimensions after flatten / reshape to inform nn size
# print(X_train.shape)

# Normalise images
X_train, X_test = normalise_images(X_train, X_test)

# # create folds
# # folds = create_folds(N_SPLITS, SEED)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = TEST_SIZE, random_state=SEED)

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train, NUM_CLASSES)
Y_val = convert_to_one_hot(Y_val, NUM_CLASSES)

# Further reshaping - we require that the training examples be stacked horizontally,
# and for cnns, h, w, c, m format
Y_train = Y_train.T
Y_val = Y_val.T
# X_train = X_train.T
# X_val = X_val.T
# X_test = X_test.T

# train model
parameters = model(X_train, Y_train, X_val, Y_val, LEARNING_RATE,
                   NUM_EPOCHS, MINIBATCH_SIZE, MODEL_TYPE, print_cost = True )

# # make predictions

# tf.reset_default_graph()
# checkpoint_path = './my-test-model'
# # Initialize all the variables
# init = tf.global_variables_initializer()

# # Start the session to compute the tensorflow graph
# with tf.Session() as sess:
    
#     # Run the initialization
#     sess.run(init)


#     ## Load the entire model previuosly saved in a checkpoint
#     print("Load the model from path", checkpoint_path)
#     the_Saver = tf.train.import_meta_graph('./my-test-model.meta')
#     the_Saver.restore(sess, checkpoint_path)

#     ## Identify the predictor of the Tensorflow graph
#     predict_op = tf.get_collection('predict_op')[0]

#     ## Identify the restored Tensorflow graph
#     dataFlowGraph = tf.get_default_graph()
#     print([n.name for n in tf.get_default_graph().as_graph_def().node])
#     x = dataFlowGraph.get_tensor_by_name("Placeholder:0")

#     ## Identify the input placeholder to feed the images into as defined in the model 
#     # (m, n_H0, n_W0, n_C0) = X_test.shape 
#     # print(X_test.shape)
#     # x = tf.placeholder("float", shape=[None, n_H0, n_W0, n_C0] )

#     ## Predict the image category
#     prediction = sess.run(predict_op, feed_dict = {x: X_test})



predictions = predict(X_test, parameters, NUM_CLASSES)


submission['Label'] = pd.Series(predictions,name="Label")
submission.to_csv("./data/output/final_submission.csv",index=False)
submission.head()


