import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
from keras.callbacks import History 
# history = History()
# import pydot
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from sklearn.model_selection import train_test_split
from utils.ml_functions import create_folds, random_mini_batches, random_mini_batches_cnn
from utils.preprocessing import *
from utils.keras_utils import identity_block, convolutional_block

## global variables
FNAMES = ['train.csv', 'test.csv', 'sample_submission.csv']
TARGET = 'label'
NUM_CLASSES = 10
N_SPLITS = 5
TEST_SIZE = 0.1
SEED = 42
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
MINIBATCH_SIZE = 64


def BaseModel(input_shape = (64, 64, 3), classes = None):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    

    Returns:
    model -- a Model() instance in Keras
    
    """
    
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='model')

    return model

def ResNet50(input_shape = (64, 64, 3), classes = None):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [64, 64, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 512], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 512], stage=3, block='c')
    X = identity_block(X, 3, [64, 64, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    print(X.shape)
    # X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='model')

    return model


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

(m, n_H0, n_W0, n_C0) = X_train.shape
print(X_train.shape)
model = ResNet50(input_shape = (n_H0, n_W0, n_C0), classes = NUM_CLASSES)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, Y_train, epochs = NUM_EPOCHS, batch_size = MINIBATCH_SIZE)
# print(history_dict.keys())
# plt.plot(model.history.history['acc'])
# plt.plot(model.history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


predictions = model.evaluate(X_val, Y_val)
print ("Loss = " + str(predictions[0]))
print ("Test Accuracy = " + str(predictions[1]))

predictions = model.predict(X_test, verbose=0)
predictions = np.argmax(predictions, axis=1)

submission['Label'] = pd.Series(predictions,name="Label")
submission.to_csv("./data/output/final_submission.csv",index=False)
submission.head()
