import os
from urllib.parse import urlencode
from urllib.request import urlopen
import pickle
import json

from collections import OrderedDict
import numpy as np
import os

from matplotlib import pyplot

# library written for this exercise providing additional functions for assignment submission, and others
import utils 
import requests
#mport submission
#import submission
requests.get('https://www-origin.coursera.org/api/', verify=False)
# define the submission/grader object for this exercise
import ssl

import numpy as np
c = np.array([[2, 3, 1], [1, 1, 1], [2, 2, 2]])
print(c)
d = np.array([[1, 4, 5], [2, 2, 2], [3, 1, 1]])
print(d)
e = c*d
print(e)
f = 1-e
print(f)

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context


# grader = utils.Grader()

########### exercise 1 ##########

# def warmUpExercise():
#     """
#     Example function in Python which computes the identity matrix.
    
#     Returns
#     -------
#     A : array_like
#         The 5x5 identity matrix.
    
#     Instructions
#     ------------
#     Return the 5x5 identity matrix.
#     """    
#     # ======== YOUR CODE HERE ======
#     A = np.eye(5)   # modify this line
    
#     # ==============================
#     return A

# warmUpExercise()
# # appends the implemented function in part 1 to the grader object
# grader[1] = warmUpExercise

# # send the added functions to coursera grader for getting a grade on this part
# #grader.grade() 

# warmUpExercise()

# ########### exercise 2 ##########

# path = '/Users/ksadhwani001/Documents/github/ml-coursera-python-assignments/Exercise1/'
# data = np.loadtxt(path+'Data/ex1data1.txt', delimiter=',')
# X, y = data[:, 0], data[:, 1]
# print(X)
# print(y)
# m = y.size  # number of training examples

# X = np.stack([np.ones(m), X], axis=1)

# def computeCost(X, y, theta):
#     # initialize some useful values
#     m = y.size  # number of training examples
#     # You need to return the following variables correctly
#     J = 0



    
#     # ====================== YOUR CODE HERE =====================

    
#     # ===========================================================
#     return J