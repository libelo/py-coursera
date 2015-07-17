## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
from numpy import *
from scipy.io import loadmat
from matplotlib.pyplot import *

from displayData import displayData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

## Setup the parameters you will use for this part of the exercise
input_layer_size = 400    # 20x20 Input Images of Digits
num_labels = 10           # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

ex3data1 = loadmat('ex3data1.mat') # training data stored in arrays X, y
X = ex3data1['X']
y = ex3data1['y'].ravel()
m = size(X, 0)

# Randomly select 100 data points to display
sel = random.permutation(m)

fig = figure()
displayData(X[sel[:100], :])
fig.show()

print('Program paused. Press enter to continue.')
input()

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print('Training One-vs-All Logistic Regression...')

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

print('Program paused. Press enter to continue.')
input()


## ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)

print('Training Set Accuracy: %f' % (mean(pred == y) * 100))

