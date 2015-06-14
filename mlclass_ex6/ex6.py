## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.py
#     dataset3Params.py
#     processEmail.py
#     emailFeatures.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from numpy import *
from scipy.io import loadmat
from matplotlib.pyplot import *
from functools import wraps

from plotData import plotData
from svmTrain import svmTrain
from linearKernel import linearKernel
from visualizeBoundaryLinear import visualizeBoundaryLinear
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

print 'Loading and Visualizing Data ...'

# Load from ex6data1:
# You will have X, y in your environment
ex6data1 = loadmat('ex6data1.mat')
X = ex6data1['X']
y = ex6data1['y'].ravel().astype(int8)

# Plot training data
fig = figure()
plotData(X, y)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1:
# You will have X, y in your environment
ex6data1 = loadmat('ex6data1.mat')
X = ex6data1['X']
y = ex6data1['y'].ravel().astype(int8)

print '\nTraining Linear SVM ...'

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1.0
model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
fig = figure()
visualizeBoundaryLinear(X, y, model)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print '\nEvaluating the Gaussian Kernel ...'

x1 = array([1, 2, 1])
x2 = array([0, 4, -1])
sigma = 2.0
sim = gaussianKernel(x1, x2, sigma)

print 'Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = 0.5 :'
print '\t%f\n(this value should be about 0.324652)' % sim

print 'Program paused. Press enter to continue.'
raw_input()

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print 'Loading and Visualizing Data ...'

# Load from ex6data2:
# You will have X, y in your environment
ex6data2 = loadmat('ex6data2.mat')
X = ex6data2['X']
y = ex6data2['y'].ravel().astype(int8)

# Plot training data
fig = figure()
plotData(X, y)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print '\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...'

# Load from ex6data2:
# You will have X, y in your environment
ex6data2 = loadmat('ex6data2.mat')
X = ex6data2['X']
y = ex6data2['y'].ravel().astype(int8)

# SVM Parameters
C = 1.0
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

# wraps(gaussianKernel)(...) allows for svmTrain and svmPredict (used in visualizeBoundary)
# to recognize the gaussian kernel and use an optimized code path

model = svmTrain(X, y, C, wraps(gaussianKernel)(lambda x1, x2: gaussianKernel(x1, x2, sigma)))
fig = figure()
visualizeBoundary(X, y, model)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print 'Loading and Visualizing Data ...'

# Load from ex6data3:
# You will have X, y in your environment
ex6data3 = loadmat('ex6data3.mat')
X = ex6data3['X']
y = ex6data3['y'].ravel().astype(int8)

# Plot training data
fig = figure()
plotData(X, y)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
#

# Load from ex6data3:
# You will have X, y in your environment
ex6data3 = loadmat('ex6data3.mat')
X = ex6data3['X']
y = ex6data3['y'].ravel().astype(int8)
Xval = ex6data3['Xval']
yval = ex6data3['yval'].ravel().astype(int8)

# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
model = svmTrain(X, y, C, wraps(gaussianKernel)(lambda x1, x2: gaussianKernel(x1, x2, sigma)))
fig = figure()
visualizeBoundary(X, y, model)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

