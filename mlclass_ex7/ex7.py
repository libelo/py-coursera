## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.py
#     projectData.py
#     recoverData.py
#     computeCentroids.py
#     findClosestCentroids.py
#     kMeansInitCentroids.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from numpy import *
from scipy.io import loadmat
from matplotlib.pyplot import *

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function.
#
print 'Finding closest centroids.\n'

# Load an example dataset that we will be using
ex7data2 = loadmat('ex7data2.mat')
X = ex7data2['X']


# Select an initial set of centroids
K = 3  # 3 Centroids
initial_centroids = array([[3., 3.],[6., 2.],[8., 5.]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids);

print 'Closest centroids for the first 3 examples:'
print ' %d %d %d' % tuple(idx[:3])
print '(the closest centroids should be 1, 3, 2 respectively)'

print 'Program paused. Press enter to continue.'
raw_input()

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print '\nComputing centroids means.\n'

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print 'Centroids computed after initial finding of closest centroids:'
print centroids
print '\n(the centroids should be'
print '   [ 2.428301 3.157924 ]'
print '   [ 5.813503 2.633656 ]'
print '   [ 7.119387 3.616684 ]\n'

print 'Program paused. Press enter to continue.'
raw_input()


## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided.
#
print '\nRunning K-Means clustering on example dataset.\n'

# Load an example dataset
ex7data2 = loadmat('ex7data2.mat')
X = ex7data2['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = array([[3., 3.],[6., 2.],[8., 5.]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print 'K-Means Done.\n'

print 'Program paused. Press enter to continue.'
raw_input()

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#
#  You should now complete the code in kMeansInitCentroids.py
#

print 'Running K-Means clustering on pixels from an image.\n'

#  Load an image of a bird
A = imread('bird_small.png').astype(float)

# If imread does not work for you, you can try instead
#   A = loadmat('bird_small.mat')['A'].astype(float) / 255

# Size of the image
img_size = shape(A)

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, (-1,3), order='F')

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly.
# You should complete the code in kMeansInitCentroids.py before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

print '\nProgram paused. Press enter to continue.'
raw_input()

## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we

print '\nApplying K-Means to compress an image.\n'

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
X_recovered = centroids[idx-1,:]

# Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size, order='F')

# Display the original image
fig = figure()
subplot(1, 2, 1)
imshow(A)
title('Original')

# Display compressed image side by side
subplot(1, 2, 2)
imshow(X_recovered)
title('Compressed, with %d colors.' % K)
fig.show()


print 'Program paused. Press enter to continue.'
raw_input()

