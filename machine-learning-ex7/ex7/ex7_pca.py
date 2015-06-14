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
from mpl_toolkits.mplot3d import axes3d, Axes3D

from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print 'Visualizing example dataset for PCA.\n'

#  The following command loads the dataset. You should now have the
#  variable X in your environment
ex7data1 = loadmat('ex7data1.mat')
X = ex7data1['X']

#  Visualize the example dataset
fig = figure()
plot(X[:, 0], X[:, 1], 'bo')
axis([0.5, 6.5, 2, 8])
axis('equal')
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.py
#
print '\nRunning PCA on example dataset.\n'

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, s = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
hold(True)
drawLine(mu, mu + 1.5 * s[0] * U[:,0].T, '-k', linewidth=2)
drawLine(mu, mu + 1.5 * s[1] * U[:,1].T, '-k', linewidth=2)
hold(False)
fig.show()

print 'Top eigenvector:'
print ' U[:,0] = %f %f' % (U[0,0], U[1,0])
print '\n(you should expect to see -0.707107 -0.707107)'

print 'Program paused. Press enter to continue.'
raw_input()


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.py and recoverData.py
#
print '\nDimension reduction on example dataset.\n'

#  Plot the normalized dataset (returned from pca)
plot(X_norm[:, 0], X_norm[:, 1], 'bo')
axis([-4, 3, -4, 3])
axis('equal')

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print 'Projection of the first example: %f' % Z[0]
print '\n(this value should be about 1.481274)\n'

X_rec  = recoverData(Z, U, K)
print 'Approximation of the first example: %f %f' % (X_rec[0,0], X_rec[0,1])
print '\n(this value should be about  -1.047419 -1.047419)\n'

#  Draw lines connecting the projected points to the original points
hold(True)
plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(size(X_norm, 0)):
    drawLine(X_norm[i,:], X_rec[i,:], '--k', linewidth=1)
hold(False)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print '\nLoading face dataset.\n'

#  Load Face dataset
X = loadmat('ex7faces.mat')['X']

#  Display the first 100 faces in the dataset
fig = figure()
displayData(X[:100, :])
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print '\nRunning PCA on face dataset.'
print '(this might take a minute or two ...)\n'

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X);

#  Run PCA
[U, s] = pca(X_norm)

#  Visualize the top 36 eigenvectors found
fig = figure()
displayData(U[:, :36].T)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
print '\nDimension reduction for face dataset.\n'

K = 100
Z = projectData(X_norm, U, K)

print 'The projected data Z has a size of: '
print shape(Z)

print '\nProgram paused. Press enter to continue.'
raw_input()

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print '\nVisualizing the projected (reduced dimension) faces.'

K = 100
X_rec  = recoverData(Z, U, K)

fig = figure()
# Display normalized data
subplot(1, 2, 1)
displayData(X_norm[:100,:])
title('Original faces')
axis('equal')

# Display reconstructed data from only k eigenfaces
subplot(1, 2, 2)
displayData(X_rec[:100,:])
title('Recovered faces')
axis('equal')

fig.show()

print 'Program paused. Press enter to continue.'
raw_input()


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

close('all')

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = imread('bird_small.png').astype(float)

# If imread does not work for you, you can try instead
#   A = loadmat('bird_small.mat')['A'].astype(float) / 255

img_size = shape(A)
X = reshape(A, (-1,3), order='F')
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = (random.rand(1000) * size(X, 0)).astype(int)

#  Visualize the data and centroid memberships in 3D
fig = figure()
ax = Axes3D(fig)
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=100, c=idx[sel], cmap=cm.hsv, vmax=K+1, facecolors='none')
title('Pixel dataset plotted in 3D. Color shows centroid memberships')
fig.show()
print 'Program paused. Press enter to continue.'
raw_input()

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, s = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
fig = figure()
plotDataPoints(Z[sel, :], idx[sel], K)
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()
