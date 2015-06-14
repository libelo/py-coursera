## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from numpy import *
from scipy.io import loadmat
from scipy.optimize import minimize
from matplotlib.pyplot import *

from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings

## ===== HELPERS =====

def serialize(*args):
    return hstack(a.ravel('F') for a in args)

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#
print 'Loading movie ratings dataset.\n'

#  Load data
ex8_movies = loadmat('ex8_movies.mat')
Y = ex8_movies['Y']
R = ex8_movies['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print 'Average rating for movie 1 (Toy Story): %f / 5\n' % mean(Y[0, R[0, :]])

#  We can "visualize" the ratings matrix by plotting it with imagesc
fig = figure()
imshow(Y, aspect=0.4)
ylabel('Movies')
xlabel('Users')
fig.show()

print '\nProgram paused. Press enter to continue.'
raw_input()

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
ex8_movieParams = loadmat('ex8_movieParams.mat')
X = ex8_movieParams['X']
Theta = ex8_movieParams['Theta']

#  Reduce the data set size so that this runs faster
num_users, num_movies, num_features = 4, 5, 3
X = X[:num_movies,:num_features]
Theta = Theta[:num_users,:num_features]
Y = Y[:num_movies,:num_users]
R = R[:num_movies,:num_users]

#  Evaluate cost function
J, _ = cofiCostFunc(serialize(X, Theta), Y, R, num_users, num_movies, num_features, 0.)

print 'Cost at loaded parameters: %f ' % J
print '(this value should be about 22.22)'

print '\nProgram paused. Press enter to continue.'
raw_input()


## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement
#  the collaborative filtering gradient function. Specifically, you should
#  complete the code in cofiCostFunc.m to return the grad argument.
#
print '\nChecking Gradients (without regularization) ... '

#  Check gradients by running checkNNGradients
checkCostFunction()

print '\nProgram paused. Press enter to continue.'
raw_input()


## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#

#  Evaluate cost function
J, _ = cofiCostFunc(serialize(X, Theta), Y, R, num_users, num_movies, num_features, 1.5)

print 'Cost at loaded parameters (lambda = 1.5): %f' % J
print '(this value should be about 31.34)'

print '\nProgram paused. Press enter to continue.'
raw_input()


## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement
#  regularization for the gradient.
#

#
print '\nChecking Gradients (with regularization) ... '

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)

print '\nProgram paused. Press enter to continue.'
raw_input()


## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
movieList = loadMovieList()

#  Initialize my ratings
my_ratings = zeros(len(movieList))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1-1] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[98-1] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[7-1] = 3
my_ratings[12-1]= 5
my_ratings[54-1] = 4
my_ratings[64-1]= 5
my_ratings[66-1]= 3
my_ratings[69-1] = 5
my_ratings[183-1] = 4
my_ratings[226-1] = 5
my_ratings[355-1]= 5

print '\n\nNew user ratings:'
for rating, name in zip(my_ratings, movieList):
    if rating > 0:
        print 'Rated %d for %s' % (rating, name)

print '\nProgram paused. Press enter to continue.'
raw_input()


## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users
#

print '\nTraining collaborative filtering...'

#  Load data
ex8_movies = loadmat('ex8_movies.mat')
Y = ex8_movies['Y']
R = ex8_movies['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = column_stack((my_ratings, Y))
R = column_stack((my_ratings != 0, R))

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_users = size(Y, 1)
num_movies = size(Y, 0)
num_features = 10

# Set Initial Parameters (Theta, X)
X = random.randn(num_movies, num_features)
Theta = random.randn(num_users, num_features)

initial_parameters = serialize(X, Theta)

# Set Regularization
lambda_ = 10
extra_args = (Y, R, num_users, num_movies, num_features, lambda_)
import sys
def callback(p): sys.stdout.write('.')

res = minimize(cofiCostFunc, initial_parameters, extra_args, method='CG',
               jac=True, options={'maxiter':100}, callback=callback)
theta = res.x
cost = res.fun

print "\nFinal cost: %e" % cost

# Unfold the returned theta back into U and W
X = reshape(theta[:num_movies*num_features], (num_movies, num_features), order='F')
Theta = reshape(theta[num_movies*num_features:], (num_users, num_features), order = 'F')

print 'Recommender system learning completed.'

print '\nProgram paused. Press enter to continue.'
raw_input()

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = dot(X, Theta.T)
nr = sum(R,1)
my_predictions = p[:,0] + Ymean

movieList = loadMovieList()

ix = argsort(my_predictions)
print '\nTop recommendations for you:'
for j in ix[:-11:-1]:
    print 'Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j])

print '\n\nOriginal ratings provided:'
for rating, name in zip(my_ratings, movieList):
    if rating > 0:
        print 'Rated %d for %s' % (rating, name)

print '\nProgram paused. Press enter to continue.'
raw_input()