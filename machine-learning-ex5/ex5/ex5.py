## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.py
#     learningCurve.py
#     polyFeatures.py
#     validationCurve.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from numpy import *
from scipy.io import loadmat
from matplotlib.pyplot import *

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

## ===== HELPERS =====

def addOnes(X):
    return column_stack((ones(size(X,0)),X))

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print 'Loading and Visualizing Data ...'

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
ex5data1 = loadmat('ex5data1.mat')
X = ex5data1['X']
y = ex5data1['y'].ravel()
Xval = ex5data1['Xval']
yval = ex5data1['yval'].ravel()
Xtest = ex5data1['Xtest']
ytest = ex5data1['ytest'].ravel()

# m = Number of examples
m = size(X, 0)

# Plot training data
fig = figure()
plot(X, y, 'rx', markersize=10, linewidth=1.5)
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')
fig.show()
print 'Program paused. Press enter to continue.'
raw_input()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#

theta = array([1., 1.])
J, _ = linearRegCostFunction(addOnes(X), y, theta, 1.)

print 'Cost at theta = [1, 1]: %f ' % J
print '(this value should be about 303.993192)'

print 'Program paused. Press enter to continue.'
raw_input()

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#

theta = array([1., 1.])
J, grad = linearRegCostFunction(addOnes(X), y, theta, 1.)

print 'Gradient at theta = [1, 1]:  [%f, %f]' % (grad[0], grad[1])
print '(this value should be about [-15.303016, 598.250744])'

print 'Program paused. Press enter to continue.'
raw_input()


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#

#  Train linear regression with lambda = 0
lambda_ = 0.
theta = trainLinearReg(addOnes(X), y, lambda_)
print theta
#  Plot fit over the data
fig = figure()
plot(X, y, 'rx', markersize=10, linewidth=1.5)
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')
hold(True)
plot(X, dot(addOnes(X), theta), '--', linewidth=2)
hold(False)
fig.show()

print 'Program paused. Press enter to continue.'
raw_input()


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
#

lambda_ = 0.
error_train, error_val = \
    learningCurve(addOnes(X), y, addOnes(Xval), yval, lambda_)

fig = figure()
plot(arange(m)+1, error_train, arange(m)+1, error_val);
title('Learning curve for linear regression')
legend(('Train', 'Cross Validation'))
xlabel('Number of training examples')
ylabel('Error')
axis([0, 13, 0, 150])
fig.show()

print '# Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
    print '  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i])

print 'Program paused. Press enter to continue.'
raw_input()

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)     # Normalize
X_poly = addOnes(X_poly)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = addOnes(X_poly_test)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = addOnes(X_poly_val)

print 'Normalized Training Example 1:'
print X_poly[0, :]

print '\nProgram paused. Press enter to continue.'
raw_input()


## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda_ = 0.
theta = trainLinearReg(X_poly, y, lambda_)

# Plot training data and fit
fig1 = figure(1)
plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')
title ('Polynomial Regression Fit (lambda = %f)' % lambda_)
fig1.show()

fig2 = figure(2)
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plot(arange(m)+1, error_train, arange(m)+1, error_val)
title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
xlabel('Number of training examples')
ylabel('Error')
axis([0, 13, 0, 100])
legend(('Train', 'Cross Validation'))
fig2.show()

print 'Polynomial Regression (lambda = %f)\n' % lambda_
print '# Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
    print '  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i])

print 'Program paused. Press enter to continue.'
raw_input()

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = \
    validationCurve(X_poly, y, X_poly_val, yval)

close('all')
fig = figure()
plot(lambda_vec, error_train, lambda_vec, error_val)
legend(('Train', 'Cross Validation'))
xlabel('lambda')
ylabel('Error')
fig.show()

print 'lambda\t\tTrain Error\tValidation Error'
for i in range(len(lambda_vec)):
    print ' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i])

print 'Program paused. Press enter to continue.'
raw_input()
