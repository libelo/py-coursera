## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from numpy import *
from scipy.io import loadmat
from scipy.optimize import minimize
from matplotlib.pyplot import *

from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict


## Setup the parameters you will use for this exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

ex4data1 = loadmat('ex4data1.mat')
X = ex4data1['X']
y = ex4data1['y'].ravel()
m = size(X, 0)

# Randomly select 100 data points to display
sel = list(range(m))
random.shuffle(sel)

fig = figure()
displayData(X[sel[:100], :])
fig.show()

print('Program paused. Press enter to continue.')
input()

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
ex4weights = loadmat('ex4weights.mat')
Theta1 = ex4weights['Theta1']
Theta2 = ex4weights['Theta2']


# Unroll parameters
nn_params = hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.py to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lambda_ = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %f ' % J)
print('(this value should be about 0.287629)')

print('\nProgram paused. Press enter to continue.')
input()

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lambda_ = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %f ' % J)
print('(this value should be about 0.383770)')

print('Program paused. Press enter to continue.')
input()

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.py file.
#

print('Evaluating sigmoid gradient...')

g = sigmoidGradient(array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ', end='')
for gg in g: print('%f ' % gg, end='')
print('\n')

print('Program paused. Press enter to continue.')
input()

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.py)

print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = hstack((initial_Theta1.ravel(order='F'),
                            initial_Theta2.ravel(order='F')))

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.py to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation...')

#  Check gradients by running checkNNGradients
checkNNGradients()

print('\nProgram paused. Press enter to continue.')
input()


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('Checking Backpropagation (w/ Regularization) ...')

#  Check gradients by running checkNNGradients
lambda_ = 3;
checkNNGradients(lambda_)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                             hidden_layer_size, num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' % debug_J)
print('(this value should be about 0.576051)\n')

print('Program paused. Press enter to continue.')
input()


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural network.
#  To train your neural network, we will now use scipy.optimize.minimize,
#  with the "CG" algorithm, instead of "BFGS" or "Newton-CG". Recall that
#  these advanced optimizers are able to train our cost functions efficiently
#  as long as we provide them with the gradient computations.
#
print('Training Neural Network...')

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options = {'maxiter': 50}

#  You should also try different values of lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p,
                                        input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)

# Setup a callback for displaying the cost at the end of each iteration
class Callback(object):
    def __init__(self):
        self.it = 0
    def __call__(self, p):
        self.it += 1
        print("Iteration %5d | Cost: %e" % (self.it, costFunction(p)[0]))
    def __del__(self):
        print()

res = minimize(costFunction, initial_nn_params, method='CG',
               jac=True, options=options, callback=Callback())
nn_params = res.x
cost = res.fun


# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, (input_layer_size + 1)), order='F')

Theta2 = reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                 (num_labels, (hidden_layer_size + 1)), order='F')

print('Program paused. Press enter to continue.')
input()


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('Visualizing Neural Network...')

fig = figure()
displayData(Theta1[:,1:])
fig.show()

print('\nProgram paused. Press enter to continue.')
input()

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('Training Set Accuracy: %f' % (mean(pred == y) * 100))


