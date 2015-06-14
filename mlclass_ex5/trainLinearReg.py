from numpy import *
from scipy.optimize import minimize
from itertools import count

from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, lambda_):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    initial_theta = zeros(size(X, 1))

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)
    # Now, costFunction is a function that takes in only one argument
    # setup an iteration counter
    counter = count()
    # Minimize using CG algorithm
    res = minimize(costFunction, initial_theta, method='CG', jac=True,
                   options={'maxiter': 200}, callback=lambda _:counter.next())
    theta = res.x
    print "Iteration %5d | Cost: %e" % (counter.next(), res.fun)

    return theta
