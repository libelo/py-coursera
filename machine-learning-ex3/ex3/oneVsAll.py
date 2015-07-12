from numpy import *
from scipy import optimize
from lrCostFunction import lrCostFunction

def oneVsAll(X, y, num_labels, lambda_):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta
    #corresponds to the classifier for label i
    #   all_theta = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m = size(X, 0)
    n = size(X, 1)

    # You need to return the following variables correctly
    all_theta = zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = column_stack((ones(m), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: You can use y == c to obtain a vector of booleans that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For the python version of this assignment, we recommend using
    #       scipy.optimize.minimize with the CG method (fmin_cg).
    #       It is okay to use a for-loop (for c in range(1,num_labels+1)) to
    #       loop over the different classes.
    #
    #
    # Example Code for minimize:
    #
    #     # Set Initial theta
    #     initial_theta = zeros(n + 1)
    #
    #     # Run minimize to obtain the optimal theta
    #     # This function will return a Result object. Theta can be retrieved in
    #     # the 'x' attribute and the cost in the 'fun' attribute.
    #     res = optimize.minimize(lrCostFunction, initial_theta, args=(X,(y == c).astype(int),lambda_), \
    #                             method='CG', jac=True, options={'maxiter':50})
    #     theta, cost = res.x, res.fun




    for c in range(1, num_labels+1):
        initial_theta = zeros(n + 1)
        res = optimize.minimize(lrCostFunction, initial_theta, args=(X,(y == c),lambda_), \
                                        method='CG', jac=True, options={'maxiter':50})
        theta, cost = res.x, res.fun
        all_theta[c-1,:] = theta






    # =========================================================================


    return all_theta