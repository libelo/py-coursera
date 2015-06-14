from numpy import *
from numpy.linalg import *

def normalEqn(X, y):
    #NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.

    theta = zeros(size(X, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    # ---------------------- Sample Solution ----------------------

    theta = linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # -------------------------------------------------------------


    # ============================================================

    return theta
