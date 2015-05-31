from numpy import *
from cofiCostFunc import cofiCostFunc

def checkCostFunction(lambda_=0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem
    #to check your cost function and gradients
    #   CHECKCOSTFUNCTION(lambda_) Creates a collaborative filering problem
    #   to check your cost function and gradients, it will output the
    #   analytical gradients produced by your code and the numerical gradients
    #   (computed using computeNumericalGradient). These two gradient
    #   computations should result in very similar values.

    ## Create small problem
    X_t = random.rand(4, 3)
    Theta_t = random.rand(5, 3)

    # Zap out most entries
    Y = dot(X_t, Theta_t.T)
    Y[random.rand(*shape(Y)) > 0.5] = 0
    R = where(Y == 0, 0, 1)

    ## Run Gradient Checking
    X = random.randn(*shape(X_t))
    Theta = random.randn(*shape(Theta_t))
    num_users = size(Y, 1)
    num_movies = size(Y, 0)
    num_features = size(Theta_t, 1)

    numgrad = computeNumericalGradient(
        lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_),
        hstack((X.ravel('F'), Theta.ravel('F'))))

    cost, grad = cofiCostFunc(hstack((X.ravel('F'), Theta.ravel('F'))), Y, R,
                              num_users, num_movies, num_features, lambda_)

    print column_stack((numgrad, grad))

    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'

    diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
    print 'If your backpropagation implementation is correct, then'
    print 'the relative difference will be small (less than 1e-9).'
    print '\nRelative Difference: %g' % diff


def computeNumericalGradient(J, theta):
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.

    # Notes: The following code implements numerical gradient checking, and
    #        returns the numerical gradient.It sets numgrad[i] to (a numerical
    #        approximation of) the partial derivative of J with respect to the
    #        i-th input argument, evaluated at theta. (i.e., numgrad[i] should
    #        be the (approximately) the partial derivative of J with respect
    #        to theta[i].)
    #

    numgrad = zeros(shape(theta))
    perturb = zeros(shape(theta))
    e = 1e-4
    for p in ndindex(shape(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad