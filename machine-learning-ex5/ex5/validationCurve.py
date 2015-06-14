from numpy import *
from trainLinearReg import trainLinearReg

def validationCurve(X, y, Xval, yval):
    #VALIDATIONCURVE Generate the train and validation errors needed to
    #plot a validation curve that we can use to select lambda
    #   lambda_vec, error_train, error_val = VALIDATIONCURVE(X, y, Xval, yval)
    #       returns the train and validation errors (in error_train, error_val)
    #       for different values of lambda. You are given the training set (X,
    #       y) and validation set (Xval, yval).
    #

    # Selected values of lambda (you should not change this)
    lambda_vec = array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.])

    # You need to return these variables correctly.
    error_train = zeros_like(lambda_vec)
    error_val = zeros_like(lambda_vec)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the validation errors in error_val. The
    #               vector lambda_vec contains the different lambda parameters
    #               to use for each calculation of the errors, i.e,
    #               error_train(i), and error_val(i) should give
    #               you the errors obtained after training with
    #               lambda = lambda_vec(i)
    #
    # Note: You can loop over lambda_vec with the following:
    #
    #       for i in range(len(lambda_vec)):
    #           lambda_ = lambda_vec[i]
    #           # Compute train / val errors when training linear
    #           # regression with regularization parameter lambda_
    #           # You should store the result in error_train[i]
    #           # and error_val[i]
    #           ....
    #
    #


    
    

    # =========================================================================

    return lambda_vec, error_train, error_val
