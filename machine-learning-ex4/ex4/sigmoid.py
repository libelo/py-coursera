from numpy import *

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.
    return 1. / (1. + exp(-z))
