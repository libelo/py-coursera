from numpy import *

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = size(X, 0)
    num_labels = size(Theta2, 0)

    # You need to return the following variables correctly
    p = zeros(size(X, 0))

    h1 = sigmoid(dot(column_stack((ones(m), X)), Theta1.T))
    h2 = sigmoid(dot(column_stack((ones(m), h1)), Theta2.T))
    p = argmax(h2, 1) + 1

    # =========================================================================

    return p