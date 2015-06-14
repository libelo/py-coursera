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

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The argmax function might come in useful. In particular, the argmax
    #       function returns the index of the max element, for more information
    #       see 'help(argmax)'. If your examples are in rows, then, you can
    #       use argmax(A, 1) to obtain the max for each row.
    #









    # =========================================================================


    return p
