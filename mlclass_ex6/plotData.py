from numpy import *
from matplotlib.pyplot import *

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    pos = where(y == 1)
    neg = where(y == 0)

    # Plot Examples
    plot(X[pos, 0], X[pos, 1], 'k+',linewidth=1, markersize=7)
    hold(True)
    plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    hold(False)
