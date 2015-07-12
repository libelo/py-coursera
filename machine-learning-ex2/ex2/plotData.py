from numpy import *
from matplotlib.pyplot import *

def plotData(X, y):
    #PLOTDATA Plots the data points x and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Create New Figure
    fig = figure()
    hold(True)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #


    pos = y==1
    neg = y==0
    plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7) # linewidth doesn't work
    plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    
    

    # ============================================================



    hold(False)

    return fig