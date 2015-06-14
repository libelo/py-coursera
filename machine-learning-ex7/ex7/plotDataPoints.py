from numpy import *
from matplotlib.pyplot import *

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    #   with the same index assignments in idx have the same color

    # Plot the data
    scatter(X[:,0], X[:,1], 100, idx, cmap=cm.hsv, vmax=K+1, facecolors='none')
