from numpy import *
from matplotlib.pyplot import *

from plotData import plotData
from svmPredict import svmPredict

def visualizeBoundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = linspace(min(X[:,1]), max(X[:,1]), 100)
    [X1, X2] = meshgrid(x1plot, x2plot)
    vals = zeros(shape(X1))
    for i in range(size(X1, 1)):
       this_X = column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = svmPredict(model, this_X)

    # Plot the SVM boundary
    hold(True)
    contour(X1, X2, vals, [0], color='b')
    hold(False)
