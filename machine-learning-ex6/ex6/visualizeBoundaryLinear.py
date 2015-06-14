from numpy import *
from matplotlib.pyplot import *
from plotData import plotData

def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    #   learned by the SVM and overlays the data on it

    w = model.w
    b = model.b
    xp = linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = - (w[0]*xp + b) / w[1]
    plotData(X, y)
    hold(True)
    plot(xp, yp, '-b')
    hold(False)


