from numpy import *
from matplotlib.pyplot import *

from multivariateGaussian import multivariateGaussian

def visualizeFit(X, mu, sigma2):
    #VISUALIZEFIT Visualize the dataset and its estimated distribution.
    #   VISUALIZEFIT(X, mu, sigma2) This visualization shows you the
    #   probability density function of the Gaussian distribution. Each example
    #   has a location (x1, x2) that depends on its feature values.
    #

    coords = linspace(0,30,61)
    X1, X2 = meshgrid(coords, coords)
    Z = multivariateGaussian(column_stack((X1.ravel(),X2.ravel())), mu, sigma2)
    Z = reshape(Z, shape(X1))

    plot(X[:, 0], X[:, 1],'bx')
    hold(True)
    # Do not plot if there are infinities
    if not any(isinf(Z)):
        contour(X1, X2, Z, power(10., arange(-20,0,3)))
    hold(False)
