from numpy import *
from matplotlib.pyplot import *

from polyFeatures import polyFeatures

def plotFit(min_x, max_x, mu, sigma, theta, p):
    #PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    #Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure
    hold(True)

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = arange(min_x - 15, max_x + 25.01, 0.05)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma

    # Add ones
    X_poly = column_stack((ones(size(x)), X_poly))

    # Plot
    plot(x, dot(X_poly, theta), '--', linewidth=2)

    # Hold off to the current figure
    hold(False)
