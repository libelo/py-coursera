from numpy import *
from matplotlib.pyplot import *

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from plotProgresskMeans import plotProgresskMeans

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example
    #   centroids, idx = RUNKMEANS(X, initial_centroids, max_iters, plot_progress=false)
    #   runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a True/False flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to False by default. runkMeans returns
    #   centroids, a K x n matrix of the computed centroids and idx, a vector of
    #   size m of centroid assignments (i.e. each entry in range [1..K])
    #

    # Plot the data if we are plotting progress
    if plot_progress:
        fig = figure()
        hold(True)

    # Initialize values
    m, n = shape(X)
    K = size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print 'K-Means iteration %d/%d...' % (i+1, max_iters)

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            fig.show()
            print 'Press enter to continue.'
            raw_input()

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    if plot_progress:
        hold(True)

    return centroids, idx
