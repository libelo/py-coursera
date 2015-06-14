from numpy import *

def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    mu = mean(X, 0)
    sigma = std(X, 0, ddof=1)
    X_norm = (X - mu) / sigma

    # ============================================================

    return X_norm, mu, sigma
