from numpy import *

def normalizeRatings(Y, R):
    #NORMALIZERATINGS Preprocess data by subtracting mean rating for every
    #movie (every row)
    #   Ynorm, Ymean = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    #   has a rating of 0 on average, and returns the mean rating in Ymean.
    #

    m, n = shape(Y)
    Ymean = zeros(m)
    Ynorm = zeros(shape(Y))
    for i in range(m):
        idx = where(R[i, :] == 1)
        Ymean[i] = mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
