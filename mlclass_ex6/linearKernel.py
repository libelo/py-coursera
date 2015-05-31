from numpy import *

def linearKernel(x1, x2):
    #LINEARKERNEL returns a linear kernel between x1 and x2
    #   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are vectors
    x1 = ravel(x1, order='F')
    x2 = ravel(x2, order='F')

    # Compute the kernel
    sim = dot(x1,x2)      # dot product

    return sim