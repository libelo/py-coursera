import pdb
from numpy import *

def computeCost(X, y, theta):
  #COMPUTECOST Compute cost for linear regression
  #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  #   parameter for linear regression to fit the data points in X and y

  # Initialize some useful values
  m = len(y) # number of training examples

  # You need to return the following variables correctly 
  J = 0

  # ====================== YOUR CODE HERE ======================
  # Instructions: Compute the cost of a particular choice of theta
  #               You should set J to the cost.

  # use "pdb.set_trace()" to drop into the debugger at this point
  
  a = dot(X, theta) # theta는 그냥 array다. ndim은 1이다. 그래도 dot계산이 되는구나. column vector로 해석하는구나.
  b = a - y
  c = b**2
  d = sum(c)
  e = d/(2*m)
  J = e

  
  # =========================================================================

  return J
