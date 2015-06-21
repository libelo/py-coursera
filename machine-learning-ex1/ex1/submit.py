from numpy import *
from submitWithConfiguration import submitWithConfiguration, sprintf
from warmUpExercise import warmUpExercise
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

def submit():

    conf = {}
    conf['assignmentSlug'] = 'linear-regression'
    conf['itemName'] = 'Linear Regression with Multiple Variables'
    conf['partArrays'] = [
    [
      '1',
      [ 'warmUpExercise.m' ],
      'Warm-up Exercise',
    ],
    [
      '2',
      [ 'computeCost.m' ],
      'Computing Cost (for One Variable)',
    ],
    [
      '3',
      [ 'gradientDescent.m' ],
      'Gradient Descent (for One Variable)',
    ],
    [
      '4',
      [ 'featureNormalize.m' ],
      'Feature Normalization',
    ],
    [ 
      '5',
      [ 'computeCostMulti.m' ],
      'Computing Cost (for Multiple Variables)',
    ],
    [
      '6',
      [ 'gradientDescentMulti.m' ],
      'Gradient Descent (for Multiple Variables)',
    ],
    [ 
      '7',
      [ 'normalEqn.m' ],
      'Normal Equations',
    ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)

def output(partId):
	# Random Test Cases
	X1 = column_stack((ones(20), exp(1) + dot(exp(2), arange(0.1, 2.1, 0.1))))
	Y1 = X1[:,1] + sin(X1[:,0]) + cos(X1[:,1])
	X2 = column_stack((X1, X1[:,1]**0.5, X1[:,1]**0.25))
	Y2 = Y1**0.5 + Y1
	if partId == '1':
		return sprintf('%0.5f ', warmUpExercise())
	elif partId == '2':
		return sprintf('%0.5f ', computeCost(X1, Y1, array([0.5 -0.5])))
	elif partId == '3':
		return sprintf('%0.5f ', gradientDescent(X1, Y1, array([0.5, -0.5]), 0.01, 10))
	elif partId == '4':
		return sprintf('%0.5f ', featureNormalize(X2[:,1:3]));
	elif partId == '5':
		return sprintf('%0.5f ', computeCostMulti(X2, Y2, array([0.1, 0.2, 0.3, 0.4])))
	elif partId == '6':
		return sprintf('%0.5f ', gradientDescentMulti(X2, Y2, array([-0.1 -0.2 -0.3 -0.4]), 0.01, 10))
	elif partId == '7':
		return sprintf('%0.5f ', normalEqn(X2, Y2))

if __name__ == '__main__':
	submit()
