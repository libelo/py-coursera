from numpy import *
from submitWithConfiguration import submitWithConfiguration, sprintf
from linearRegCostFunction import linearRegCostFunction
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from validationCurve import validationCurve

def submit():

    conf = {}
    conf['assignmentSlug'] = 'regularized-linear-regression-and-bias-variance'
    conf['itemName'] = 'Regularized Linear Regression and Bias/Variance'
    conf['partArrays'] = [
    [
      '1',
      [ 'linearRegCostFunction.m' ],
      'Regularized Linear Regression Cost Function',
    ],
    [
      '2',
      [ 'linearRegCostFunction.m' ],
      'Regularized Linear Regression Gradient',
    ],
    [
      '3',
      [ 'learningCurve.m' ],
      'Learning Curve',
    ],
    [
      '4',
      [ 'polyFeatures.m' ],
      'Polynomial Feature Mapping',
    ],
    [
      '5',
      [ 'validationCurve.m' ],
      'Validation Curve',
    ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)

def output(partId):
    # Random Test Cases
    X = column_stack([ones(10), sin(arange(1,15,1.5)), cos(arange(1,15,1.5))])
    y = sin(arange(1,30,3))
    Xval = column_stack([ones(10), sin(arange(0,14,1.5)), cos(arange(0,14,1.5))])
    yval = sin(arange(1,11))
    if partId == '1':
        J, _ = linearRegCostFunction(X, y, hstack([0.1, 0.2, 0.3]), 0.5)
        return sprintf('%0.5f ', J)
    elif partId == '2':
        J, grad = linearRegCostFunction(X, y, hstack([0.1, 0.2, 0.3]), 0.5)
        return sprintf('%0.5f ', grad)
    elif partId == '3':
        error_train, error_val = \
            learningCurve(X, y, Xval, yval, 1);
        return sprintf('%0.5f ', vstack([error_train, error_val]))
    elif partId == '4':
        X_poly = polyFeatures(X[1,:], 8)
        return sprintf('%0.5f ', X_poly)
    elif partId == '5':
        lambda_vec, error_train, error_val = \
                validationCurve(X, y, Xval, yval)
        return sprintf('%0.5f ', \
            vstack([lambda_vec, error_train, error_val]))
    
if __name__ == '__main__':
    submit()
