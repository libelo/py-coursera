from numpy import *
from submitWithConfiguration import submitWithConfiguration, sprintf
from sigmoid import sigmoid
from costFunction import costFunction
from predict import predict
from costFunctionReg import costFunctionReg

def submit():

    conf = {}
    conf['assignmentSlug'] = 'logistic-regression'
    conf['itemName'] = 'Logistic Regression'
    conf['partArrays'] = [
    [
      '1',
      [ 'sigmoid.m' ],
      'Sigmoid Function',
    ],
    [
      '2',
      [ 'costFunction.m' ],
      'Logistic Regression Cost',
    ],
    [
      '3',
      [ 'costFunction.m' ],
      'Logistic Regression Gradient',
    ],
    [
      '4',
      [ 'predict.m' ],
      'Predict',
    ],
    [ 
      '5',
      [ 'costFunctionReg.m' ],
      'Regularized Logistic Regression Cost',
    ],
    [
      '6',
      [ 'costFunctionReg.m' ],
      'Regularized Logistic Regression Gradient',
    ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)

def output(partId):
    # Random Test Cases
    X = column_stack((ones(20), exp(1) * sin(arange(1, 21, 1)), exp(0.5) * cos(arange(1, 21, 1))))
    y = (sin(X[:,0] + X[:,1]) > 0).astype(int)
    if partId == '1':
        return sprintf('%0.5f ', sigmoid(X))
    elif partId == '2':
        return sprintf('%0.5f ', costFunction(array([0.25, 0.5, -0.5]), X, y))
    elif partId == '3':
        cost, grad = costFunction(array([0.25, 0.5, -0.5]), X, y)
        return sprintf('%0.5f ', grad)
    elif partId == '4':
        return sprintf('%0.5f ', predict(array([0.25, 0.5, -0.5]), X))
    elif partId == '5':
        return sprintf('%0.5f ', costFunctionReg(array([0.25, 0.5, -0.5]), X, y, 0.1))
    elif partId == '6':
        cost, grad = costFunctionReg(array([0.25, 0.5, -0.5]), X, y, 0.1)
        return sprintf('%0.5f ', grad)
    
if __name__ == '__main__':
    submit()
