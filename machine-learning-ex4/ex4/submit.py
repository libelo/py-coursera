from numpy import *
from submitWithConfiguration import submitWithConfiguration, sprintf
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient

def submit():

    conf = {}
    conf['assignmentSlug'] = 'neural-network-learning'
    conf['itemName'] = 'Neural Networks Learning'
    conf['partArrays'] = [
    [
      '1',
      [ 'nnCostFunction.m' ],
      'Feedforward and Cost Function',
    ],
    [
      '2',
      [ 'nnCostFunction.m' ],
      'Regularized Cost Function',
    ],
    [
      '3',
      [ 'sigmoidGradient.m' ],
      'Sigmoid Gradient',
    ],
    [
      '4',
      [ 'nnCostFunction.m' ],
      'Neural Network Gradient (Backpropagation)',
    ],
    [
      '5',
      [ 'nnCostFunction.m' ],
      'Regularized Gradient',
    ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)

def output(partId):
    # Random Test Cases
    X = reshape(3 * sin(arange(1, 31, 1)), (3,10), order='F')
    Xm = reshape(sin(arange(1, 33)), (16,2), order='F') / 5
    ym = 1 + arange(1, 17) % 4
    t1 = sin(reshape(arange(1,25,2), (4,3), order='F'))
    t2 = cos(reshape(arange(1,41,2), (4,5), order='F'))
    t = hstack([t1.ravel('F'), t2.ravel('F')])
    if partId == '1':
        J, _ = nnCostFunction(t, 2, 4, 4, Xm, ym, 0)
        return sprintf('%0.5f ', J)
    elif partId == '2':
        J, _ = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5)
        return sprintf('%0.5f ', J)
    elif partId == '3':
        return sprintf('%0.5f ', sigmoidGradient(X))
    elif partId == '4':
        J, grad = nnCostFunction(t, 2, 4, 4, Xm, ym, 0)
        out = sprintf('%0.5f ', J)
        return out + sprintf('%0.5f ', grad)
    elif partId == '5':
        J, grad = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5)
        out = sprintf('%0.5f ', J)
        return out + sprintf('%0.5f ', grad)
    
if __name__ == '__main__':
    submit()
