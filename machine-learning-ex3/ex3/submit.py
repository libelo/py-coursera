from numpy import *
from submitWithConfiguration import submitWithConfiguration, sprintf
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from predict import predict

def submit():

    conf = {}
    conf['assignmentSlug'] = 'multi-class-classification-and-neural-networks'
    conf['itemName'] = 'Multi-class Classification and Neural Networks'
    conf['partArrays'] = [
    [
      '1',
      [ 'lrCostFunction.m' ],
      'Regularized Logistic Regression',
    ],
    [
      '2',
      [ 'oneVsAll.m' ],
      'One-vs-All Classifier Training',
    ],
    [
      '3',
      [ 'predictOneVsAll.m' ],
      'One-vs-All Classifier Prediction',
    ],
    [
      '4',
      [ 'predict.m' ],
      'Neural Network Prediction Function',
    ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)

def output(partId):
    # Random Test Cases
    X = column_stack((ones(20), exp(1) * sin(arange(1, 21, 1)), exp(0.5) * cos(arange(1, 21, 1))))
    y = sin(X[:,0] + X[:,1]) > 0
    Xm = array([[-1, -1], [-1, -2], [-2, -1], [-2, -2], [1, 1], [1, 2], [2, 1], [2, 2],
      [-1, 1], [-1, 2], [-2, 1], [-2, 2], [1, -1], [1, -2], [-2, -1], [-2, -2]])
    ym = array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    t1 = sin(arange(1, 25, 2).reshape(4, 3, order='F'))
    t2 = cos(arange(1, 41, 2).reshape(4, 5, order='F'))
    
    if partId == '1':
        J, grad = lrCostFunction(array([0.25, 0.5, -0.5]), X, y, 0.1)
        out = sprintf('%0.5f ', J)
        return out + sprintf('%0.5f ', grad)
    elif partId == '2':
        return sprintf('%0.5f ', oneVsAll(Xm, ym, 4, 0.1))
    elif partId == '3':
        return sprintf('%0.5f ', predictOneVsAll(t1, Xm))
    elif partId == '4':
        return sprintf('%0.5f ', predict(t1, t2, Xm))
    
if __name__ == '__main__':
    submit()
