from sklearn import linear_model
from numpy import asarray, outer
from scipy.optimize import nnls

def infer(signal, coeff, constrained = False):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    y = signal.flatten()*coeff['graylevel']/coeff['qe']
    A = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    if not constrained:
        reg = linear_model.LinearRegression()
        reg.fit(A.T, y)
        results = reg.coef_
    else:
        results, rnorm = nnls(A.T, y)

    return results
