from sklearn import linear_model
from numpy import asarray, outer, ones, dot
from scipy.optimize import nnls

def infer(signal, coeff, method = 'rl'):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    y = signal.flatten()*coeff['graylevel']/coeff['qe']
    A = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2]).T

    if method == 'rl':
        results = rl(A, y, iterations=2000)
    elif method == 'nnls':
        results, rnorm = nnls(A, y)
    else:
        reg = linear_model.LinearRegression()
        reg.fit(A, y)
        results = reg.coef_

    return results

def rl(A, y, iterations=100):
    A = A.astype('float')
    An = A/A.sum(axis=0)
    y = y.astype('float')

    x = ones(A.shape[1])/A.shape[1]
    for _ in range(iterations):
        c = dot(An, x)
        x *= dot((y+1e-6)/(c+1e-6), An)
    return x*sum(y)/sum(dot(A,x))
