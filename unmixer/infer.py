from sklearn import linear_model
from numpy import asarray, outer

def infer(signal, coeff):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    y = signal.flatten()*coeff['graylevel']/coeff['qe']
    A = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    reg = linear_model.LinearRegression()
    reg.fit(A.T, y)
    return reg.coef_
