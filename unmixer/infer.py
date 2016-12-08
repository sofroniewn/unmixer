from sklearn import linear_model
from numpy import asarray, outer

def infer(signal, coeff):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    norm = tmp.sum(axis=(1,2))
    norm_bin = asarray([z/x for x,z in zip(norm, tmp)])
    y = signal.flatten()*coeff['graylevel']/coeff['qe']
    A = norm_bin.reshape(norm_bin.shape[0], norm_bin.shape[1]*norm_bin.shape[2])
    reg = linear_model.LinearRegression()
    reg.fit(A.T, y)
    return reg.coef_
