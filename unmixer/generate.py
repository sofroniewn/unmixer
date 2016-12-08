from numpy import outer, asarray
from numpy.random import poisson, normal

def generate(counts, coeff):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    norm = tmp.sum(axis=(1,2))
    count_bin = asarray([y*z/x for x,y,z in zip(norm, counts, tmp)])

    signal = poisson(count_bin).sum(axis=0)*coeff['qe']
    signal += normal(loc=0, scale=coeff['noise'], size=signal.shape)
    return (signal/coeff['graylevel']).astype(int)
