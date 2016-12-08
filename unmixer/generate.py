from numpy import outer, asarray
from numpy.random import poisson, normal

def generate(counts, coeff):
    emission = asarray(coeff['emission'])
    excitation = asarray(coeff['excitation'])
    tmp = asarray([outer(x, y) for x,y in zip(emission, excitation)])
    count_bin = asarray([y*x for x,y in zip(counts, tmp)])

    signal = poisson(count_bin).sum(axis=0)*coeff['qe']
    signal += normal(loc=0, scale=coeff['noise'], size=signal.shape)
    return (signal/coeff['graylevel']).astype(int).clip(0, 2**16)
