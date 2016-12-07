from numpy import outer, asarray
from numpy.random import poisson, normal

def generate(coeff, config=None):
	emission = asarray(coeff['emission'])
	excitation = asarray(coeff['excitation'])
	noise = config['noise']
	count = poisson(lam=noise[0], size=(emission.shape[0], excitation.shape[0]))
	signal = count * outer(emission, excitation)
	signal += normal(loc=0, scale=noise[1], size=signal.shape)
	return signal, count
