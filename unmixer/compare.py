from numpy import asarray

def compare(truth, result):
	a = asarray(truth)
	b = asarray(result)
	return ((a - b) ** 2).sum()