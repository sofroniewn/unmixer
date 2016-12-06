from unmixer import generate, infer, compare
from numpy.random import normal

config = {'noise': [0.1, 0.1]}
coeff = {'emission': normal(size=(32,)), 'excitation': normal(size=(5,))}

signal, count = generate(coeff=coeff, config=config)
estimate = infer(signal=signal, coeff=coeff)
accuracy = compare(estimate, count)

print(accuracy)