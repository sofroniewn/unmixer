# unmixer

> package for unmixing flourophores

## example

Create a set of coefficients and a configuration, genreate data, perform inference, and compare the results

```python
from unmixer import generate, infer, compare
from numpy.random import normal

config = {'noise': [0.1, 0.1]}
coeff = {'emission': normal(size=(32,)), 'excitation': normal(size=(5,))}

data = generate(coeff=coeff, config=config)
counts = infer(data=data)
accuracy = compare(data, counts)
```

## components

#### `generate`
Generate measurements from a given coefficient set and configuration

#### `infer`
Infer flourophore concentrations given a set of measurements and coefficients

#### `compare`
Compare true and estimated flourophoe concentrations
