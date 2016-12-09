from pandas import read_csv
from glob import glob
from os.path import basename, splitext
from numpy import asarray
from .infer import infer
from .generate import generate

def load(path='./spectra/*.csv'):
    files = glob(path)
    return [{'name' : basename(splitext(f)[0]), 'spectrum' : read_csv(f)} for f in files]

def convert(laser_lines, filter_bands, spectra, sum_lasers=False):
    y_ex = asarray([spectra.Excitation[c-300]/100 for c in laser_lines])
    if sum_lasers:
        y_ex = y_ex.sum()
    y_em = asarray([spectra.Emission[filter_bands[i]-300:filter_bands[i+1]-300].sum()/spectra.Emission.sum() for i in range(len(filter_bands)-1)])
    return y_ex, y_em

def run(coeff, counts, repeats, constrained = False):
    return asarray([infer(generate(counts, coeff), coeff, constrained) for i in range(repeats)])
