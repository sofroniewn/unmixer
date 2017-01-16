from pandas import read_csv
from glob import glob
from os.path import basename, splitext
from numpy import asarray, ones, zeros
from numpy.random import choice, binomial, poisson
from pandas import DataFrame, melt, concat
from .infer import infer
from .generate import generate

def load(path='./spectra/*.csv'):
    files = glob(path)
    dyes = [{'name' : basename(splitext(f)[0]), 'spectrum' : read_csv(f)} for f in files]
    ex = melt(DataFrame({d['name']:d['spectrum']['Excitation'] for d in dyes}), var_name='dye', value_name='excitation')
    em = melt(DataFrame({d['name']:d['spectrum']['Emission'] for d in dyes}), var_name='dye', value_name='emission')
    wv = melt(DataFrame({d['name']:d['spectrum']['Wavelength'] for d in dyes}), var_name='dye', value_name='wavelength_nm')
    df = concat((ex, em, wv), axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df = df.clip(lower=0)
    df['emission_normalized'] = df.groupby('dye')['emission'].transform(lambda x: x/x.sum())
    return df, dyes

def convert(laser_lines, filter_bands, spectra, sum_lasers=False):
    y_ex = asarray([spectra.Excitation[c-300]/100 for c in laser_lines])
    if sum_lasers:
        y_ex = y_ex.sum()
    y_em = asarray([spectra.Emission[filter_bands[i]-300:filter_bands[i+1]-300].sum()/spectra.Emission.sum() for i in range(len(filter_bands)-1)])
    return y_ex, y_em

def run(coeff, counts, repeats, constrained = False):
    return asarray([infer(generate(counts, coeff), coeff, constrained) for i in range(repeats)])

def simulation(coeff, nMixes=100, nRepeats=1000, method='nnls'):
    nDyes = len(coeff['emission'])
    k = []
    for i in range(nMixes):
        counts = coeff['nPhotonsBackground']*ones(nDyes)
        inds = choice(nDyes, coeff['nDyesPerGene'], replace=False)
        if coeff['amplification']:
            nDM = nDyeMolecules(coeff['nTargetsPerDye'], coeff['pTargetBinding'], coeff['nDyeMoleculesPerTarget'])
            counts[inds] = coeff['nPhotonsPerDyeMolecule']*nDM
        else:
            counts[inds] = coeff['nPhotonsPerDyeMolecule']
        results = run(coeff, counts, nRepeats, constrained = method)
        e = errors(results, inds)
        k.append(e)
    return asarray(k)

def errors(results, truth):
    counts = zeros(results.shape[1])
    counts[truth] = 1
    tmp = asarray([(x/x.max()).round() - counts for x in results])
    return abs(tmp).any(axis=1).mean()*100

def nDyeMolecules(nTargetsPerDye, pTargetBinding, nDyeMoleculesPerTarget):
    nInitiators = binomial(nTargetsPerDye, pTargetBinding)
    return sum([poisson(nDyeMoleculesPerTarget) for i in range(nInitiators)])
