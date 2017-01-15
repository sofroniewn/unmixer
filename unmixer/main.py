from pandas import read_csv
from glob import glob
from os.path import basename, splitext
from numpy import asarray
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
