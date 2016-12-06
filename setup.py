#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='unmixer',
    version=version,
    description='package for unmixing flourophores',
    author='sofroinewn',
    author_email='sfroniewn@gmail.com',
    url='https://github.com/sofroniewn/unmixer',
    packages=['unmixer'],
    install_requires=required,
    long_description='See ' + 'https://github.com/sofroniewn/unmixer',
    license='MIT'
)
