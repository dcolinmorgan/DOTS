#!/usr/bin/env python

from setuptools import setup, find_packages

core_requires = [
  'spacy',
  'bs4',
  'sklearn',
  'transformers',
  'requests',
  'xml',
  'nltk',
  'string',
  'numpy',
  'setuptools'
]

setup(
    name='mlx_grph',
    version='0.0.1',
    packages = find_packages(),
    platforms='any',
    python_requires='>=3.7',
    install_requires=core_requires,
    license='BSD',
    classifiers=[
        'Development Status :: 0 - Fun',
    ],
    keywords=['news cpu featurizer']
)
