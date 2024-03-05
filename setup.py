#!/usr/bin/env python

from setuptools import setup, find_packages

def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

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
]


setup(
    name='mlx_grph',
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
