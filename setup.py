#!/usr/bin/env python

from setuptools import setup, find_packages

core_requires = [
  'pytest',
  'spacy',
  'python-dotenv',
  'bs4',
  'scikit-learn',
  'transformers',
  'torch',
  'requests',
  'nltk',
  'numpy',
]

setup(
    name='DOTS',
    version='0.0.1',
    packages=find_packages(),
    platforms='any',
    python_requires='>=3.8',
    install_requires=core_requires,
    license='BSD',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
    ],
    keywords=['OpenSearch news featurizer pipeline']
)
