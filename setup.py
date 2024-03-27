#!/usr/bin/env python

from setuptools import setup, find_packages

core_requires = [
  'pytest',
  'pyarrow'
  'spacy',
  'python-dotenv',
  'bs4',
  'pandas',
  'scikit-learn',
  'transformers',
  'torch',
  'opensearch-py',
  'requests',
  'nltk',
  'pandas'
  'numpy',
  'GNews',
]

setup(
    name='dots',
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
