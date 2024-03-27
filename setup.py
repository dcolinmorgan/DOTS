#!/usr/bin/env python
USE_CUDA=0
from setuptools import setup, find_packages

core_requires = [
  'pytest',
  'pyarrow',
  'spacy',
  'python-dotenv',
  'bs4',
  'pandas',
  'scikit-learn',
  'transformers',
  'pytorch-cpu' if USE_CUDA==0 else 'torch',
  'opensearch-py',
  'requests',
  'nltk',
  'numpy',
  'GNews',
]

# extras_requires =[
#   'torch' if USE_CUDA==1 else 'torch',
# ]
setup(
    name='dots',
    version='0.0.1',
    packages=find_packages(),
    platforms='any',
    python_requires='>=3.8',
    install_requires=core_requires,
    # extras_requires=extras_requires,
    license='BSD',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
    ],
    keywords=['OpenSearch news featurizer pipeline']
)
