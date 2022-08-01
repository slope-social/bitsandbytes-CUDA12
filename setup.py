# Copyright (c) Facebook, Inc. and its affiliates. 
#   
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup, find_packages


import os

def all_libs(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


version = os.getenv("CUDA_VERSION", "cpu")
prefix = '' if version == 'cpu' else 'cuda'

setup(
    name=f"bitsandbytes-{prefix}{version}",
    version=f"0.30.2",
    author="Tim Dettmers",
    author_email="dettmers@cs.washington.edu",
    description="8-bit optimizers and matrix multiplication routines.",
    license="MIT",
    keywords="gpu optimizers optimization 8-bit quantization compression",
    url="http://packages.python.org/bitsandbytes",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["debug_cuda = bitsandbytes.debug_cli:cli"],
    },
    package_data={'': ['libbitsandbytes*.so']},
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 4 - Beta",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
