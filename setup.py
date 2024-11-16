# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import os

from setuptools import find_packages, setup
from setuptools.dist import Distribution

libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.*"))
libs = [os.path.basename(p) for p in libs]
print("libs:", libs)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="bitsandbytes",
    version="1.5",
    author="slope-social",
    author_email="hey@slope.social",
    description="k-bit optimizers and matrix multiplication routines.",
    license="MIT",
    keywords="gpu optimizers optimization 8-bit quantization compression",
    url="https://github.com/slope-social/bitsandbytes-CUDA12",
    packages=find_packages(),
    package_data={"": libs},
    install_requires=[
            "torch>=2.6.0.dev20241112,<=2.6.1",
            "numpy>=2.1.2,<3.0",
            "nvidia-cublas-cu12==12.1.3.1",
            "nvidia-cuda-cupti-cu12==12.1.105",
            "nvidia-cuda-nvrtc-cu12==12.1.105",
            "nvidia-cuda-runtime-cu12==12.1.105",
            "nvidia-cudnn-cu12==9.1.0.70",
            "nvidia-cufft-cu12==11.0.2.54",
            "nvidia-curand-cu12==10.3.2.106",
            "nvidia-cusolver-cu12==11.4.5.107",
            "nvidia-cusparse-cu12==12.1.0.106",
            "nvidia-nccl-cu12==2.21.5",
            "nvidia-nvjitlink-cu12==12.1.105",
            "nvidia-nvtx-cu12==12.1.105"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/nightly/cu121"
    ],
    extras_require={
        "benchmark": ["pandas", "matplotlib"],
        "test": ["scipy", "lion_pytorch"],
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    distclass=BinaryDistribution,
)
