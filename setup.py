import glob
import os
from setuptools import find_packages, setup
from setuptools.dist import Distribution

libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.*"))
libs = [os.path.basename(p) for p in libs]
print("libs:", libs)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name="bitsandbytes",
    version="0.44.2.dev",
    author="Tim Dettmers",
    author_email="dettmers@cs.washington.edu",
    description="k-bit optimizers and matrix multiplication routines with CUDA 12.1 support",
    license="MIT",
    keywords="gpu optimizers optimization 8-bit quantization compression",
    url="https://github.com/slope-social/bitsandbytes-CUDA12",
    packages=find_packages(),
    package_data={
        "": libs,
        "bitsandbytes": [
            "libbitsandbytes_cuda121.so",
            "libbitsandbytes_cuda121_nocublaslt.so",
            "precompiled/cuda121/*"
        ]
    },
    install_requires=[
        'tokenizers>=0.19,<0.20',
        'transformers==4.43.1',
        'torch==2.6.0.dev20241112+cu121; platform_system!="Windows"',
        'torchaudio==2.5.0.dev20241112+cu121; platform_system!="Windows"'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/nightly/cu121'
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
    distclass=BinaryDistribution
)
