from . import research, utils
from .autograd._functions import (
    MatmulLtState,
    bmm_cublas,
    matmul,
    matmul_4bit,
    matmul_cublas,
    mm_cublas,
)
from .nn import modules
from .optim import adam
from .cextension import lib  # Add this import
from .cuda_specs import get_cuda_specs  # Add this import

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "1.5"

# Add these exports
COMPILED_WITH_CUDA = lib is not None and lib.compiled_with_cuda
cuda_setup = get_cuda_specs()
