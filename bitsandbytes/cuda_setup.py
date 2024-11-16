# cuda_setup.py
import ctypes
from typing import Optional, Union
import os
import sys
import torch
from pathlib import Path

from .utils import logger
from .cextension import COMPILED_WITH_CUDA

def get_cuda_lib_handle() -> Optional[ctypes.CDLL]:
    try:
        # Get the library path
        lib_path = os.path.join(os.path.dirname(__file__), 'libbitsandbytes_cuda121.so')
        
        if not os.path.exists(lib_path):
            logger.error(f"CUDA library not found at {lib_path}")
            return None
            
        # Load the library
        lib = ctypes.CDLL(lib_path)
        
        if lib is not None:
            return lib
            
    except Exception as e:
        logger.error(f"Failed to load CUDA library: {str(e)}")
        return None
        
    return None

def is_cuda_available() -> bool:
    return COMPILED_WITH_CUDA and torch.cuda.is_available()
