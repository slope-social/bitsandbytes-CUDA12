#!/bin/bash

# Standard CUDA URLs
URL110=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
URL111=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
URL112=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
URL113=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
URL114=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
URL115=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run
URL116=https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
URL117=https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
URL118=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
URL120=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
URL121=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
URL122=https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
URL123=https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
URL124=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
URL125=https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run

# PyTorch Nightly Index
PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cu121"

# Function to install PyTorch nightly packages
install_pytorch_nightly() {
    echo "Installing PyTorch nightly packages for CUDA 12.1..."
    pip install --index-url "$PYTORCH_NIGHTLY_INDEX" --pre \
        torch==2.6.0.dev20241112+cu121 \
        torchvision==0.20.0.dev20241112+cu121 \
        torchaudio==2.5.0.dev20241112+cu121 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.21.5 \
        nvidia-nvjitlink-cu12==12.1.105 \
        nvidia-nvtx-cu12==12.1.105 \
        pytorch-triton==3.1.0+cf34004b8a
}

# Function to setup environment variables
setup_env() {
    local folder=$1
    local base_path=$2
    
    echo "Setting up environment variables..."
    echo "export CUDA_HOME=$base_path/$folder" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$base_path/$folder/lib64" >> ~/.bashrc
    echo "export PATH=\$PATH:$base_path/$folder/bin" >> ~/.bashrc
    
    # Also export for current session
    export CUDA_HOME=$base_path/$folder
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$base_path/$folder/lib64
    export PATH=$PATH:$base_path/$folder/bin
}

# Parse arguments
CUDA_VERSION=$1
BASE_PATH=${2:-"/usr/local"}  # Default to /usr/local if not specified
EXPORT_BASHRC=${3:-0}        # Default to not exporting to bashrc

# Show usage if no arguments provided
if [[ -z "$CUDA_VERSION" ]]; then
    echo "Usage: $0 <cuda_version> [base_path] [export_bashrc]"
    echo "Available versions: 110-125 or 121_nightly"
    echo "Example: $0 121 /usr/local 1"
    exit 1
fi

# Handle PyTorch nightly installation
if [[ "$CUDA_VERSION" == "121_nightly" ]]; then
    install_pytorch_nightly
    exit 0
fi

# Map CUDA version to URL and folder name
if [[ "$CUDA_VERSION" =~ ^[0-9]{3}$ ]]; then
    URL_VAR="URL${CUDA_VERSION}"
    URL=${!URL_VAR}
    FOLDER="cuda-${CUDA_VERSION:0:2}.${CUDA_VERSION:2:1}"
    
    if [[ -z "$URL" ]]; then
        echo "Error: Invalid CUDA version. Choose from: 110-125"
        exit 1
    fi
else
    echo "Error: Invalid CUDA version format. Use format: XXX (e.g., 121)"
    exit 1
fi

# Create installation directory if it doesn't exist
mkdir -p "$BASE_PATH"

# Download and install CUDA
echo "Installing CUDA $CUDA_VERSION..."
FILE=$(basename "$URL")
wget "$URL" -O "/tmp/$FILE"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download CUDA installer"
    exit 1
fi

# Remove existing installation if present
if [[ -d "$BASE_PATH/$FOLDER" ]]; then
    echo "Removing existing installation at $BASE_PATH/$FOLDER..."
    rm -rf "$BASE_PATH/$FOLDER"
fi

# Install CUDA
bash "/tmp/$FILE" --no-drm --no-man-page --override --toolkitpath="$BASE_PATH/$FOLDER/" --toolkit --silent

if [[ $? -ne 0 ]]; then
    echo "Error: CUDA installation failed"
    rm -f "/tmp/$FILE"
    exit 1
fi

# Clean up
rm -f "/tmp/$FILE"

# Setup environment if requested
if [[ "$EXPORT_BASHRC" -eq "1" ]]; then
    setup_env "$FOLDER" "$BASE_PATH"
    echo "Environment variables have been added to ~/.bashrc"
    echo "Please run 'source ~/.bashrc' or start a new terminal session"
fi

echo "CUDA $CUDA_VERSION installation complete"
echo "Installation path: $BASE_PATH/$FOLDER"
