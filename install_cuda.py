import os
import subprocess
import sys
from urllib.request import urlretrieve

cuda_versions = {
    # Standard CUDA releases
    "110": "https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run",
    "111": "https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run",
    "112": "https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run",
    "113": "https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run",
    "114": "https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run",
    "115": "https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run",
    "116": "https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run",
    "117": "https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run",
    "118": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run",
    "120": "https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run",
    "121": "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run",
    "122": "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run",
    "123": "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run",
    "124": "https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run",
    "125": "https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run",
    
    # PyTorch Nightly Builds
    "121_nightly": {
        "index_url": "https://download.pytorch.org/whl/nightly/cu121",
        "packages": [
            "torch==2.6.0.dev20241112+cu121",
            "torchvision==0.20.0.dev20241112+cu121",
            "torchaudio==2.5.0.dev20241112+cu121",
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
            "nvidia-nvtx-cu12==12.1.105",
            "pytorch-triton==3.1.0+cf34004b8a"
        ]
    }
}

def install_cuda_toolkit(version, filepath, install_path):
    """Install CUDA toolkit from the downloaded installer."""
    install_command = [
        "bash",
        filepath,
        "--no-drm",
        "--no-man-page",
        "--override",
        "--toolkitpath=" + install_path,
        "--toolkit",
        "--silent",
    ]

    print(f"Running command: {' '.join(install_command)}")
    subprocess.run(install_command, check=True)

def install_pytorch_nightly(version_info):
    """Install PyTorch nightly build and dependencies."""
    print(f"Installing PyTorch nightly CUDA dependencies...")
    
    # Install packages using the nightly index
    install_command = [
        "pip", "install",
        "--index-url", version_info["index_url"],
        "--pre"
    ] + version_info["packages"]
    
    print(f"Running command: {' '.join(install_command)}")
    subprocess.run(install_command, check=True)

def install_cuda(version, base_path, download_path):
    """Install CUDA version, handling both toolkit and nightly builds."""
    # Handle PyTorch nightly builds
    if version.endswith("_nightly"):
        if version not in cuda_versions:
            raise ValueError(f"Unknown nightly version: {version}")
        install_pytorch_nightly(cuda_versions[version])
        return

    # Handle standard CUDA toolkit installation
    formatted_version = f"{version[:-1]}.{version[-1]}"
    folder = f"cuda-{formatted_version}"
    install_path = os.path.join(base_path, folder)

    if os.path.exists(install_path):
        print(f"Removing existing CUDA version {version} at {install_path}...")
        subprocess.run(["rm", "-rf", install_path], check=True)

    if version not in cuda_versions or isinstance(cuda_versions[version], dict):
        raise ValueError(f"Invalid CUDA version: {version}")

    url = cuda_versions[version]
    filename = url.split("/")[-1]
    filepath = os.path.join(download_path, filename)

    # Download installer if needed
    if not os.path.exists(filepath):
        print(f"Downloading CUDA version {version} from {url}...")
        urlretrieve(url, filepath)
    else:
        print(f"Installer for CUDA version {version} already downloaded.")

    try:
        # Make installer executable
        subprocess.run(["chmod", "+x", filepath], check=True)
        # Install CUDA toolkit
        install_cuda_toolkit(version, filepath, install_path)
        print(f"CUDA version {version} installed at {install_path}")

    except subprocess.CalledProcessError as e:
        print(f"Installation failed for CUDA version {version}: {e}")
        raise
    finally:
        # Clean up installer
        if os.path.exists(filepath):
            os.remove(filepath)

def setup_environment(base_path, version):
    """Setup environment variables for CUDA."""
    if not version.endswith("_nightly"):
        formatted_version = f"{version[:-1]}.{version[-1]}"
        cuda_path = os.path.join(base_path, f"cuda-{formatted_version}")
        
        # Set environment variables
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        print("\nEnvironment variables set:")
        print(f"CUDA_HOME={os.environ['CUDA_HOME']}")
        print(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
        
        print("\nTo make these permanent, add to your .bashrc:")
        print(f"export CUDA_HOME={cuda_path}")
        print(f"export LD_LIBRARY_PATH={cuda_path}/lib64:$LD_LIBRARY_PATH")

def main():
    user_base_path = os.path.expanduser("~/cuda")
    system_base_path = "/usr/local"
    base_path = user_base_path  # default to user-specific installation
    download_path = "/tmp"  # default download path

    if len(sys.argv) < 2:
        print("Usage: python install_cuda.py <version/all> [user/system] [download_path]")
        print("\nAvailable versions:")
        print("Standard CUDA versions:", ", ".join(v for v in cuda_versions.keys() if not v.endswith("_nightly")))
        print("Nightly builds:", ", ".join(v for v in cuda_versions.keys() if v.endswith("_nightly")))
        sys.exit(1)

    version = sys.argv[1]
    if len(sys.argv) > 2:
        base_path = system_base_path if sys.argv[2] == "system" else user_base_path
    if len(sys.argv) > 3:
        download_path = sys.argv[3]

    # Create necessary directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(download_path, exist_ok=True)

    try:
        # Install CUDA version(s)
        if version == "all":
            for ver in cuda_versions.keys():
                if not ver.endswith("_nightly"):  # Skip nightly builds for 'all'
                    install_cuda(ver, base_path, download_path)
                    setup_environment(base_path, ver)
        elif version in cuda_versions:
            install_cuda(version, base_path, download_path)
            setup_environment(base_path, version)
        else:
            print(f"Invalid CUDA version: {version}")
            print("Available versions:", ", ".join(cuda_versions.keys()))
            sys.exit(1)

    except Exception as e:
        print(f"Installation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
