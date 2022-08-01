#!/bin/bash
BASE_PATH=$1

echo "MAKE SURE LD_LIBRARY_PATH IS EMPTY!"
echo $LD_LIBRARY_PATH

if [[ ! -z "${LD_LIBRARY_PATH}" ]]; then
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi


module unload cuda
module unload gcc

rm -rf dist build
make cleaneggs

make clean
export CUDA_HOME=
make cpuonly CUDA_VERSION=CPU

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110 CUDA_VERSION=110

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.1
#make cuda11x CUDA_VERSION=111
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.2
#make cuda11x CUDA_VERSION=112
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=112 python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.3
#make cuda11x CUDA_VERSION=113
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=113 python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.4
#make cuda11x CUDA_VERSION=114
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=114 python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.5
#make cuda11x CUDA_VERSION=115
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=115 python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.6
#
#make cuda11x CUDA_VERSION=116
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=116 python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.7
#make cuda11x CUDA_VERSION=117
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=117 python -m build
#
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-10.2
#make cuda10x_nomatmul CUDA_VERSION=102
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=102-nomatmul python -m build
#
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.0
#make cuda110_nomatmul CUDA_VERSION=110
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=110-nomatmul python -m build
#
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.1
#make cuda11x_nomatmul CUDA_VERSION=111
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=111-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.2
#make cuda11x_nomatmul CUDA_VERSION=112
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=112-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.3
#make cuda11x_nomatmul CUDA_VERSION=113
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=113-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.4
#make cuda11x_nomatmul CUDA_VERSION=114
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=114-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.5
#make cuda11x_nomatmul CUDA_VERSION=115
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=115-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.6
#
#make cuda11x_nomatmul CUDA_VERSION=116
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=116-nomatmul python -m build
#
#make clean
#export CUDA_HOME=$BASE_PATH/cuda-11.7
#make cuda11x_nomatmul CUDA_VERSION=117
#
#if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  echo "Compilation unsuccessul!" 1>&2
#  exit 64
#fi
#CUDA_VERSION=117-nomatmul python -m build

python -m twine upload dist/* --verbose --repository testpypi
