#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda env create -n mrcnet -f environment.yaml
conda activate mrcnet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#conda install -y pytorch3d::pytorch3d --freeze-installed

CUDA_HOME=$CONDA_PREFIX
CUDA_PATH=$CUDA_HOME
export GPU_ARCH="120"
export NVCC_FLAGS="--gpu-architecture=sm_120"

pip install imgaug==0.4.0 kornia==0.7.1 mmcv==1.7.1 pycocotools==2.0.7 trimesh==4.0.10
pip install "../bop_toolkit"

pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" spatial-correlation-sampler
