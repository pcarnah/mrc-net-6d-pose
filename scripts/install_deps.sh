#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda env create -n mrcnet -f environment.yaml
conda activate mrcnet

conda install -y pytorch3d::pytorch3d --freeze-installed

CUDA_HOME=$CONDA_PREFIX
CUDA_PATH=$CUDA_HOME

pip install imgaug==0.4.0 kornia==0.7.1 mmcv==1.7.1 pycocotools==2.0.7 trimesh==4.0.10
pip install "git+https://github.com/thodan/bop_toolkit"
pip install spatial-correlation-sampler==0.4.0
