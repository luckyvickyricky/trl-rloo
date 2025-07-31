#!/bin/bash
conda remove -n trl_rloo --all -y 2>/dev/null || true
conda create -n trl_rloo python=3.10 -y
source ~/miniconda/etc/profile.d/conda.sh
conda activate trl_rloo
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy pandas transformers accelerate datasets scikit-learn bitsandbytes nvidia-ml-py3
cd trl && pip install -e . 