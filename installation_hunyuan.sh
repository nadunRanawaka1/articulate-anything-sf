#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail  

# Trap to show which command failed
trap 'echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; exit 1' ERR

eval "$(mamba shell hook --shell bash)"

CUDA_VERSION="12.8"
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

# Hunyuan3D-Part is fetched from its PUBLIC upstream at a pinned commit and patched with our
# changes, rather than redistributed as a fork. NOTE: upstream is under the Tencent Hunyuan
# 3D-Part Community License (non-commercial); review before any redistribution. See patches/README.md.
HUNYUAN_UPSTREAM="https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git"
HUNYUAN_PIN="df0c911"   # "switch to safetensors" / PR #15 merge == patch base (tree a5f2033)
if [ ! -d "Hunyuan3D-Part" ]; then
  git clone "$HUNYUAN_UPSTREAM" Hunyuan3D-Part
  cd Hunyuan3D-Part
  # The public hash may differ from our old mirror (duplicated history). If the pin is missing,
  # fall back to the default branch and 3-way apply. See patches/README.md.
  if git checkout "$HUNYUAN_PIN" 2>/dev/null; then
    git apply ../../patches/hunyuan-simfoundry.patch
  else
    echo "WARN: pinned base $HUNYUAN_PIN not found in upstream; applying patch to current HEAD with 3-way merge."
    git apply --3way ../../patches/hunyuan-simfoundry.patch
  fi
  cd ..
fi

cd Hunyuan3D-Part
git submodule update --init --recursive

mamba create -n articulate-anything-hunyuan python=3.10 -y
mamba activate articulate-anything-hunyuan
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install "setuptools<80"
pip install spconv-cu126
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install packaging
pip install ninja psutil
pip install flash-attn --no-build-isolation
pip install huggingface_hub timm omegaconf
pip install viser fpsample trimesh numba gradio pymeshlab
cd P3-SAM/utils/chamfer3D
python setup.py install

cd ../../ # back to P3-SAM

pip install -e ".[demo,dev]"

cd ../../../ # back to articulate-anything

pip install -e .

# Install CoTracker
echo "Installing CoTracker..."
(
  cd deps
  if [ ! -d "co-tracker" ]; then
    git clone https://github.com/facebookresearch/co-tracker
  fi
  cd co-tracker
  pip install -e .
  pip install matplotlib flow_vis tqdm tensorboard
  
  if [ ! -d "checkpoints" ]; then
    mkdir -p checkpoints
  fi
  
  cd checkpoints
  if [ ! -f "cotracker2v1.pth" ]; then
    wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2v1.pth
  fi

  if [ ! -f "cotracker2.pth" ]; then
    wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
  fi
)
echo "CoTracker installed successfully!"

mamba deactivate
