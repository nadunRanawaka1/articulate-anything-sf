#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail  

# Trap to show which command failed
trap 'echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; exit 1' ERR

eval "$(mamba shell hook --shell bash)"


if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

if [ ! -d "PartField" ]; then
  git clone https://github.com/nv-tlabs/PartField.git
fi

cd PartField

git apply ../../patches/partfield.patch || echo "Patch could not be applied, ignoring."

if [ ! -d "model" ]; then
  mkdir model
fi
cd model

if [ ! -f "model.ckpt" ]; then
  wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt
  mv model_objaverse.ckpt model.ckpt
fi
cd ..


git submodule update --init --recursive

mamba create -n articulate-anything-partfield python=3.10 -y
mamba activate articulate-anything-partfield
pip install psutil
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
# Must track the torch/CUDA pair pinned above.
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install packaging
pip install ninja
sudo apt install libx11-6 libgl1 libxrender1
pip install vtk
pip install flash-attn --no-build-isolation
pip install huggingface_hub timm omegaconf
pip install viser fpsample trimesh numba gradio



cd ../../ # back to articulate-anything

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
