#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail  # Catch errors in pipelines

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

# SAMesh is fetched from its PUBLIC upstream at a pinned commit and patched with our changes,
# rather than redistributed as a fork (upstream is unlicensed). See patches/README.md.
SAMESH_UPSTREAM="https://github.com/gtangg12/samesh.git"
SAMESH_PIN="8ec0410"   # last upstream (gtangg12) commit == patch base
if [ ! -d "samesh" ]; then
  git clone "$SAMESH_UPSTREAM" samesh
  cd samesh
  git checkout "$SAMESH_PIN"
  git apply ../../patches/samesh-simfoundry.patch
  cd ..
fi
cd samesh
git submodule update --init --recursive   # SAM2 (third_party/segment-anything-2)


mamba create -n articulate-anything-samesh python=3.11 -y
mamba activate articulate-anything-samesh

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install "setuptools<80"
pip install -e .
pip install open3d pygltflib
cd third_party/segment-anything-2
pip install -e ".[demo]"

cd ../../../../ # back to articulate-anything
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
  
  mkdir -p checkpoints
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
