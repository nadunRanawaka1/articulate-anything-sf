#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail  

# Trap to show which command failed
trap 'echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; exit 1' ERR

eval "$(mamba shell hook --shell bash)"

# The torch pinned below is +cu128, so CUDA 12.8 is the matching toolkit. Honor a
# caller-supplied CUDA_HOME rather than assuming the fixed system path exists (see the
# nvcc fallback after env activation for hosts that have neither).
CUDA_VERSION="${CUDA_VERSION:-12.8}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

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

# nvcc is needed for the source builds below (flash-attn, chamfer3D), and torch's
# cpp_extension refuses a toolkit whose version does not match the +cu128 torch pinned
# below. If CUDA_HOME does not hold one, fall back to an nvcc on PATH only when it
# reports the matching release, and otherwise conda-install the matching toolkit into
# this env instead of failing on a fixed system path.
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  NVCC_PATH="$(command -v nvcc 2>/dev/null || true)"
  if [ -n "${NVCC_PATH}" ] && "${NVCC_PATH}" --version 2>/dev/null | grep -q "release ${CUDA_VERSION}"; then
    CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
  else
    if [ -n "${NVCC_PATH}" ]; then
      echo "nvcc on PATH (${NVCC_PATH}) does not report CUDA ${CUDA_VERSION}; installing cuda-toolkit ${CUDA_VERSION} into the env instead."
    else
      echo "nvcc not found at ${CUDA_HOME}/bin/nvcc or on PATH; installing cuda-toolkit ${CUDA_VERSION} into the env."
    fi
    mamba install -c nvidia "cuda-toolkit=${CUDA_VERSION}" -y
    CUDA_HOME="${CONDA_PREFIX}"
  fi
  export CUDA_HOME
  export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
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

cd P3-SAM
pip install -e ".[demo,dev]"

# chamfer3D is only imported by the P3-SAM gradio demo (auto_mask_no_postprocess), not by
# the pipeline path (auto_mask). Build it AFTER the editable installs and tolerate failure,
# so a CUDA toolchain problem cannot take P3-SAM, articulate-anything and CoTracker down
# with it.
if ! (cd utils/chamfer3D && python setup.py install); then
  echo "WARNING: chamfer3D CUDA extension failed to build; continuing." >&2
  echo "         Only the P3-SAM gradio demo needs it; the pipeline does not." >&2
fi

cd ../../../ # back to articulate-anything

pip install -e .

# Install CoTracker
echo "Installing CoTracker..."
(
  cd deps
  # Pin: cotracker_utils.py passes v2=True, which needs this checkout's predictor API,
  # and the shipped cotracker2v1.pth checkpoint matches the CoTracker2 architecture.
  COTRACKER_COMMIT="${COTRACKER_COMMIT:-82e02e8029753ad4ef13cf06be7f4fc5facdda4d}"
  if [ ! -d "co-tracker" ]; then
    git clone https://github.com/facebookresearch/co-tracker
  fi
  cd co-tracker
  git checkout --detach "${COTRACKER_COMMIT}"
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
