#!/usr/bin/env bash
# Launch the Isaac Lab XR teleop session pinned to physical GPU1 (nvidia-smi idx 1).
set -u
cd "$HOME/dev/furniture_assembly/IsaacLab"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME/dev/furniture_assembly/conda_env"
export OMNI_KIT_ACCEPT_EULA=YES
# Make CUDA ordinals match nvidia-smi/PCI order so --device cuda:1 is unambiguous.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Load a single NVIDIA Vulkan ICD; see cloudxr_gpu1.env for why.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task IsaacContrib-Stack-Cube-Franka-IK-Abs \
    --device cuda:1 \
    --visualizer none \
    --xr \
    --cloudxr_env "$HOME/dev/furniture_assembly/cloudxr_gpu1.env" \
    --kit_args "--/renderer/multiGpu/activeCudaGpus=1,"
