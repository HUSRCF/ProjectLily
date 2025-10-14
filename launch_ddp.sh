#!/bin/bash

# DDP Training Launcher Script for AudioSR
# Usage: bash launch_ddp.sh [num_gpus]

# Set default number of GPUs
NUM_GPUS=${1:-2}

echo "🚀 Launching DDP Training with $NUM_GPUS GPUs"
echo "============================================="

# Check if CUDA is available
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA GPU not detected. DDP training requires CUDA."
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
echo "📊 Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "⚠️ Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available."
    echo "   Adjusting to use $AVAILABLE_GPUS GPUs."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "🔧 Environment configured for DDP training"
echo "   NCCL_DEBUG=INFO"
echo "   CUDA_LAUNCH_BLOCKING=0"

# Launch training with torchrun (recommended for PyTorch 1.9+)
echo "🏃 Starting training with torchrun..."
echo "   Command: torchrun --nproc_per_node=$NUM_GPUS trainMGPU_DDP_Compile.py --distributed"
echo ""

torchrun --nproc_per_node=$NUM_GPUS trainMGPU_DDP_Compile.py --distributed

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "❌ Training failed. Check the logs above for errors."
    exit 1
fi