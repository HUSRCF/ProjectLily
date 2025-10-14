# DataParallel Training Development Manual

## Overview

This manual covers the `trainMGPU_DataParallel.py` script, which provides AMD GPU-compatible multi-GPU training using PyTorch's DataParallel with torch.compile() support.

## Key Features

- **DataParallel Multi-GPU**: Compatible with AMD GPUs (unlike DDP which requires NCCL)
- **torch.compile()**: Modern PyTorch optimization for better performance
- **Mixed Precision (AMP)**: Automatic Mixed Precision training
- **Advanced Loss Monitoring**: Loss explosion detection and adaptive learning rate
- **Selective Layer Training**: Train specific model components
- **GUI + Headless modes**: Flexible usage options

## Architecture Differences from DDP

### DataParallel vs DDP
```python
# DataParallel (used in this script)
model = nn.DataParallel(model)  # Works on AMD GPUs
loss = model(batch)  # Automatically splits batch across GPUs
loss = loss.mean()   # Average loss across GPUs

# DDP (incompatible with AMD)
model = DDP(model)   # Requires NCCL (NVIDIA only)
```

## Running the Script

### GUI Mode (Recommended)
```bash
python trainMGPU_DataParallel.py
```

### Headless Mode
```bash
python trainMGPU_DataParallel.py --headless
```

### With Your Checkpoint
Make sure `config.yaml` contains:
```yaml
train:
  pretrained_path: "/path/to/your/best_step_900.pt"
```

## Key Configuration Options

### Mixed Precision Settings
```yaml
train:
  use_mixed_precision: true
  amp_init_scale: 65536.0  # Starting scale for gradient scaler
```

### torch.compile() Settings
```yaml
train:
  use_torch_compile: true
  compile_mode: "default"  # Options: default, reduce-overhead, max-autotune
  compile_fullgraph: false
```

### DataParallel Specific
```yaml
train:
  batch_size: 8  # Per-GPU batch size (total = batch_size * num_gpus)
  num_workers: 4  # Data loading workers
```

## Debugging Guide

### Common Issues and Solutions

#### 1. Import Hanging (transformers library)
**Symptoms**: Script hangs during import
**Solution**:
```bash
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
python trainMGPU_DataParallel.py
```

#### 2. CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA out of memory`
**Solutions**:
- Reduce batch size in config.yaml
- Enable gradient checkpointing
- Reduce model size or use gradient accumulation

#### 3. torch.compile() Issues
**Symptoms**: Compilation errors or slow startup
**Solutions**:
```python
# Disable torch.compile() in GUI or config
use_torch_compile: false

# Or try different compile modes
compile_mode: "reduce-overhead"  # Instead of "default"
```

#### 4. Loss Explosion
**Symptoms**: Loss becomes NaN or very large
**Solutions**:
- Enable loss explosion protection in GUI
- Reduce learning rate
- Check gradient clipping (default: 1.0)

#### 5. GPU Not Detected
**Symptoms**: "Only 1 GPU available" message
**Debug**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## Performance Optimization

### For AMD GPUs
1. **Use ROCm optimized PyTorch**: Install PyTorch with ROCm support
2. **Set environment variables**:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6000 series
   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
   ```

### Memory Optimization
```yaml
train:
  gradient_accumulation_steps: 16  # Simulate larger batch sizes
  use_gradient_checkpointing: true  # Trade compute for memory
  preload_data_to_ram: true  # If you have enough RAM
```

### Speed Optimization
```yaml
train:
  use_torch_compile: true
  compile_mode: "max-autotune"  # Slower compilation, faster execution
  use_mixed_precision: true
  num_workers: 8  # Adjust based on CPU cores
```

## Monitoring and Logging

### Loss Monitoring Features
- **Explosion Detection**: Automatically reduces LR if loss > threshold
- **Plateau Detection**: Reduces LR if no improvement
- **Early Stopping**: Stops training if no improvement for N steps

### Key Metrics to Watch
1. **Loss Scale** (AMP): Should be stable, not constantly decreasing
2. **Learning Rate**: Should follow warmup + cosine schedule
3. **GPU Utilization**: Check with `nvidia-smi` or `rocm-smi`
4. **Memory Usage**: Monitor for memory leaks

## Development Workflow

### 1. Initial Setup
```bash
# Test GPU detection
python -c "import torch; print(torch.cuda.device_count())"

# Quick model test (if available in original script)
python trainMGPU_DataParallel.py --test-model
```

### 2. Development Cycle
1. **Start with small batch size** (e.g., 2) to test functionality
2. **Enable all debugging options** in GUI
3. **Run for a few steps** to verify everything works
4. **Gradually increase batch size** and optimize settings
5. **Monitor GPU memory usage** throughout

### 3. Production Settings
```yaml
train:
  batch_size: 8  # Adjust based on GPU memory
  gradient_accumulation_steps: 16
  use_mixed_precision: true
  use_torch_compile: true
  compile_mode: "reduce-overhead"
  epochs: 1000
  save_interval_steps: 100
  valid_interval_steps: 50
```

## Code Structure

### Key Classes
- **`TrainingApp`**: Main GUI application
- **`EMA`**: Exponential Moving Average (DataParallel compatible)
- **`AdvancedLossMonitor`**: Loss monitoring and adaptive LR
- **`WarmupCosine`**: Learning rate scheduler

### Critical Sections
```python
# DataParallel setup (lines ~620-640)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    is_dataparallel = True

# Loss handling (lines ~740-750)
if is_dataparallel:
    loss = loss.mean()  # Average across GPUs

# EMA update (lines ~785-790)
unwrapped_model = model.module if is_dataparallel else model
ema.update(unwrapped_model)
```

## Troubleshooting Checklist

### Before Training
- [ ] GPU detection working (`torch.cuda.device_count() > 0`)
- [ ] Config file exists and valid
- [ ] Pretrained checkpoint path correct
- [ ] Data directories exist and contain files
- [ ] Sufficient disk space for checkpoints

### During Training
- [ ] Loss decreasing (not NaN or exploding)
- [ ] GPU utilization high (>80%)
- [ ] Memory usage stable (not increasing)
- [ ] Learning rate following expected schedule
- [ ] Checkpoints saving successfully

### Performance Issues
- [ ] Try different compile modes
- [ ] Adjust batch size and gradient accumulation
- [ ] Check data loading bottlenecks
- [ ] Monitor CPU usage during data loading
- [ ] Verify mixed precision is working

## Advanced Features

### Custom Model Modifications
To modify the model architecture:
1. Edit `modelV1.py`
2. Update `config.yaml` model parameters
3. Test with small batch first
4. Use selective layer training for fine-tuning

### Selective Layer Training
```python
# Available modes in GUI:
- "full": Train entire model
- "global_encoder": Train only audio encoder
- "unet_only": Train U-Net, freeze encoder
- "custom": Select specific layers
```

### Loss Function Customization
Edit the model's forward method to return custom losses:
```python
def forward(self, batch):
    # Your custom loss computation
    total_loss = reconstruction_loss + kl_loss + custom_loss
    return total_loss, {"recon": reconstruction_loss, "kl": kl_loss}
```

## Best Practices

1. **Always start with GUI mode** for initial setup
2. **Use gradient accumulation** instead of very large batch sizes
3. **Enable mixed precision** for faster training
4. **Monitor loss scale** to ensure AMP stability
5. **Save checkpoints frequently** (every 25-100 steps)
6. **Use EMA weights** for final model evaluation
7. **Validate regularly** to catch overfitting early

## Emergency Procedures

### Training Stuck/Frozen
1. Check GPU memory: `nvidia-smi` or `rocm-smi`
2. Kill process: `pkill -f trainMGPU_DataParallel`
3. Check logs for last successful checkpoint
4. Restart with lower batch size

### Corrupted Checkpoint
1. Use previous checkpoint: `step_XXX.pt` instead of `best_step_XXX.pt`
2. Check `final_ema.pt` if available
3. Verify checkpoint integrity:
   ```python
   import torch
   ckpt = torch.load("checkpoint.pt")
   print(ckpt.keys())  # Should contain 'state_dict' and 'ema'
   ```

### Out of Disk Space
1. Clean old checkpoints: Keep only best and recent ones
2. Move data to different drive
3. Reduce save frequency in config
4. Use model compression if available

## Contact and Support

For issues specific to this implementation:
1. Check this manual first
2. Review the original `trainMGPU.py` for reference
3. Test with smaller configurations
4. Enable verbose logging in the console

Remember: DataParallel is simpler than DDP but less efficient for very large scale training. It's perfect for 2-4 GPU setups, especially with AMD GPUs.