# 🎛️ ProjectLily Z IV - Operation Guide

## 📋 Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [Installation & Setup](#installation--setup)
3. [GUI Operation Guide](#gui-operation-guide)
4. [Command Line Usage](#command-line-usage)
5. [Data Preparation](#data-preparation)
6. [Configuration Guide](#configuration-guide)
7. [Training Workflow](#training-workflow)
8. [Inference Workflow](#inference-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🚀 Quick Start Guide

### For Audio Enhancement (Inference)
```bash
# 1. Navigate to project directory
cd ProjectLily_Z_IV

# 2. Launch inference GUI
python inference.py

# 3. In the GUI:
#    - Load a trained model checkpoint
#    - Select input audio file
#    - Click "Start Inference"
#    - Save enhanced audio
```

### For Model Training
```bash
# 1. Prepare your dataset in data/train/high/
# 2. Launch training GUI
python train_gui.py

# 3. In the GUI:
#    - Verify dataset path
#    - Adjust training parameters
#    - Click "Start Training"
```

---

## 🛠️ Installation & Setup

### System Requirements
- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible (recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB+ free space

### Step-by-Step Installation

#### 1. Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv audiosr_env
source audiosr_env/bin/activate  # Linux/Mac
# or
audiosr_env\Scripts\activate     # Windows
```

#### 2. Install Core Dependencies
```bash
# Install PyTorch (CUDA version for GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install audio processing libraries
pip install soundfile librosa scipy numpy

# Install GUI and visualization
pip install matplotlib tkinter rich plotext

# Install additional dependencies
pip install transformers pyyaml tqdm
```

#### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'Torchaudio: {torchaudio.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🖥️ GUI Operation Guide

### Inference GUI (`inference.py`)

#### Main Interface Components
- **Model Section**: Load trained checkpoint files
- **Input Section**: Select audio files for enhancement
- **Parameters**: Adjust inference settings
- **Output Section**: Save and preview results
- **Progress**: Real-time processing status

#### Step-by-Step Operation
1. **Launch Application**
   ```bash
   python inference.py
   ```

2. **Load Model Checkpoint**
   - Click "Browse Model"
   - Navigate to checkpoint file (`.pt` or `.ckpt`)
   - Verify model loads successfully (green indicator)

3. **Select Input Audio**
   - Click "Browse Audio"
   - Choose audio file to enhance
   - Supported formats: WAV, MP3, FLAC, M4A

4. **Configure Parameters (Optional)**
   - **Chunk Size**: Processing segment length (default: 4.0s)
   - **Sample Steps**: Diffusion sampling steps (default: 50)
   - **Guidance Scale**: Enhancement strength (default: 1.5)

5. **Run Inference**
   - Click "Start Inference"
   - Monitor progress bar and status
   - Wait for completion

6. **Save Results**
   - Preview enhanced audio (if player available)
   - Click "Save Output"
   - Choose output location and format

#### Advanced Features
- **Batch Processing**: Process multiple files
- **Real-time Preview**: Listen during processing
- **Quality Analysis**: Compare input/output spectrograms
- **Parameter Presets**: Save/load configuration sets

### Training GUI (`train_gui.py`)

#### Main Interface Components
- **Dataset Section**: Configure data paths and validation
- **Model Section**: Architecture and checkpoint settings
- **Training Section**: Hyperparameters and optimization
- **Progress Section**: Training metrics and visualization
- **Control Section**: Start/stop/resume training

#### Step-by-Step Operation
1. **Launch Application**
   ```bash
   python train_gui.py
   ```

2. **Configure Dataset**
   - Set data root path
   - Verify train/valid folders exist
   - Check audio file counts
   - Validate sample rates

3. **Model Configuration**
   - Choose architecture settings
   - Load pretrained checkpoint (optional)
   - Set up conditioning parameters

4. **Training Parameters**
   - Batch size (start with 1-2)
   - Learning rate (default: 0.0001)
   - Number of epochs
   - Save interval

5. **Start Training**
   - Click "Start Training"
   - Monitor loss curves
   - Check GPU utilization
   - Save checkpoints regularly

6. **Monitor Progress**
   - Training loss visualization
   - Validation metrics
   - Generated samples
   - Resource usage

### Main GUI (`run_gui.py`)

#### Unified Interface
- **Preprocessing Tab**: Audio data preparation
- **Training Tab**: Model training interface
- **Inference Tab**: Audio enhancement
- **Analysis Tab**: Model and data inspection

---

## 💻 Command Line Usage

### Training
```bash
# Basic training
python train.py --config config.yaml

# Training with custom parameters
python train.py \
    --config config.yaml \
    --batch_size 2 \
    --learning_rate 0.0001 \
    --epochs 500

# Resume from checkpoint
python train.py \
    --config config.yaml \
    --resume checkpoints/step_10000.pt
```

### Inference
```bash
# Basic inference
python -c "
from inference import InferenceEngine
import yaml

# Load config
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Create engine and process
engine = InferenceEngine(cfg)
engine.load_checkpoint('path/to/model.pt')
result = engine.enhance_audio('input.wav')
engine.save_audio(result, 'output.wav')
"
```

### Batch Processing
```bash
# Process multiple files
python -c "
import os
from inference import InferenceEngine
import yaml

# Setup
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
engine = InferenceEngine(cfg)
engine.load_checkpoint('model.pt')

# Process all files in directory
input_dir = 'input_audio/'
output_dir = 'enhanced_audio/'
for file in os.listdir(input_dir):
    if file.endswith('.wav'):
        result = engine.enhance_audio(f'{input_dir}/{file}')
        engine.save_audio(result, f'{output_dir}/enhanced_{file}')
"
```

---

## 📁 Data Preparation

### Directory Structure Setup
```
data/
├── train/
│   └── high/           # Training audio files
│       ├── song1.wav
│       ├── song2.wav
│       └── ...
└── valid/
    └── high/           # Validation audio files
        ├── val1.wav
        ├── val2.wav
        └── ...
```

### Audio File Requirements
- **Format**: WAV recommended (16-bit or 24-bit)
- **Sample Rate**: 48kHz (configurable in config.yaml)
- **Duration**: 10+ seconds per file
- **Quality**: Clean, high-quality source material
- **Quantity**: 100+ files for training, 10+ for validation

### Data Preprocessing
```bash
# Use preprocessing GUI
python run_gui.py
# Navigate to Preprocessing tab

# Or use command line tools
python preprocess_audio.py \
    --input_dir raw_audio/ \
    --output_dir data/train/high/ \
    --target_sr 48000 \
    --normalize True
```

### Data Quality Checks
1. **Audio Length**: Verify sufficient duration
2. **Sample Rate**: Consistent across all files
3. **Dynamic Range**: Avoid overly compressed audio
4. **Noise Level**: Clean source material preferred
5. **File Integrity**: Check for corrupted files

---

## ⚙️ Configuration Guide

### Main Configuration File (`config.yaml`)

#### Data Configuration
```yaml
data:
  dataset_root: "data"                  # Dataset directory
  sample_rate: 48000                    # Audio sample rate
  n_mels: 256                          # Mel spectrogram bins
  segment_seconds: 10.24               # Training segment length
  hop_length: 480                      # STFT hop length
  n_fft: 2048                         # STFT window size
  valid_ratio: 0.05                   # Validation split ratio
```

#### Model Configuration
```yaml
model:
  target: "model.LatentDiffusion"       # Model class
  params:
    base_learning_rate: 0.0001          # Base learning rate
    timesteps: 1000                     # Diffusion timesteps
    beta_schedule: "cosine"             # Noise schedule
    parameterization: "v"               # Parameterization type
```

#### Training Configuration
```yaml
train:
  batch_size: 1                         # Batch size (adjust for GPU)
  epochs: 500                           # Total epochs
  lr: 0.0001                           # Learning rate
  gradient_accumulation_steps: 8        # Gradient accumulation
  save_interval_steps: 2000            # Checkpoint frequency
  valid_interval_steps: 1000           # Validation frequency
```

#### Inference Configuration
```yaml
inference:
  chunk_seconds: 4.0                    # Processing chunk size
  chunk_hop_seconds: 3.0               # Chunk overlap
  sample_steps: 50                     # Sampling steps
  guidance_scale: 1.5                  # Enhancement strength
  sampler: "ddim"                      # Sampling method
```

### Parameter Tuning Guide

#### For Better Quality
- Increase `sample_steps` (50 → 100)
- Adjust `guidance_scale` (1.0-2.0)
- Use longer `segment_seconds` during training

#### For Faster Processing
- Reduce `sample_steps` (50 → 25)
- Increase `chunk_seconds` (limited by GPU memory)
- Use `ddim` sampler (fastest)

#### For Memory Optimization
- Reduce `batch_size` (2 → 1)
- Decrease `chunk_seconds` (4.0 → 2.0)
- Enable gradient checkpointing

---

## 🎓 Training Workflow

### Phase 1: Preparation
1. **Data Collection**
   - Gather high-quality audio files
   - Minimum 100+ files, ideally 1000+
   - Diverse content (music, speech, etc.)

2. **Data Organization**
   ```bash
   # Organize files
   mkdir -p data/train/high data/valid/high
   
   # Move 95% to training, 5% to validation
   # Use random sampling for fair split
   ```

3. **Environment Setup**
   ```bash
   # Verify GPU availability
   python -c "import torch; print(torch.cuda.get_device_name())"
   
   # Check disk space
   df -h
   ```

### Phase 2: Initial Training
1. **Start with Default Config**
   - Use provided `config.yaml`
   - Small batch size (1-2)
   - Monitor initial losses

2. **First 1000 Steps**
   ```bash
   python train.py --config config.yaml
   ```
   - Watch for loss decrease
   - Check for NaN values
   - Monitor GPU memory usage

3. **Validation**
   - Run inference on validation set
   - Listen to early results
   - Adjust parameters if needed

### Phase 3: Optimization
1. **Hyperparameter Tuning**
   - Adjust learning rate based on loss curves
   - Experiment with batch size
   - Tune gradient accumulation

2. **Checkpointing Strategy**
   - Save every 2000 steps
   - Keep best checkpoints based on validation
   - Monitor for overfitting

3. **Long Training**
   - Train for 50,000+ steps
   - Use learning rate scheduling
   - Regular quality evaluation

### Phase 4: Evaluation
1. **Quantitative Metrics**
   - Validation loss trends
   - Inference speed benchmarks
   - Memory usage analysis

2. **Qualitative Assessment**
   - A/B testing with original AudioSR
   - Subjective quality evaluation
   - Edge case testing

---

## 🔍 Inference Workflow

### Phase 1: Model Selection
1. **Choose Checkpoint**
   - Latest vs best validation loss
   - Consider training duration
   - Test on representative samples

2. **Model Loading**
   ```python
   # Verify model loads correctly
   engine = InferenceEngine(cfg)
   engine.load_checkpoint('model.pt')
   ```

### Phase 2: Parameter Optimization
1. **Quality vs Speed Tradeoff**
   - Test different `sample_steps` values
   - Adjust `guidance_scale` for content type
   - Optimize `chunk_seconds` for memory

2. **Content-Specific Settings**
   - Music: Higher guidance scale (1.5-2.0)
   - Speech: Lower guidance scale (1.0-1.5)
   - Noisy audio: More sampling steps

### Phase 3: Processing
1. **Single File**
   ```python
   result = engine.enhance_audio('input.wav')
   engine.save_audio(result, 'enhanced.wav')
   ```

2. **Batch Processing**
   - Process multiple files in sequence
   - Monitor progress and errors
   - Save intermediate results

3. **Quality Control**
   - Listen to outputs
   - Check for artifacts
   - Compare with inputs

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors
**Symptom**: "KeyError" or "Missing keys" when loading checkpoint
```bash
# Solution: Check model compatibility
python inspector.py --checkpoint model.pt
```

**Alternative**: Use model conversion tool
```bash
python convert_trained_model.py --input old_model.pt --output new_model.pt
```

#### 2. CUDA Out of Memory
**Symptom**: "RuntimeError: CUDA out of memory"
```yaml
# Solution: Reduce memory usage in config.yaml
train:
  batch_size: 1              # Reduce from 2
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size

inference:
  chunk_seconds: 2.0         # Reduce from 4.0
```

#### 3. Silent Output Audio
**Symptom**: Output is very quiet or silent
```python
# Check input audio levels
import soundfile as sf
import numpy as np

audio, sr = sf.read('input.wav')
rms = np.sqrt(np.mean(audio**2))
print(f"Input RMS: {rms}")

# If RMS < 0.001, audio is too quiet
# The system will auto-normalize, but verify it's working
```

#### 4. Training Instability
**Symptom**: Loss spikes or NaN values
```yaml
# Solution: Adjust training parameters
train:
  lr: 0.00005              # Reduce learning rate
  gradient_clip_val: 1.0   # Add gradient clipping
  warmup_steps: 5000       # Increase warmup
```

#### 5. GUI Crashes
**Symptom**: Interface freezes or crashes
```bash
# Check dependencies
pip install --upgrade tkinter matplotlib

# Run with error output
python inference.py 2>&1 | tee error.log
```

### Debugging Tools

#### 1. Model Inspector
```bash
python inspector.py --checkpoint model.pt --verbose
```

#### 2. Data Validator
```bash
python dataset_loader.py --data_dir data/ --check_integrity
```

#### 3. System Monitor
```bash
# GPU monitoring
nvidia-smi -l 1

# Memory usage
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'CPU: {psutil.cpu_percent()}%')
"
```

---

## 💡 Best Practices

### Training Best Practices

#### 1. Data Quality
- **Source Material**: Use high-quality, uncompressed audio
- **Diversity**: Include various genres, speakers, instruments
- **Balance**: Equal distribution across categories
- **Preprocessing**: Normalize levels, remove DC bias

#### 2. Training Strategy
- **Start Small**: Begin with subset of data
- **Progressive Training**: Increase complexity gradually
- **Regular Validation**: Monitor overfitting early
- **Checkpoint Management**: Keep multiple saved states

#### 3. Resource Management
- **GPU Utilization**: Monitor with `nvidia-smi`
- **Memory Optimization**: Use gradient accumulation
- **Storage**: Regular cleanup of old checkpoints
- **Monitoring**: Log training metrics consistently

### Inference Best Practices

#### 1. Model Selection
- **Validation Performance**: Choose based on metrics, not just loss
- **Generalization**: Test on out-of-domain audio
- **Stability**: Verify consistent results across runs

#### 2. Processing Optimization
- **Batch Size**: Process multiple files together when possible
- **Chunk Size**: Balance quality vs memory usage
- **Preprocessing**: Consistent normalization and format

#### 3. Quality Assurance
- **A/B Testing**: Compare with baseline methods
- **Edge Cases**: Test with challenging audio (very quiet, noisy, etc.)
- **Format Compatibility**: Verify output format requirements

### Deployment Best Practices

#### 1. Environment Consistency
- **Dependencies**: Pin exact versions
- **Configuration**: Use version control for configs
- **Documentation**: Keep operation logs

#### 2. Performance Monitoring
- **Speed Benchmarks**: Regular performance testing
- **Quality Metrics**: Automated quality assessment
- **Error Tracking**: Log and analyze failures

#### 3. Maintenance
- **Model Updates**: Regular retraining with new data
- **System Updates**: Keep dependencies current
- **Backup Strategy**: Regular checkpoint and config backups

---

## 📞 Support and Resources

### Getting Help
1. **Check Documentation**: README.md, SOLUTION_SUMMARY.md
2. **Debug Tools**: Use built-in inspector and validator tools
3. **Community**: Search for similar issues online
4. **Issue Reporting**: Create detailed bug reports

### Useful Commands Cheat Sheet
```bash
# Quick setup
python -m venv env && source env/bin/activate
pip install torch torchaudio soundfile matplotlib tkinter

# Training
python train.py --config config.yaml

# Inference GUI
python inference.py

# Model inspection
python inspector.py --checkpoint model.pt

# Data validation
python dataset_loader.py --data_dir data/ --validate

# GPU monitoring
nvidia-smi -l 1
```

### Performance Expectations
- **Training**: 1-2 hours per 1000 steps (RTX 3080)
- **Inference**: 5-10x real-time (GPU), 0.1x real-time (CPU)
- **Memory**: 8-12GB GPU RAM for batch_size=1
- **Storage**: 1-2GB per 10,000 step checkpoint

---

**Last Updated**: 2024  
**Version**: 4.0  
**Compatibility**: Python 3.8+, PyTorch 2.0+