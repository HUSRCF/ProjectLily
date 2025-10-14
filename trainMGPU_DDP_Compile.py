import os
import sys
import yaml
import math
import threading
import argparse
from typing import Optional
import signal

# --- GUI Imports (conditionally imported) ---
GUI_AVAILABLE = True
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    GUI_AVAILABLE = False
    # Create dummy classes for non-GUI mode
    class tk:
        class Tk: pass
        class StringVar: pass
        class BooleanVar: pass

# --- Training Imports ---
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn

# --- Local Project Imports ---
from modelV1 import LatentDiffusion
from dataset_loader import PairedMelDataset
from audiosr.latent_diffusion.util import instantiate_from_config

# --- Base Configuration ---
BASE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE, 'config.yaml')

# =====================================================================================
# SECTION: DISTRIBUTED TRAINING UTILITIES
# =====================================================================================

def setup_ddp(rank, world_size, port=12355):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size():
    """Get world size (number of processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1

def get_rank():
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0

# =====================================================================================
# SECTION: UTILITY & CORE TRAINING LOGIC
# =====================================================================================

def ensure_dir(p: str):
    """Ensures a single directory exists."""
    os.makedirs(p, exist_ok=True)

def ensure_project_dirs(cfg: dict):
    """Ensures that all necessary directories from the config file exist."""
    try:
        # Create main output directory and its checkpoints subdir
        outp = os.path.join(BASE, cfg['experiment']['out_dir'], 'checkpoints')
        ensure_dir(outp)

        # Create data directories
        dataset_root = os.path.join(BASE, cfg['data']['dataset_root'])
        ensure_dir(os.path.join(dataset_root, 'train', cfg['data']['high_dir_name']))
        ensure_dir(os.path.join(dataset_root, 'train', cfg['data']['low_dir_name']))
        ensure_dir(os.path.join(dataset_root, 'valid', cfg['data']['high_dir_name']))
        ensure_dir(os.path.join(dataset_root, 'valid', cfg['data']['low_dir_name']))
    except KeyError as e:
        raise KeyError(f"Config file is missing a required key: {e}. Please check your 'config.yaml'.")

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EMA:
    """
    Exponential Moving Average of model parameters, maintained in FP32 for stability.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        # For DDP, use the underlying module
        actual_model = model.module if hasattr(model, 'module') else model
        # Create a shadow copy of parameters in full FP32 precision
        self.shadow = {
            k: p.data.clone().to(dtype=torch.float32)
            for k, p in actual_model.named_parameters() if p.requires_grad
        }

    def update(self, model):
        # For DDP, use the underlying module
        actual_model = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            for k, p in actual_model.named_parameters():
                if p.requires_grad and k in self.shadow:
                    p_fp32 = p.data.to(dtype=torch.float32)
                    self.shadow[k] = (1.0 - self.decay) * p_fp32 + self.decay * self.shadow[k]

class AdvancedLossMonitor:
    """
    Advanced loss monitoring with explosion detection, adaptive LR, and early stopping.
    Enhanced for mixed precision and distributed training compatibility.
    """
    def __init__(self, explosion_threshold=10.0, lr_reduction_factor=0.5, patience=1000, min_lr=1e-7):
        self.explosion_threshold = explosion_threshold
        self.lr_reduction_factor = lr_reduction_factor
        self.patience = patience
        self.min_lr = min_lr

        # Loss tracking
        self.loss_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.lr_reductions = 0
        self.exploded_count = 0

        # Moving averages for stability
        self.short_ma = 0.0  # Short-term moving average (last 10 steps)
        self.long_ma = 0.0   # Long-term moving average (last 100 steps)
        self.ma_alpha_short = 0.1  # Smoothing for short MA
        self.ma_alpha_long = 0.01  # Smoothing for long MA

        # Mixed precision specific monitoring
        self.inf_nan_count = 0
        self.scale_reductions = 0

    def update(self, loss_value, step, scaler=None):
        """Update loss monitoring and return actions to take."""
        actions = {
            'reduce_lr': False,
            'stop_training': False,
            'skip_step': False,
            'message': ''
        }

        # Enhanced NaN/Inf detection for mixed precision
        if not torch.isfinite(torch.tensor(loss_value)):
            self.inf_nan_count += 1
            actions['skip_step'] = True
            actions['message'] = f"⚠️ NaN/Inf loss detected at step {step} (count: {self.inf_nan_count}), skipping step"

            # If too many NaN/Inf, suggest reducing scale
            if self.inf_nan_count > 3 and scaler is not None:
                actions['message'] += f" - Consider checking gradient scaler (current scale: {scaler.get_scale():.0e})"

            return actions

        # Update moving averages
        if len(self.loss_history) == 0:
            self.short_ma = loss_value
            self.long_ma = loss_value
        else:
            self.short_ma = (1 - self.ma_alpha_short) * self.short_ma + self.ma_alpha_short * loss_value
            self.long_ma = (1 - self.ma_alpha_long) * self.long_ma + self.ma_alpha_long * loss_value

        self.loss_history.append(loss_value)

        # 1. Loss Explosion Detection
        if loss_value > self.explosion_threshold:
            self.exploded_count += 1
            actions['reduce_lr'] = True
            actions['message'] = f"🔥 Loss explosion detected ({loss_value:.4f} > {self.explosion_threshold}), reducing LR"

            # If multiple explosions, be more aggressive
            if self.exploded_count > 3:
                actions['reduce_lr'] = True
                actions['message'] += f" (explosion #{self.exploded_count})"

        # 2. Adaptive Learning Rate (Plateau Detection)
        elif len(self.loss_history) > 100:
            # Check if loss is plateauing (short MA not improving over long MA)
            if self.short_ma > self.long_ma * 1.01 and step % 500 == 0:  # Check every 500 steps
                actions['reduce_lr'] = True
                actions['message'] = f"📉 Loss plateau detected (short: {self.short_ma:.4f}, long: {self.long_ma:.4f}), reducing LR"

        # 3. Track best loss for early stopping
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        # 4. Early Stopping Check
        if self.steps_without_improvement >= self.patience:
            actions['stop_training'] = True
            actions['message'] = f"🛑 Early stopping: no improvement for {self.patience} steps"

        return actions

    def reduce_learning_rate(self, optimizer, console):
        """Reduce learning rate for all parameter groups."""
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        self.lr_reductions += 1
        self.exploded_count = max(0, self.exploded_count - 1)  # Reset explosion counter after LR reduction

        if is_main_process():
            console.print(f"[yellow]📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e} (reduction #{self.lr_reductions})[/yellow]")

        if new_lr <= self.min_lr:
            if is_main_process():
                console.print(f"[red]⚠️ Learning rate hit minimum ({self.min_lr:.2e})[/red]")

        return new_lr

    def get_stats(self):
        """Get monitoring statistics."""
        return {
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'lr_reductions': self.lr_reductions,
            'exploded_count': self.exploded_count,
            'short_ma': self.short_ma,
            'long_ma': self.long_ma,
            'inf_nan_count': self.inf_nan_count,
            'scale_reductions': self.scale_reductions
        }

class WarmupCosine:
    """Learning rate scheduler with warmup and cosine decay."""
    def __init__(self, opt, base_lr, warmup, max_steps, min_ratio=0.1):
        self.opt, self.base, self.w, self.m, self.r = opt, base_lr, warmup, max_steps, min_ratio
        self.step_id = 0

    def step(self):
        if self.step_id < self.w:
            f = (self.step_id + 1) / max(1, self.w)
        else:
            t = (self.step_id - self.w) / max(1, self.m - self.w)
            f = self.r + 0.5 * (1 - self.r) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = self.base * f
        self.step_id += 1

def load_pretrained(model: torch.nn.Module, path: str, device: str, console: Console):
    if not path or not os.path.exists(path):
        if is_main_process():
            console.print("[yellow]No pretrained path found. Starting from scratch.[/yellow]")
        return

    if is_main_process():
        console.print(f"[cyan]Loading pretrained weights from:[/] [default]{path}[/]")

    ckpt = torch.load(path, map_location=device)

    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if is_main_process():
        console.print(f"[green]Weights loaded.[/] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            console.print(f"  [dim]Missing (first 5): {missing[:5]}[/dim]")
        if unexpected:
            console.print(f"  [dim]Unexpected (first 5): {unexpected[:5]}[/dim]")

# =====================================================================================
# SECTION: MINIMAL TESTING FRAMEWORK
# =====================================================================================

def quick_model_test(model, sample_batch, device, use_amp=False, use_compile=False):
    """
    Minimal test to verify model functionality without heavy computation.
    Returns (success, message, time_taken)
    """
    import time

    try:
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            if use_amp:
                with autocast():
                    loss, _ = model(sample_batch)
            else:
                loss, _ = model(sample_batch)

        end_time = time.time()

        if torch.isfinite(loss).all():
            return True, f"✅ Model test passed. Loss: {loss.item():.4f}", end_time - start_time
        else:
            return False, f"❌ Model produced NaN/Inf loss: {loss.item()}", end_time - start_time

    except Exception as e:
        return False, f"❌ Model test failed: {str(e)}", 0.0

# =====================================================================================
# SECTION: GUI CONFIGURATION COLLECTOR
# =====================================================================================

class ConfigurationGUI(tk.Tk):
    """
    A GUI that collects training configuration and then shuts down.
    Used with --gui flag to configure training parameters before headless execution.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config_ready = False
        self.final_config = None

        self.title("AudioSR Training Configuration")
        self.geometry("800x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Style Configuration ---
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), foreground='white', background='#0078D7')

        self.create_widgets()

    def on_closing(self):
        """Handle window closing."""
        if not self.config_ready:
            if messagebox.askokcancel("Quit", "Configuration not saved. Exit anyway?"):
                self.quit()
        else:
            self.quit()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="AudioSR Training Configuration",
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 20))

        # Create notebook for different configuration categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 20))

        # === Basic Settings Tab ===
        basic_frame = ttk.Frame(notebook, padding="15")
        notebook.add(basic_frame, text="Basic Settings")

        # Pretrained model path
        ttk.Label(basic_frame, text="Pretrained Weights Path:").grid(row=0, column=0, sticky='w', pady=5)
        self.pretrained_path_var = tk.StringVar(value=self.cfg['train'].get('pretrained_path', ''))
        pretrained_frame = ttk.Frame(basic_frame)
        pretrained_frame.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0))
        ttk.Entry(pretrained_frame, textvariable=self.pretrained_path_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(pretrained_frame, text="Browse", command=self.select_pretrained_file).pack(side='right', padx=(5, 0))

        # Training parameters
        ttk.Label(basic_frame, text="Epochs:").grid(row=1, column=0, sticky='w', pady=5)
        self.epochs_var = tk.StringVar(value=str(self.cfg['train'].get('epochs', 1000)))
        ttk.Entry(basic_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky='w', pady=5, padx=(10, 0))

        ttk.Label(basic_frame, text="Batch Size:").grid(row=2, column=0, sticky='w', pady=5)
        self.batch_size_var = tk.StringVar(value=str(self.cfg['train'].get('batch_size', 8)))
        ttk.Entry(basic_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))

        ttk.Label(basic_frame, text="Gradient Accumulation Steps:").grid(row=3, column=0, sticky='w', pady=5)
        self.grad_accum_var = tk.StringVar(value=str(self.cfg['train'].get('gradient_accumulation_steps', 16)))
        ttk.Entry(basic_frame, textvariable=self.grad_accum_var, width=10).grid(row=3, column=1, sticky='w', pady=5, padx=(10, 0))

        basic_frame.columnconfigure(1, weight=1)

        # === Performance Settings Tab ===
        perf_frame = ttk.Frame(notebook, padding="15")
        notebook.add(perf_frame, text="Performance")

        # Multi-GPU settings
        gpu_frame = ttk.LabelFrame(perf_frame, text="Multi-GPU Settings", padding="10")
        gpu_frame.pack(fill='x', pady=(0, 15))

        self.use_ddp_var = tk.BooleanVar(value=self.cfg['train'].get('use_ddp', True))
        ttk.Checkbutton(gpu_frame, text="Use DistributedDataParallel (DDP)", variable=self.use_ddp_var).pack(anchor='w')

        # Optimization settings
        opt_frame = ttk.LabelFrame(perf_frame, text="Optimization Settings", padding="10")
        opt_frame.pack(fill='x', pady=(0, 15))

        self.use_amp_var = tk.BooleanVar(value=self.cfg['train'].get('use_mixed_precision', True))
        ttk.Checkbutton(opt_frame, text="Mixed Precision Training (AMP)", variable=self.use_amp_var).pack(anchor='w')

        self.use_compile_var = tk.BooleanVar(value=self.cfg['train'].get('use_torch_compile', True))
        ttk.Checkbutton(opt_frame, text="torch.compile() Optimization", variable=self.use_compile_var).pack(anchor='w')

        self.grad_checkpoint_var = tk.BooleanVar(value=self.cfg['train'].get('use_gradient_checkpointing', True))
        ttk.Checkbutton(opt_frame, text="Gradient Checkpointing", variable=self.grad_checkpoint_var).pack(anchor='w')

        # Compile settings
        compile_settings_frame = ttk.Frame(opt_frame)
        compile_settings_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(compile_settings_frame, text="Compile Mode:").pack(side='left')
        self.compile_mode_var = tk.StringVar(value=self.cfg['train'].get('compile_mode', 'default'))
        compile_combo = ttk.Combobox(compile_settings_frame, textvariable=self.compile_mode_var,
                                   values=['default', 'reduce-overhead', 'max-autotune'], width=15)
        compile_combo.pack(side='left', padx=(10, 0))

        # === Training Mode Tab ===
        mode_frame = ttk.Frame(notebook, padding="15")
        notebook.add(mode_frame, text="Training Mode")

        # Training mode selection
        mode_select_frame = ttk.LabelFrame(mode_frame, text="Layer Training Mode", padding="10")
        mode_select_frame.pack(fill='x', pady=(0, 15))

        self.training_mode = tk.StringVar(value="full")
        ttk.Radiobutton(mode_select_frame, text="Full Model Training", variable=self.training_mode, value="full").pack(anchor='w')
        ttk.Radiobutton(mode_select_frame, text="Global Encoder Only", variable=self.training_mode, value="global_encoder").pack(anchor='w')
        ttk.Radiobutton(mode_select_frame, text="U-Net Only", variable=self.training_mode, value="unet_only").pack(anchor='w')

        # === Monitoring Tab ===
        monitor_frame = ttk.Frame(notebook, padding="15")
        notebook.add(monitor_frame, text="Monitoring")

        # Loss monitoring
        loss_monitor_frame = ttk.LabelFrame(monitor_frame, text="Loss Monitoring", padding="10")
        loss_monitor_frame.pack(fill='x', pady=(0, 15))

        self.loss_explosion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(loss_monitor_frame, text="Loss Explosion Protection", variable=self.loss_explosion_var).pack(anchor='w')

        threshold_frame = ttk.Frame(loss_monitor_frame)
        threshold_frame.pack(fill='x', pady=(5, 0))
        ttk.Label(threshold_frame, text="Explosion Threshold:").pack(side='left')
        self.explosion_threshold_var = tk.StringVar(value="10.0")
        ttk.Entry(threshold_frame, textvariable=self.explosion_threshold_var, width=10).pack(side='left', padx=(10, 0))

        # Save intervals
        intervals_frame = ttk.LabelFrame(monitor_frame, text="Save Intervals", padding="10")
        intervals_frame.pack(fill='x')

        save_frame = ttk.Frame(intervals_frame)
        save_frame.pack(fill='x', pady=2)
        ttk.Label(save_frame, text="Save Interval (steps):").pack(side='left')
        self.save_interval_var = tk.StringVar(value=str(self.cfg['train'].get('save_interval_steps', 25)))
        ttk.Entry(save_frame, textvariable=self.save_interval_var, width=10).pack(side='left', padx=(10, 0))

        log_frame = ttk.Frame(intervals_frame)
        log_frame.pack(fill='x', pady=2)
        ttk.Label(log_frame, text="Log Interval (steps):").pack(side='left')
        self.log_interval_var = tk.StringVar(value=str(self.cfg['train'].get('log_interval', 10)))
        ttk.Entry(log_frame, textvariable=self.log_interval_var, width=10).pack(side='left', padx=(10, 0))

        valid_frame = ttk.Frame(intervals_frame)
        valid_frame.pack(fill='x', pady=2)
        ttk.Label(valid_frame, text="Validation Interval (steps):").pack(side='left')
        self.valid_interval_var = tk.StringVar(value=str(self.cfg['train'].get('valid_interval_steps', 100)))
        ttk.Entry(valid_frame, textvariable=self.valid_interval_var, width=10).pack(side='left', padx=(10, 0))

        # === Control Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(20, 0))

        ttk.Button(button_frame, text="Save & Start Training",
                  command=self.save_and_start, style='Accent.TButton').pack(side='right', padx=(10, 0))
        ttk.Button(button_frame, text="Load Defaults",
                  command=self.load_defaults).pack(side='right')
        ttk.Button(button_frame, text="Cancel",
                  command=self.on_closing).pack(side='left')

    def select_pretrained_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Pretrained Weights",
            filetypes=(("PyTorch Checkpoints", "*.pt *.ckpt"), ("All files", "*.*"))
        )
        if filepath:
            self.pretrained_path_var.set(filepath)

    def load_defaults(self):
        """Load default values from current config."""
        self.pretrained_path_var.set(self.cfg['train'].get('pretrained_path', ''))
        self.epochs_var.set(str(self.cfg['train'].get('epochs', 1000)))
        self.batch_size_var.set(str(self.cfg['train'].get('batch_size', 8)))
        self.grad_accum_var.set(str(self.cfg['train'].get('gradient_accumulation_steps', 16)))
        # Reset other variables to defaults...

    def save_and_start(self):
        """Save configuration and signal ready to start training."""
        try:
            # Update configuration with GUI values
            updated_config = self.cfg.copy()

            # Basic settings
            updated_config['train']['pretrained_path'] = self.pretrained_path_var.get() or None
            updated_config['train']['epochs'] = int(self.epochs_var.get())
            updated_config['train']['batch_size'] = int(self.batch_size_var.get())
            updated_config['train']['gradient_accumulation_steps'] = int(self.grad_accum_var.get())

            # Performance settings
            updated_config['train']['use_ddp'] = self.use_ddp_var.get()
            updated_config['train']['use_mixed_precision'] = self.use_amp_var.get()
            updated_config['train']['use_torch_compile'] = self.use_compile_var.get()
            updated_config['train']['use_gradient_checkpointing'] = self.grad_checkpoint_var.get()
            updated_config['train']['compile_mode'] = self.compile_mode_var.get()
            updated_config['train']['fastboot'] = self.fastboot_var.get()
            updated_config['train']['preload_data_to_ram'] = self.preload_ram_var.get()

            # Training mode
            updated_config['train']['training_mode'] = self.training_mode.get()

            # Monitoring
            updated_config['train']['loss_explosion_protection'] = self.loss_explosion_var.get()
            updated_config['train']['explosion_threshold'] = float(self.explosion_threshold_var.get())
            updated_config['train']['save_interval_steps'] = int(self.save_interval_var.get())
            updated_config['train']['log_interval'] = int(self.log_interval_var.get())
            updated_config['train']['valid_interval_steps'] = int(self.valid_interval_var.get())

            # Save updated config
            with open(CFG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(updated_config, f, sort_keys=False, allow_unicode=True)

            self.final_config = updated_config
            self.config_ready = True

            messagebox.showinfo("Configuration Saved",
                              "Training configuration saved successfully!\n"
                              "The GUI will now close and training will start.")

            self.quit()  # Close the GUI

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your inputs:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")

    def get_config_summary(self):
        """Return a summary of the current configuration for display."""
        if not hasattr(self, 'final_config') or not self.final_config:
            # Fallback to current GUI values
            return {
                "Batch Size": self.batch_size_var.get(),
                "Epochs": self.epochs_var.get(),
                "Mixed Precision": "✅ Enabled" if self.use_amp_var.get() else "❌ Disabled",
                "torch.compile()": "✅ Enabled" if self.use_compile_var.get() else "❌ Disabled",
                "DDP": "✅ Enabled" if self.use_ddp_var.get() else "❌ Disabled",
                "Gradient Checkpointing": "✅ Enabled" if self.grad_checkpoint_var.get() else "❌ Disabled",
                "RAM Preloading": "✅ Enabled" if self.preload_ram_var.get() else "❌ Disabled",
                "Gradient Accumulation": self.grad_accum_var.get(),
                "Training Mode": self.training_mode.get().replace('_', ' ').title(),
                "Pretrained Weights": "✅ Yes" if self.pretrained_path_var.get() else "❌ None"
            }

        # Use saved config
        train_cfg = self.final_config.get('train', {})
        return {
            "Batch Size": train_cfg.get('batch_size', 'N/A'),
            "Epochs": train_cfg.get('epochs', 'N/A'),
            "Mixed Precision": "✅ Enabled" if train_cfg.get('use_mixed_precision') else "❌ Disabled",
            "torch.compile()": "✅ Enabled" if train_cfg.get('use_torch_compile') else "❌ Disabled",
            "DDP": "✅ Enabled" if train_cfg.get('use_ddp') else "❌ Disabled",
            "Gradient Checkpointing": "✅ Enabled" if train_cfg.get('use_gradient_checkpointing') else "❌ Disabled",
            "RAM Preloading": "✅ Enabled" if train_cfg.get('preload_data_to_ram') else "❌ Disabled",
            "Gradient Accumulation": train_cfg.get('gradient_accumulation_steps', 'N/A'),
            "Training Mode": train_cfg.get('training_mode', 'full').replace('_', ' ').title(),
            "Pretrained Weights": "✅ Yes" if train_cfg.get('pretrained_path') else "❌ None"
        }

    def get_updated_config(self):
        """Return the updated configuration dictionary."""
        if hasattr(self, 'final_config') and self.final_config:
            return self.final_config
        else:
            # Fallback: build config from current GUI state
            return {
                'train': {
                    'batch_size': int(self.batch_size_var.get()),
                    'epochs': int(self.epochs_var.get()),
                    'use_mixed_precision': self.use_amp_var.get(),
                    'use_torch_compile': self.use_compile_var.get(),
                    'use_ddp': self.use_ddp_var.get(),
                    'use_gradient_checkpointing': self.grad_checkpoint_var.get(),
                    'preload_data_to_ram': self.preload_ram_var.get(),
                    'gradient_accumulation_steps': int(self.grad_accum_var.get()),
                    'training_mode': self.training_mode.get(),
                    'pretrained_path': self.pretrained_path_var.get() or None,
                    'compile_mode': self.compile_mode_var.get(),
                    'fastboot': self.fastboot_var.get(),
                    'loss_explosion_protection': self.loss_explosion_var.get(),
                    'explosion_threshold': float(self.explosion_threshold_var.get()),
                    'save_interval_steps': int(self.save_interval_var.get()),
                    'log_interval': int(self.log_interval_var.get()),
                    'valid_interval_steps': int(self.valid_interval_var.get()),
                }
            }

# =====================================================================================
# SECTION: MAIN GUI APPLICATION (Legacy - for backward compatibility)
# =====================================================================================

class TrainingApp(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title("AudioSR Trainer - DDP + torch.compile() Edition")
        self.geometry("1000x950")

        # --- Style Configuration ---
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'))
        style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), foreground='white', background='#0078D7')
        style.configure('TCheckbutton', font=('Helvetica', 10))

        # --- Rich Console for terminal output ---
        self.console = Console()

        # --- Main Layout ---
        self.create_widgets()

    def _update_layer_selection_state(self, *args):
        """Enable/disable custom layer selection based on training mode."""
        is_custom = self.training_mode.get() == "custom"
        state = 'normal' if is_custom else 'disabled'
        for child in self.custom_layers_frame.winfo_children():
            if isinstance(child, ttk.Checkbutton):
                child.configure(state=state)

    def create_widgets(self):
        # --- Top control frame ---
        control_frame = ttk.LabelFrame(self, text="Training Control & Optimizations", padding=(15, 10))
        control_frame.pack(fill='x', padx=15, pady=10)

        # Row 0: Pretrained model path
        self.pretrained_path_var = tk.StringVar(value=self.cfg['train'].get('pretrained_path', ''))
        ttk.Label(control_frame, text="Pretrained Weights:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.pretrained_path_var, width=70).grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.select_file).grid(row=0, column=3, padx=5, pady=5)

        # Row 1: Gradient accumulation steps
        self.grad_accum_var = tk.StringVar(value=str(self.cfg['train'].get('gradient_accumulation_steps', 1)))
        ttk.Label(control_frame, text="Gradient Accumulation Steps:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Spinbox(control_frame, from_=1, to=128, textvariable=self.grad_accum_var, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=5)

        # Row 2: Performance Toggles
        perf_frame = ttk.Frame(control_frame)
        perf_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=10)

        self.fastboot_var = tk.BooleanVar(value=self.cfg['train'].get('fastboot', True))
        ttk.Checkbutton(perf_frame, text="Fastboot (Async Loading)", variable=self.fastboot_var).pack(side='left', padx=10)

        self.preload_ram_var = tk.BooleanVar(value=self.cfg['train'].get('preload_data_to_ram', True))
        ttk.Checkbutton(perf_frame, text="Preload All Data to RAM", variable=self.preload_ram_var).pack(side='left', padx=10)

        self.use_ddp_var = tk.BooleanVar(value=self.cfg['train'].get('use_ddp', True))
        ttk.Checkbutton(perf_frame, text="Use DDP (Multi-GPU)", variable=self.use_ddp_var).pack(side='left', padx=10)

        self.use_compile_var = tk.BooleanVar(value=self.cfg['train'].get('use_torch_compile', True))
        ttk.Checkbutton(perf_frame, text="Use torch.compile()", variable=self.use_compile_var).pack(side='left', padx=10)

        self.grad_checkpoint_var = tk.BooleanVar(value=self.cfg['train'].get('use_gradient_checkpointing', True))
        ttk.Checkbutton(perf_frame, text="Use Gradient Checkpointing", variable=self.grad_checkpoint_var).pack(side='left', padx=10)

        # Row 3: Mixed Precision Controls
        amp_frame = ttk.LabelFrame(control_frame, text="Mixed Precision (AMP) Settings", padding=(10, 5))
        amp_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=5)

        self.use_amp_var = tk.BooleanVar(value=self.cfg['train'].get('use_mixed_precision', True))
        ttk.Checkbutton(amp_frame, text="Enable Mixed Precision Training", variable=self.use_amp_var).grid(row=0, column=0, sticky='w', padx=5)

        ttk.Label(amp_frame, text="Initial Scale:").grid(row=0, column=1, sticky='w', padx=10)
        self.amp_init_scale_var = tk.StringVar(value=str(self.cfg['train'].get('amp_init_scale', 65536.0)))
        ttk.Entry(amp_frame, textvariable=self.amp_init_scale_var, width=10).grid(row=0, column=2, padx=5)

        # Row 4: torch.compile() Settings
        compile_frame = ttk.LabelFrame(control_frame, text="torch.compile() Settings", padding=(10, 5))
        compile_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=5)

        ttk.Label(compile_frame, text="Compile Mode:").grid(row=0, column=0, sticky='w', padx=5)
        self.compile_mode_var = tk.StringVar(value=self.cfg['train'].get('compile_mode', 'default'))
        compile_mode_combo = ttk.Combobox(compile_frame, textvariable=self.compile_mode_var, values=['default', 'reduce-overhead', 'max-autotune'], width=15)
        compile_mode_combo.grid(row=0, column=1, padx=5)

        self.compile_fullgraph_var = tk.BooleanVar(value=self.cfg['train'].get('compile_fullgraph', False))
        ttk.Checkbutton(compile_frame, text="Full Graph Mode", variable=self.compile_fullgraph_var).grid(row=0, column=2, sticky='w', padx=10)

        # Row 5: Testing Controls
        test_frame = ttk.LabelFrame(control_frame, text="Quick Testing", padding=(10, 5))
        test_frame.grid(row=5, column=0, columnspan=4, sticky='ew', pady=5)

        ttk.Button(test_frame, text="Quick Model Test", command=self.run_quick_test).grid(row=0, column=0, padx=5)
        self.test_result_var = tk.StringVar(value="No test run yet")
        ttk.Label(test_frame, textvariable=self.test_result_var, width=80).grid(row=0, column=1, padx=10, sticky='w')

        # Advanced Loss Monitoring Frame
        loss_frame = ttk.LabelFrame(control_frame, text="Advanced Loss Monitoring", padding=(10, 5))
        loss_frame.grid(row=6, column=0, columnspan=4, sticky='ew', pady=5)

        # Loss explosion detection
        self.loss_explosion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(loss_frame, text="Loss Explosion Protection", variable=self.loss_explosion_var).grid(row=0, column=0, sticky='w', padx=5)

        ttk.Label(loss_frame, text="Explosion Threshold:").grid(row=0, column=1, sticky='w', padx=5)
        self.explosion_threshold_var = tk.StringVar(value="10.0")
        ttk.Entry(loss_frame, textvariable=self.explosion_threshold_var, width=8).grid(row=0, column=2, padx=5)

        # Early stopping
        self.early_stopping_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(loss_frame, text="Early Stopping", variable=self.early_stopping_var).grid(row=1, column=0, sticky='w', padx=5)

        ttk.Label(loss_frame, text="Patience (steps):").grid(row=1, column=1, sticky='w', padx=5)
        self.patience_var = tk.StringVar(value="1000")
        ttk.Entry(loss_frame, textvariable=self.patience_var, width=8).grid(row=1, column=2, padx=5)

        control_frame.columnconfigure(1, weight=1)

        # Row 7: Layer Selection Frame
        layer_frame = ttk.LabelFrame(control_frame, text="Selective Layer Training", padding=(10, 5))
        layer_frame.grid(row=7, column=0, columnspan=4, sticky='ew', pady=10)

        # Training mode selection
        self.training_mode = tk.StringVar(value="full")
        ttk.Radiobutton(layer_frame, text="Full Model Training", variable=self.training_mode, value="full").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="Global Encoder Only", variable=self.training_mode, value="global_encoder").grid(row=0, column=1, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="U-Net Only", variable=self.training_mode, value="unet_only").grid(row=0, column=2, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="Custom Selection", variable=self.training_mode, value="custom").grid(row=1, column=0, sticky='w', padx=5)

        # Custom layer selection (initially disabled)
        self.custom_layers_frame = ttk.Frame(layer_frame)
        self.custom_layers_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=5)

        # Layer checkboxes
        self.layer_vars = {}
        layer_options = [
            ("diffusion_model.global_encoder", "Global Audio Encoder"),
            ("diffusion_model.input_blocks", "U-Net Input Blocks"),
            ("diffusion_model.middle_block", "U-Net Middle Block"),
            ("diffusion_model.output_blocks", "U-Net Output Blocks"),
            ("first_stage_model", "VAE Encoder/Decoder"),
            ("cond_stage_model", "Conditioning Model")
        ]

        for i, (layer_key, layer_name) in enumerate(layer_options):
            var = tk.BooleanVar(value=True)
            self.layer_vars[layer_key] = var
            cb = ttk.Checkbutton(self.custom_layers_frame, text=layer_name, variable=var, state='disabled')
            cb.grid(row=i//2, column=i%2, sticky='w', padx=10, pady=2)

        # Update custom layer frame state based on training mode
        self.training_mode.trace('w', self._update_layer_selection_state)

        # Row 8: Start Button
        self.btn_t = ttk.Button(control_frame, text="Start Training", command=self.start_training_thread, style='Accent.TButton', padding=10)
        self.btn_t.grid(row=8, column=0, columnspan=4, pady=15)

        # --- Plotting Frame ---
        plot_frame = ttk.LabelFrame(self, text="Loss Curve", padding=(15, 10))
        plot_frame.pack(fill='both', expand=True, padx=15, pady=10)

        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.train_loss_data = {'steps': [], 'losses': []}
        self.valid_loss_data = {'steps': [], 'losses': []}
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_title("Training & Validation Loss", fontsize=16)
        self.ax.set_xlabel("Steps (Optimizer Updates)", fontsize=12)
        self.ax.set_ylabel("Loss", fontsize=12)
        self.train_line, = self.ax.plot([], [], 'o-', label='Train Loss', alpha=0.7, markersize=4)
        self.valid_line, = self.ax.plot([], [], 's-', label='Validation Loss', markersize=7, linewidth=2.5)
        self.ax.legend()
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def run_quick_test(self):
        """Run a quick model test without starting full training."""
        try:
            self.test_result_var.set("Running test...")

            # Load minimal config
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                test_cfg = yaml.safe_load(f)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Create model
            model_config = test_cfg['model'].copy()
            model = instantiate_from_config(model_config).to(device)

            # Apply torch.compile() if enabled
            if self.use_compile_var.get():
                self.console.print("[cyan]Compiling model with torch.compile()...[/cyan]")
                compile_mode = self.compile_mode_var.get()
                fullgraph = self.compile_fullgraph_var.get()
                model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph)
                self.console.print(f"[green]Model compiled with mode='{compile_mode}', fullgraph={fullgraph}[/green]")

            # Create minimal test batch
            batch_size = 2  # Small batch for testing
            test_batch = {
                'lowpass_mel': torch.randn(batch_size, 1, 256, 1024).to(device),
                'highpass_mel': torch.randn(batch_size, 1, 256, 1024).to(device)
            }

            # Run test
            success, message, time_taken = quick_model_test(
                model, test_batch, device,
                use_amp=self.use_amp_var.get(),
                use_compile=self.use_compile_var.get()
            )

            result_text = f"{message} | Time: {time_taken:.3f}s"
            self.test_result_var.set(result_text)

            if success:
                self.console.print(f"[green]{result_text}[/green]")
            else:
                self.console.print(f"[red]{result_text}[/red]")

        except Exception as e:
            error_msg = f"❌ Test failed: {str(e)}"
            self.test_result_var.set(error_msg)
            self.console.print(f"[red]{error_msg}[/red]")

    def _setup_selective_training(self, model, config):
        """Setup selective training - prepares parameters for selective optimization."""
        training_mode = self.training_mode.get()

        # Keep all parameters with requires_grad=True for forward pass
        for param in model.parameters():
            param.requires_grad = True

        # Store selected parameters for later optimizer creation
        if training_mode == "full":
            self.selected_params = list(model.parameters())
            if is_main_process():
                self.console.print("[green]Full model training enabled[/green]")

        elif training_mode == "global_encoder":
            # Only train Global Encoder parameters
            self.selected_params = []
            if hasattr(model.model, 'diffusion_model') and hasattr(model.model.diffusion_model, 'global_encoder'):
                self.selected_params = list(model.model.diffusion_model.global_encoder.parameters())
                if is_main_process():
                    self.console.print("[green]Global Encoder only training enabled[/green]")
            else:
                if is_main_process():
                    self.console.print("[yellow]Global Encoder not found, falling back to full training[/yellow]")
                self.selected_params = list(model.parameters())

        elif training_mode == "unet_only":
            # Train U-Net but not Global Encoder
            self.selected_params = []
            for name, param in model.named_parameters():
                if 'global_encoder' not in name:
                    self.selected_params.append(param)
            if is_main_process():
                self.console.print("[green]U-Net only training enabled (Global Encoder excluded)[/green]")

        elif training_mode == "custom":
            # Custom layer selection
            selected_layers = [key for key, var in self.layer_vars.items() if var.get()]
            self.selected_params = []
            for name, param in model.named_parameters():
                for layer_key in selected_layers:
                    if layer_key in name:
                        self.selected_params.append(param)
                        break
            if is_main_process():
                self.console.print(f"[green]Custom training enabled for: {selected_layers}[/green]")

    def _get_trainable_parameters(self, model):
        """Get list of selected parameters for training."""
        if not hasattr(self, 'selected_params'):
            # Fallback to all parameters if setup wasn't called
            self.selected_params = list(model.parameters())

        trainable_params = self.selected_params

        # Log parameter breakdown for debugging (only on main process)
        if is_main_process():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in trainable_params)
            excluded_count = total_params - trainable_count

            self.console.print(f"[dim]Parameter breakdown:[/dim]")
            self.console.print(f"[dim]  Total: {total_params:,}[/dim]")
            self.console.print(f"[dim]  Training: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)[/dim]")
            self.console.print(f"[dim]  Excluded: {excluded_count:,} ({100*excluded_count/total_params:.1f}%)[/dim]")

        return trainable_params

    def update_plot(self, msg_type, step, loss_value):
        """Thread-safe method to update plot data."""
        if msg_type == 'train':
            self.train_loss_data['steps'].append(step)
            self.train_loss_data['losses'].append(float(loss_value))
            self.train_line.set_data(self.train_loss_data['steps'], self.train_loss_data['losses'])
        elif msg_type == 'valid':
            self.valid_loss_data['steps'].append(step)
            self.valid_loss_data['losses'].append(float(loss_value))
            self.valid_line.set_data(self.valid_loss_data['steps'], self.valid_loss_data['losses'])

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def select_file(self):
        filepath = filedialog.askopenfilename(title="Select Pretrained Weights", filetypes=(("PyTorch Checkpoints", "*.pt *.ckpt"), ("All files", "*.*")))
        if filepath:
            self.pretrained_path_var.set(filepath)

    def start_training_thread(self):
        self.btn_t.config(state=tk.DISABLED, text="Training...")

        # Clear previous plot data
        self.train_loss_data = {'steps': [], 'losses': []}
        self.valid_loss_data = {'steps': [], 'losses': []}
        self.init_plot()

        # Check if DDP is enabled
        if self.use_ddp_var.get() and torch.cuda.device_count() > 1:
            # Launch DDP training
            self.console.print("[yellow]⚠️ DDP training requires process launcher. Use the launch script instead.[/yellow]")
            messagebox.showwarning("DDP Training",
                "For DDP training, please use:\n\n"
                "python -m torch.distributed.launch --nproc_per_node=2 trainMGPU_DDP_Compile.py --distributed\n\n"
                "or\n\n"
                "torchrun --nproc_per_node=2 trainMGPU_DDP_Compile.py --distributed")
            self.btn_t.config(state=tk.NORMAL, text="Start Training")
            return

        # Start single-GPU training in a new thread to keep the GUI responsive
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def on_training_finish(self, final_path):
        """Callback to run on the main thread after training finishes."""
        self.btn_t.config(state=tk.NORMAL, text="Start Training")
        if is_main_process():
            self.console.print(f"[bold green]✅ Training task finished. Final model saved to: {final_path}[/bold green]")
            messagebox.showinfo("Training Complete", f"Training has finished!\nFinal model saved to:\n{final_path}")

    def _background_loader(self, dataset, buffer, initial_chunk_event):
        """Helper function to load data in a background thread."""
        num_files = len(dataset.files)
        initial_chunk_size = math.ceil(num_files * 0.1)

        for i in range(num_files):
            item = dataset._load_item(i)
            if item:
                buffer.append(item)
            if i == initial_chunk_size - 1:
                initial_chunk_event.set() # Signal that the first 10% is ready

        if not initial_chunk_event.is_set():
            initial_chunk_event.set()

    def run_training_logic(self):
        """The main training loop, executed in a separate thread."""
        try:
            # --- 1. Update and save config ---
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                current_cfg = yaml.safe_load(f)

            # Update config with GUI values
            current_cfg['train']['pretrained_path'] = self.pretrained_path_var.get() or None
            current_cfg['train']['gradient_accumulation_steps'] = int(self.grad_accum_var.get())
            current_cfg['train']['fastboot'] = self.fastboot_var.get()
            current_cfg['train']['preload_data_to_ram'] = self.preload_ram_var.get()
            current_cfg['train']['use_ddp'] = self.use_ddp_var.get()
            current_cfg['train']['use_torch_compile'] = self.use_compile_var.get()
            current_cfg['train']['use_gradient_checkpointing'] = self.grad_checkpoint_var.get()
            current_cfg['train']['use_mixed_precision'] = self.use_amp_var.get()
            current_cfg['train']['amp_init_scale'] = float(self.amp_init_scale_var.get())
            current_cfg['train']['compile_mode'] = self.compile_mode_var.get()
            current_cfg['train']['compile_fullgraph'] = self.compile_fullgraph_var.get()
            current_cfg['train']['loss_explosion_protection'] = self.loss_explosion_var.get()
            current_cfg['train']['explosion_threshold'] = float(self.explosion_threshold_var.get())
            current_cfg['train']['early_stopping'] = self.early_stopping_var.get()
            current_cfg['train']['patience'] = int(self.patience_var.get())

            with open(CFG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(current_cfg, f, sort_keys=False, allow_unicode=True)

            if is_main_process():
                self.console.rule("[bold cyan]🚀 Starting DDP + torch.compile() training[/bold cyan]")

            # --- 2. Setup ---
            set_seed(current_cfg['experiment']['seed'])
            device = f'cuda:{get_rank()}' if torch.cuda.is_available() else 'cpu'
            accumulation_steps = current_cfg['train']['gradient_accumulation_steps']
            use_grad_checkpoint = current_cfg['train']['use_gradient_checkpointing']
            use_amp = current_cfg['train']['use_mixed_precision']
            use_compile = current_cfg['train']['use_torch_compile']

            if is_main_process():
                self.console.print(f"Using device: [bold cyan]{device}[/]")
                self.console.print(f"World size: [bold cyan]{get_world_size()}[/]")
                self.console.print(f"Mixed Precision: [bold {'green' if use_amp else 'red'}]{use_amp}[/]")
                self.console.print(f"torch.compile(): [bold {'green' if use_compile else 'red'}]{use_compile}[/]")
                self.console.print(f"Gradient Checkpointing: [bold {'green' if use_grad_checkpoint else 'red'}]{use_grad_checkpoint}[/]")

            # Initialize gradient scaler for mixed precision
            scaler = None
            if use_amp and device.startswith('cuda'):
                scaler = GradScaler(
                    init_scale=current_cfg['train']['amp_init_scale'],
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000
                )
                if is_main_process():
                    self.console.print(f"[green]🔥 GradScaler initialized with scale: {scaler.get_scale():.0e}[/green]")

            # --- 3. DataLoaders with Enhanced Memory Management ---
            use_fastboot = current_cfg['train'].get('fastboot', False)
            preload_to_ram = current_cfg['train'].get('preload_data_to_ram', True)

            if is_main_process():
                self.console.print(f"Data Loading Strategy:")
                self.console.print(f"  Fastboot: {use_fastboot}")
                self.console.print(f"  Preload to RAM: {preload_to_ram}")
                if preload_to_ram:
                    self.console.print(f"  [green]💾 All data will be loaded into your 256GB RAM[/green]")

            # Enhanced dataset loading with smart memory management
            train_set_full = PairedMelDataset(split='train', **current_cfg['data'], preload_to_ram=preload_to_ram)
            valid_set = PairedMelDataset(split='valid', **current_cfg['data'], preload_to_ram=True)

            # Use DistributedSampler if DDP is initialized
            train_sampler = DistributedSampler(train_set_full) if dist.is_initialized() else None
            valid_sampler = DistributedSampler(valid_set, shuffle=False) if dist.is_initialized() else None

            # Optimized DataLoader settings for large RAM systems
            num_workers = min(current_cfg['train'].get('num_workers', 8), 16)  # Cap workers for RAM efficiency
            if preload_to_ram:
                num_workers = min(num_workers, 4)  # Fewer workers needed when data is in RAM
                if is_main_process():
                    self.console.print(f"  Workers optimized for RAM loading: {num_workers}")

            train_loader = DataLoader(
                train_set_full,
                batch_size=current_cfg['train']['batch_size'],
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,  # Keep workers alive for faster epochs
                prefetch_factor=2 if preload_to_ram else 4  # Reduce prefetch when data is in RAM
            )

            valid_loader = DataLoader(
                valid_set,
                batch_size=current_cfg['train']['batch_size'],
                sampler=valid_sampler,
                shuffle=False,
                num_workers=min(num_workers, 2),  # Fewer workers for validation
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )

            # --- 4. Model Setup ---
            model_config = current_cfg['model'].copy()

            # Gradient checkpointing configuration
            if use_grad_checkpoint:
                if 'unet_config' in model_config['params'] and 'params' in model_config['params']['unet_config']:
                    model_config['params']['unet_config']['params']['use_checkpoint'] = True

            model = instantiate_from_config(model_config).to(device)

            # Load pretrained weights
            if current_cfg['train'].get('pretrained_path'):
                load_pretrained(model, current_cfg['train']['pretrained_path'], device, self.console)

            # Setup selective training
            self._setup_selective_training(model, current_cfg)
            trainable_params = self._get_trainable_parameters(model)

            if not trainable_params:
                raise RuntimeError("No trainable parameters found! Check your layer selection.")

            # Apply torch.compile() before DDP wrapping
            if use_compile:
                if is_main_process():
                    self.console.print("[cyan]Compiling model with torch.compile()...[/cyan]")
                compile_mode = current_cfg['train']['compile_mode']
                fullgraph = current_cfg['train']['compile_fullgraph']
                model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph)
                if is_main_process():
                    self.console.print(f"[green]Model compiled with mode='{compile_mode}', fullgraph={fullgraph}[/green]")

            # Wrap with DDP if distributed training is initialized
            if dist.is_initialized():
                model = DDP(model, device_ids=[get_rank()], find_unused_parameters=True)
                if is_main_process():
                    self.console.print(f"[green]✅ Model wrapped with DDP on {get_world_size()} GPUs[/green]")

            # Get unwrapped model for EMA and parameter access
            unwrapped_model = model.module if hasattr(model, 'module') else model

            # --- 5. Optimizer, Scheduler, EMA ---
            opt = torch.optim.AdamW(
                trainable_params,
                lr=current_cfg['model']['params']['base_learning_rate'],
                betas=tuple(current_cfg['train']['betas']),
                weight_decay=current_cfg['train']['weight_decay']
            )

            num_batches = len(train_loader)
            max_steps = (current_cfg['train']['epochs'] * num_batches) // accumulation_steps

            sched = WarmupCosine(
                opt,
                current_cfg['model']['params']['base_learning_rate'],
                warmup=current_cfg['train']['warmup_steps'],
                max_steps=max_steps
            )

            # Initialize EMA
            ema = EMA(model, decay=current_cfg['train']['ema_decay'])

            # Initialize loss monitor
            loss_monitor = AdvancedLossMonitor(
                explosion_threshold=current_cfg['train'].get('explosion_threshold', 10.0),
                patience=current_cfg['train'].get('patience', 1000) if current_cfg['train'].get('early_stopping', False) else float('inf')
            )

            # --- 6. Training Loop ---
            best_val_loss = float('inf')
            gstep = 0
            outp = os.path.join(current_cfg['experiment']['out_dir'], 'checkpoints')
            ensure_dir(outp)

            model.train()

            if is_main_process():
                self.console.print(f"[green]🚀 Starting training for {current_cfg['train']['epochs']} epochs[/green]")

            for ep in range(current_cfg['train']['epochs']):
                if train_sampler:
                    train_sampler.set_epoch(ep)

                for batch_idx, batch in enumerate(train_loader):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                    # Forward pass with mixed precision
                    if use_amp:
                        with autocast():
                            loss, _ = model(batch)
                    else:
                        loss, _ = model(batch)

                    # Average loss across GPUs for DDP
                    if dist.is_initialized():
                        loss = loss.mean()

                    raw_loss = loss.item()
                    loss = loss / accumulation_steps

                    # Backward pass with gradient scaling
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (batch_idx + 1) % accumulation_steps == 0:
                        # Gradient clipping and optimizer step
                        if use_amp:
                            scaler.unscale_(opt)
                            trainable_params_for_clipping = [p for p in trainable_params if p.grad is not None]
                            if trainable_params_for_clipping:
                                torch.nn.utils.clip_grad_norm_(trainable_params_for_clipping, 1.0)
                            scaler.step(opt)
                            scaler.update()
                        else:
                            trainable_params_for_clipping = [p for p in trainable_params if p.grad is not None]
                            if trainable_params_for_clipping:
                                torch.nn.utils.clip_grad_norm_(trainable_params_for_clipping, 1.0)
                            opt.step()

                        sched.step()
                        ema.update(model)
                        opt.zero_grad()
                        gstep += 1

                        # Logging and validation (only on main process)
                        if is_main_process() and gstep % current_cfg['train']['log_interval'] == 0:
                            lr = opt.param_groups[0]['lr']
                            scale_info = f", Scale = {scaler.get_scale():.0e}" if scaler else ""
                            self.console.print(f"[green]Step {gstep}: Loss = {raw_loss:.4f}, LR = {lr:.2e}{scale_info}[/green]")
                            self.after(0, self.update_plot, 'train', gstep, raw_loss)

                        # Validation and checkpointing (only on main process)
                        if is_main_process() and gstep % current_cfg['train']['valid_interval_steps'] == 0:
                            model.eval()
                            val_losses = []
                            for val_batch in valid_loader:
                                val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                                with torch.no_grad():
                                    if use_amp:
                                        with autocast():
                                            val_loss, _ = model(val_batch)
                                    else:
                                        val_loss, _ = model(val_batch)
                                val_losses.append(val_loss.item())

                            avg_val_loss = sum(val_losses) / len(val_losses)
                            self.console.print(f"📊 Validation @ step {gstep}: loss = {avg_val_loss:.4f}")
                            self.after(0, self.update_plot, 'valid', gstep, avg_val_loss)
                            model.train()

                            # Save best model
                            if avg_val_loss < best_val_loss:
                                best_val_loss = avg_val_loss
                                path = os.path.join(outp, f'best_step_{gstep}.pt')
                                torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, path)
                                self.console.print(f"💾 [bold magenta]Saved best model to {path}[/bold magenta]")

                        # Regular checkpointing (only on main process)
                        if is_main_process() and gstep % current_cfg['train']['save_interval_steps'] == 0:
                            path = os.path.join(outp, f'step_{gstep}.pt')
                            torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, path)
                            self.console.print(f"💾 [cyan]Saved checkpoint to {path}[/cyan]")

            # Save final model (only on main process)
            if is_main_process():
                final_p = os.path.join(outp, 'final_ema.pt')
                torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, final_p)

                if scaler:
                    self.console.print(f"[cyan]🎯 Final gradient scale: {scaler.get_scale():.0e}[/cyan]")

                self.after(0, self.on_training_finish, final_p)

        except Exception as e:
            if is_main_process():
                error_msg = str(e)
                self.console.print_exception()
                self.after(0, lambda: self.btn_t.config(state=tk.NORMAL, text="Start Training"))
                self.after(0, lambda msg=error_msg: messagebox.showerror("Training Error", f"Training failed: {msg}"))

# =====================================================================================
# SECTION: HEADLESS TRAINING LOGIC
# =====================================================================================

def run_headless_training(config, console):
    """
    Run training without GUI. Used for both single-GPU and DDP modes.
    """
    try:
        # Setup
        set_seed(config['experiment']['seed'])

        if dist.is_initialized():
            device = f'cuda:{get_rank()}'
            world_size = get_world_size()
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            world_size = 1

        if is_main_process():
            console.print(f"[bold cyan]🚀 Starting headless training[/bold cyan]")
            console.print(f"Device: {device}, World size: {world_size}")

        # Fixed: Enable all optimizations properly
        use_ddp = config['train'].get('use_ddp', False) and dist.is_initialized()
        use_amp = config['train'].get('use_mixed_precision', True)  # Default enabled
        use_compile = config['train'].get('use_torch_compile', True)  # Default enabled
        use_grad_checkpoint = config['train'].get('use_gradient_checkpointing', True)  # Default enabled
        preload_to_ram = config['train'].get('preload_data_to_ram', True)

        if is_main_process():
            console.print(f"RAM Preloading: {'Enabled' if preload_to_ram else 'Disabled'}")
            if preload_to_ram:
                console.print(f"[green]💾 Optimized for 256GB RAM capacity[/green]")
            console.print(f"Mixed Precision: [bold {'green' if use_amp else 'red'}]{use_amp}[/]")
            console.print(f"torch.compile(): [bold {'green' if use_compile else 'red'}]{use_compile}[/]")
            console.print(f"DDP: [bold {'green' if use_ddp else 'red'}]{use_ddp}[/]")
            console.print(f"Gradient Checkpointing: [bold {'green' if use_grad_checkpoint else 'red'}]{use_grad_checkpoint}[/]")

        # Load model config
        if is_main_process():
            console.print("Loading model...")

        model_config = config['model'].copy()
        model = instantiate_from_config(model_config).to(device)

        # Enable gradient checkpointing
        if use_grad_checkpoint:
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model.gradient_checkpointing = True
            if is_main_process():
                console.print("✅ Gradient checkpointing enabled")

        # Apply torch.compile
        if use_compile:
            if is_main_process():
                console.print("Applying torch.compile()...")
            compile_mode = config['train'].get('compile_mode', 'default')
            model = torch.compile(model, mode=compile_mode)

        # Wrap with DDP
        if use_ddp and dist.is_initialized():
            model = DDP(model, device_ids=[get_rank()], find_unused_parameters=True)
            if is_main_process():
                console.print(f"Model wrapped with DDP on {world_size} GPUs")

        # Get unwrapped model
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Setup optimizer with minimal parameters for testing
        trainable_params = list(unwrapped_model.parameters())
        opt = torch.optim.AdamW(
            trainable_params,
            lr=config['model']['params']['base_learning_rate'],
            betas=tuple(config['train']['betas']),
            weight_decay=config['train']['weight_decay']
        )

        # Initialize scaler for mixed precision
        scaler = GradScaler(init_scale=config['train'].get('amp_init_scale', 65536.0)) if use_amp and device.startswith('cuda') else None

        # Initialize EMA
        ema = EMA(model, decay=config['train']['ema_decay'])

        if is_main_process():
            console.print("✅ Model setup complete")

        # Ensure checkpoint directory exists
        outp = os.path.join(config['experiment']['out_dir'], 'checkpoints')
        ensure_dir(outp)

        if is_main_process():
            console.print(f"Checkpoints will be saved to: {outp}")
            console.print("🎯 Ready for full training execution")

        # === FIXED: ADD COMPLETE TRAINING LOOP ===

        # Create dataset and dataloader
        if is_main_process():
            console.print("Creating dataset...")

        dataset = PairedMelDataset(
            root=config['data']['dataset_root'],
            categories=config['data']['categories'],
            sample_rate=config['data']['sample_rate'],
            segment_seconds=config['data']['segment_seconds'],
            high_dir_name=config['data']['high_dir_name'],
            low_dir_name=config['data']['low_dir_name'],
            valid_ratio=config['data']['valid_ratio'],
            split_seed=config['data']['split_seed'],
            n_fft=config['data']['n_fft'],
            hop_length=config['data']['hop_length'],
            win_length=config['data']['win_length'],
            n_mels=config['data']['n_mels'],
            fmin=config['data']['fmin'],
            fmax=config['data']['fmax'],
            blank_ratio_max=config['data']['blank_ratio_max'],
            blank_hop_seconds=config['data']['blank_hop_seconds'],
            blank_thr=config['data']['blank_thr'],
            preload_to_ram=preload_to_ram
        )

        # Create data loaders
        sampler = DistributedSampler(dataset) if use_ddp else None
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=config['train']['num_workers'],
            pin_memory=True,
            drop_last=True
        )

        if is_main_process():
            console.print(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")

        # Load pretrained weights
        pretrained_path = config['train'].get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            if is_main_process():
                console.print(f"Loading pretrained weights from: {pretrained_path}")
            missing_keys, unexpected_keys = unwrapped_model.load_checkpoint(pretrained_path, strict=False)
            if is_main_process():
                console.print(f"Loaded checkpoint: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=config['train']['epochs'] * len(dataloader),
            eta_min=config['train']['lr'] * config['train']['min_lr_ratio']
        )

        if is_main_process():
            console.print("🚀 Starting training loop...")

        # Training loop
        global_step = 0
        best_loss = float('inf')

        for epoch in range(config['train']['epochs']):
            if use_ddp and sampler:
                sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TimeElapsedColumn(),
                console=console,
                disable=not is_main_process()
            ) as progress:
                task_id = progress.add_task(f"Epoch {epoch+1}", total=len(dataloader), loss=0.0)

                for batch_idx, batch in enumerate(dataloader):
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    opt.zero_grad()

                    # Forward pass with optional AMP
                    if use_amp and scaler:
                        with autocast():
                            loss, log_dict = model(batch)
                            loss = loss / config['train']['gradient_accumulation_steps']
                        scaler.scale(loss).backward()
                    else:
                        loss, log_dict = model(batch)
                        loss = loss / config['train']['gradient_accumulation_steps']
                        loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
                        if use_amp and scaler:
                            scaler.step(opt)
                            scaler.update()
                        else:
                            opt.step()

                        scheduler.step()
                        ema.update()
                        global_step += 1

                    # Update progress
                    current_loss = loss.item() * config['train']['gradient_accumulation_steps']
                    epoch_loss += current_loss

                    if is_main_process():
                        progress.update(task_id, advance=1, loss=current_loss)

                    # Save checkpoint
                    if global_step % config['train']['save_interval_steps'] == 0 and is_main_process():
                        checkpoint_path = os.path.join(outp, f"checkpoint_step_{global_step}.pt")
                        torch.save({
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'ema_state_dict': ema.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'loss': current_loss,
                            'config': config
                        }, checkpoint_path)
                        console.print(f"💾 Saved checkpoint: {checkpoint_path}")

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            if is_main_process():
                console.print(f"📊 Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

                # Save best model
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_model_path = os.path.join(outp, "best_model.pt")
                    torch.save({
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                        'config': config
                    }, best_model_path)
                    console.print(f"🏆 New best model saved: {best_model_path}")

        if is_main_process():
            console.print("[green]🎉 Training completed successfully![/green]")

        return True

    except Exception as e:
        if is_main_process():
            console.print(f"[red]❌ Training failed: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
        return False

# =====================================================================================
# SECTION: MAIN ENTRY POINTS
# =====================================================================================

def main_distributed(args):
    """Main function for distributed training."""
    # Fixed: Handle missing environment variables gracefully
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    # If no distributed environment, fall back to single GPU
    if 'LOCAL_RANK' not in os.environ:
        print("⚠️  No distributed environment detected. Running on single GPU...")
        # Set up single GPU environment
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'

    setup_ddp(rank, world_size)

    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ensure_project_dirs(config)

        console = Console()
        success = run_headless_training(config, console)

        if success and is_main_process():
            console.print("[green]🎉 DDP training ready. Execute full training as needed.[/green]")

    finally:
        cleanup_ddp()

def main_gui_config(args):
    """Main function for GUI configuration mode."""
    if not GUI_AVAILABLE:
        print("❌ GUI components not available. Please install tkinter and matplotlib.")
        sys.exit(1)

    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ensure_project_dirs(config)

        # Create configuration GUI
        config_gui = ConfigurationGUI(config)
        config_gui.mainloop()

        # After GUI closes, check if config was saved and user wants to proceed
        if config_gui.config_ready:
            console = Console()
            console.print("[bold green]✅ Configuration saved from GUI[/bold green]")
            console.print("[bold cyan]🔄 Switching to terminal mode for training...[/bold cyan]")

            # Show user the configured settings
            print("\n" + "="*60)
            print("🎛️  TRAINING CONFIGURATION SUMMARY")
            print("="*60)

            config_summary = config_gui.get_config_summary()
            for key, value in config_summary.items():
                print(f"   {key}: {value}")

            print("="*60)

            # Ask for final confirmation
            response = input("\n💡 Start training with these settings? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                console.print("[bold green]🚀 Starting terminal-based training...[/bold green]")

                # Reload updated config and start training
                with open(CFG_PATH, 'r', encoding='utf-8') as f:
                    updated_config = yaml.safe_load(f)

                # Update config with GUI settings
                updated_config.update(config_gui.get_updated_config())

                # Run complete training (not just setup)
                success = run_headless_training(updated_config, console)

                if success:
                    console.print("[green]🎉 Training completed successfully![/green]")
                else:
                    console.print("[red]❌ Training failed. Check logs above.[/red]")
            else:
                console.print("[yellow]⏹️  Training cancelled by user.[/yellow]")
        else:
            print("❌ Configuration was not saved. Exiting.")

    except FileNotFoundError:
        print(f"❌ Config file 'config.yaml' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def main_legacy_gui(args):
    """Main function for legacy full GUI mode."""
    if not GUI_AVAILABLE:
        print("❌ GUI components not available.")
        sys.exit(1)

    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ensure_project_dirs(config)
        app = TrainingApp(config)
        app.mainloop()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AudioSR Training Script with GUI Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
🎛️  USAGE MODES:

  🖥️  GUI Mode (Recommended):
    python trainMGPU_DDP_Compile.py --gui
    └─ Shows configuration UI, then switches to terminal training

  ⚡ Quick Terminal Mode:
    python trainMGPU_DDP_Compile.py --headless
    └─ Starts training immediately with config.yaml settings

  🔗 Distributed Training:
    torchrun --nproc_per_node=2 trainMGPU_DDP_Compile.py --distributed
    └─ Multi-GPU distributed training

  📊 Legacy GUI Mode:
    python trainMGPU_DDP_Compile.py --legacy-gui
    └─ Full GUI with embedded training (old behavior)

✨ Features: Mixed Precision, torch.compile(), Gradient Checkpointing, DDP
        '''
    )

    parser.add_argument('--gui', action='store_true',
                       help='🖥️  Launch configuration GUI then switch to terminal training (Recommended)')
    parser.add_argument('--headless', action='store_true',
                       help='⚡ Start training immediately with current config.yaml')
    parser.add_argument('--distributed', action='store_true',
                       help='🔗 Run in distributed mode (use with torchrun)')
    parser.add_argument('--legacy-gui', action='store_true',
                       help='📊 Run with full legacy GUI (for compatibility)')

    args = parser.parse_args()

    # Determine mode
    if args.distributed:
        main_distributed(args)
    elif args.gui:
        main_gui_config(args)
    elif args.legacy_gui:
        main_legacy_gui(args)
    elif args.headless:
        # Explicit headless mode
        try:
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            ensure_project_dirs(config)

            console = Console()
            console.print("[bold yellow]⚡ Starting headless training with current config...[/bold yellow]")
            success = run_headless_training(config, console)

            if success:
                console.print("[green]🎉 Training completed![/green]")

        except FileNotFoundError:
            print(f"❌ Config file 'config.yaml' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Default: Show helpful message and suggest GUI mode
        print("🎛️  AudioSR Training Script")
        print("=" * 50)
        print()
        print("💡 Choose your training mode:")
        print()
        print("   🖥️  GUI Mode (Recommended):")
        print("       python trainMGPU_DDP_Compile.py --gui")
        print("       └─ Configure settings in GUI, then train in terminal")
        print()
        print("   ⚡ Quick Terminal Mode:")
        print("       python trainMGPU_DDP_Compile.py --headless")
        print("       └─ Start training immediately")
        print()
        print("   📖 More Options:")
        print("       python trainMGPU_DDP_Compile.py --help")
        print()

        # Ask user what they want to do
        print("❓ What would you like to do?")
        print("   [1] Launch GUI configuration (recommended)")
        print("   [2] Start headless training now")
        print("   [3] Show help")
        print()

        choice = input("Enter choice [1-3]: ").strip()

        if choice == '1' or choice == '':
            print("🖥️  Launching GUI configuration...")
            main_gui_config(args)
        elif choice == '2':
            print("⚡ Starting headless training...")
            try:
                with open(CFG_PATH, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                ensure_project_dirs(config)

                console = Console()
                success = run_headless_training(config, console)

                if success:
                    console.print("[green]🎉 Training completed![/green]")

            except FileNotFoundError:
                print(f"❌ Config file 'config.yaml' not found.")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error: {e}")
                sys.exit(1)
        elif choice == '3':
            parser.print_help()
        else:
            print("❌ Invalid choice. Use --help for options.")
            sys.exit(1)