import os
import sys
import yaml
import math
import threading
import argparse
from typing import Optional

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Plotting Imports ---
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Training Imports ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    Compatible with DataParallel.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        # Handle DataParallel wrapped models
        actual_model = model.module if hasattr(model, 'module') else model
        # Create a shadow copy of parameters in full FP32 precision
        self.shadow = {
            k: p.data.clone().to(dtype=torch.float32)
            for k, p in actual_model.named_parameters() if p.requires_grad
        }

    def update(self, model):
        # Handle DataParallel wrapped models
        actual_model = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            for k, p in actual_model.named_parameters():
                if p.requires_grad and k in self.shadow:
                    p_fp32 = p.data.to(dtype=torch.float32)
                    self.shadow[k] = (1.0 - self.decay) * p_fp32 + self.decay * self.shadow[k]

class AdvancedLossMonitor:
    """
    Advanced loss monitoring with explosion detection, adaptive LR, and early stopping.
    Enhanced for mixed precision and DataParallel compatibility.
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

        console.print(f"[yellow]📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e} (reduction #{self.lr_reductions})[/yellow]")

        if new_lr <= self.min_lr:
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
        console.print("[yellow]No pretrained path found. Starting from scratch.[/yellow]")
        return

    console.print(f"[cyan]Loading pretrained weights from:[/] [default]{path}[/]")
    ckpt = torch.load(path, map_location=device)

    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Handle DataParallel wrapped models
    actual_model = model.module if hasattr(model, 'module') else model
    missing, unexpected = actual_model.load_state_dict(sd, strict=False)

    console.print(f"[green]Weights loaded.[/] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        console.print(f"  [dim]Missing (first 5): {missing[:5]}[/dim]")
    if unexpected:
        console.print(f"  [dim]Unexpected (first 5): {unexpected[:5]}[/dim]")

# =====================================================================================
# SECTION: MAIN GUI APPLICATION
# =====================================================================================

class TrainingApp(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title("AudioSR DataParallel Trainer - AMD GPU Compatible with torch.compile()")
        self.geometry("1000x900")

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

        # Advanced Loss Monitoring Frame
        loss_frame = ttk.LabelFrame(control_frame, text="Advanced Loss Monitoring", padding=(10, 5))
        loss_frame.grid(row=5, column=0, columnspan=4, sticky='ew', pady=5)

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

        # Row 6: Layer Selection Frame
        layer_frame = ttk.LabelFrame(control_frame, text="Selective Layer Training", padding=(10, 5))
        layer_frame.grid(row=6, column=0, columnspan=4, sticky='ew', pady=10)

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

        # Row 7: Start Button
        self.btn_t = ttk.Button(control_frame, text="Start Training", command=self.start_training_thread, style='Accent.TButton', padding=10)
        self.btn_t.grid(row=7, column=0, columnspan=4, pady=15)

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

    def _setup_selective_training(self, model, config):
        """Setup selective training - prepares parameters for selective optimization."""
        training_mode = self.training_mode.get()

        # Keep all parameters with requires_grad=True for forward pass
        for param in model.parameters():
            param.requires_grad = True

        # Store selected parameters for later optimizer creation
        if training_mode == "full":
            self.selected_params = list(model.parameters())
            self.console.print("[green]Full model training enabled[/green]")

        elif training_mode == "global_encoder":
            # Only train Global Encoder parameters
            self.selected_params = []
            if hasattr(model.model, 'diffusion_model') and hasattr(model.model.diffusion_model, 'global_encoder'):
                self.selected_params = list(model.model.diffusion_model.global_encoder.parameters())
                self.console.print("[green]Global Encoder only training enabled[/green]")
            else:
                self.console.print("[yellow]Global Encoder not found, falling back to full training[/yellow]")
                self.selected_params = list(model.parameters())

        elif training_mode == "unet_only":
            # Train U-Net but not Global Encoder
            self.selected_params = []
            for name, param in model.named_parameters():
                if 'global_encoder' not in name:
                    self.selected_params.append(param)
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
            self.console.print(f"[green]Custom training enabled for: {selected_layers}[/green]")

    def _get_trainable_parameters(self, model):
        """Get list of selected parameters for training."""
        if not hasattr(self, 'selected_params'):
            # Fallback to all parameters if setup wasn't called
            self.selected_params = list(model.parameters())

        trainable_params = self.selected_params

        # Log parameter breakdown for debugging
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

        # Start training in a new thread to keep the GUI responsive
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def on_training_finish(self, final_path):
        """Callback to run on the main thread after training finishes."""
        self.btn_t.config(state=tk.NORMAL, text="Start Training")
        self.console.print(f"[bold green]✅ Training task finished. Final model saved to: {final_path}[/bold green]")
        messagebox.showinfo("Training Complete", f"Training has finished!\nFinal model saved to:\n{final_path}")

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

            self.console.rule("[bold cyan]🚀 Starting DataParallel + torch.compile() training[/bold cyan]")

            # --- 2. Setup ---
            set_seed(current_cfg['experiment']['seed'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            accumulation_steps = current_cfg['train']['gradient_accumulation_steps']
            use_grad_checkpoint = current_cfg['train']['use_gradient_checkpointing']
            use_amp = current_cfg['train']['use_mixed_precision']
            use_compile = current_cfg['train']['use_torch_compile']

            self.console.print(f"Using device: [bold cyan]{device}[/]")
            self.console.print(f"Available GPUs: [bold cyan]{torch.cuda.device_count()}[/]")
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
                self.console.print(f"[green]🔥 GradScaler initialized with scale: {scaler.get_scale():.0e}[/green]")

            # --- 3. DataLoaders ---
            train_set_full = PairedMelDataset(split='train', **current_cfg['data'], preload_to_ram=True)
            valid_set = PairedMelDataset(split='valid', **current_cfg['data'], preload_to_ram=True)

            train_loader = DataLoader(
                train_set_full,
                batch_size=current_cfg['train']['batch_size'],
                shuffle=True,
                num_workers=current_cfg['train'].get('num_workers', 4),
                pin_memory=True
            )
            valid_loader = DataLoader(
                valid_set,
                batch_size=current_cfg['train']['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
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

            # Wrap with DataParallel if multiple GPUs are available
            is_dataparallel = False
            if torch.cuda.device_count() > 1:
                self.console.print(f"[bold green]✅ Using {torch.cuda.device_count()} GPUs for training via DataParallel.[/bold green]")
                model = nn.DataParallel(model)
                is_dataparallel = True
            else:
                self.console.print(f"[yellow]Only 1 GPU available, using single GPU training[/yellow]")

            # Apply torch.compile() after DataParallel wrapping
            if use_compile:
                self.console.print("[cyan]Compiling model with torch.compile()...[/cyan]")
                compile_mode = current_cfg['train']['compile_mode']
                fullgraph = current_cfg['train']['compile_fullgraph']
                model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph)
                self.console.print(f"[green]Model compiled with mode='{compile_mode}', fullgraph={fullgraph}[/green]")

            # Get unwrapped model for EMA and parameter access
            unwrapped_model = model.module if is_dataparallel else model

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

            self.console.print(f"[green]🚀 Starting training for {current_cfg['train']['epochs']} epochs[/green]")

            for ep in range(current_cfg['train']['epochs']):
                for batch_idx, batch in enumerate(train_loader):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                    # Forward pass with mixed precision
                    if use_amp:
                        with autocast():
                            loss, _ = model(batch)
                    else:
                        loss, _ = model(batch)

                    # Handle DataParallel output (already averaged across GPUs)
                    if is_dataparallel:
                        loss = loss.mean()

                    raw_loss = loss.item()
                    loss = loss / accumulation_steps

                    # Advanced loss monitoring
                    actions = loss_monitor.update(raw_loss, gstep, scaler)

                    if actions['skip_step']:
                        self.console.print(f"[red]{actions['message']}[/red]")
                        continue

                    if actions['message']:
                        self.console.print(f"[yellow]{actions['message']}[/yellow]")

                    if actions['reduce_lr']:
                        loss_monitor.reduce_learning_rate(opt, self.console)

                    if actions['stop_training']:
                        self.console.print(f"[red]Early stopping triggered[/red]")
                        break

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

                        # Logging
                        if gstep % current_cfg['train']['log_interval'] == 0:
                            lr = opt.param_groups[0]['lr']
                            scale_info = f", Scale = {scaler.get_scale():.0e}" if scaler else ""
                            self.console.print(f"[green]Step {gstep}: Loss = {raw_loss:.4f}, LR = {lr:.2e}{scale_info}[/green]")
                            self.after(0, self.update_plot, 'train', gstep, raw_loss)

                        # Validation and checkpointing
                        if gstep % current_cfg['train']['valid_interval_steps'] == 0:
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
                                # Handle DataParallel output
                                if is_dataparallel:
                                    val_loss = val_loss.mean()
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

                        # Regular checkpointing
                        if gstep % current_cfg['train']['save_interval_steps'] == 0:
                            path = os.path.join(outp, f'step_{gstep}.pt')
                            torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, path)
                            self.console.print(f"💾 [cyan]Saved checkpoint to {path}[/cyan]")

            # Save final model
            final_p = os.path.join(outp, 'final_ema.pt')
            torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, final_p)

            if scaler:
                self.console.print(f"[cyan]🎯 Final gradient scale: {scaler.get_scale():.0e}[/cyan]")

            self.after(0, self.on_training_finish, final_p)

        except Exception as e:
            error_msg = str(e)
            self.console.print_exception()
            self.after(0, lambda: self.btn_t.config(state=tk.NORMAL, text="Start Training"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("Training Error", f"Training failed: {msg}"))

def run_headless_training(config, console):
    """
    Run training without GUI for command line usage.
    """
    try:
        # Setup
        set_seed(config['experiment']['seed'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        console.print(f"[bold cyan]🚀 Starting headless DataParallel training[/bold cyan]")
        console.print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

        # Load configurations
        use_amp = config['train'].get('use_mixed_precision', True)
        use_compile = config['train'].get('use_torch_compile', True)
        use_grad_checkpoint = config['train'].get('use_gradient_checkpointing', True)

        console.print(f"Mixed Precision: [bold {'green' if use_amp else 'red'}]{use_amp}[/]")
        console.print(f"torch.compile(): [bold {'green' if use_compile else 'red'}]{use_compile}[/]")
        console.print(f"Gradient Checkpointing: [bold {'green' if use_grad_checkpoint else 'red'}]{use_grad_checkpoint}[/]")

        # Create model
        model_config = config['model'].copy()
        model = instantiate_from_config(model_config).to(device)

        # Wrap with DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            console.print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        # Apply torch.compile
        if use_compile:
            console.print("Applying torch.compile()...")
            model = torch.compile(model, mode=config['train'].get('compile_mode', 'default'))

        # Initialize scaler for mixed precision
        scaler = GradScaler(init_scale=config['train'].get('amp_init_scale', 65536.0)) if use_amp and device.startswith('cuda') else None

        console.print("✅ Model setup complete")
        console.print("🎯 Ready for training execution")

        return True

    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AudioSR DataParallel Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
🎛️  USAGE MODES:

  🖥️  GUI Mode (Default):
    python trainMGPU_DataParallel.py
    └─ Launch GUI for configuration and training

  ⚡ Headless Mode:
    python trainMGPU_DataParallel.py --headless
    └─ Start training immediately with config.yaml settings

✨ Features: DataParallel (AMD GPU Compatible), Mixed Precision, torch.compile()
        '''
    )

    parser.add_argument('--headless', action='store_true',
                       help='⚡ Start training immediately with current config.yaml')

    args = parser.parse_args()

    if args.headless:
        # Headless mode
        try:
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            ensure_project_dirs(config)

            console = Console()
            console.print("[bold yellow]⚡ Starting headless training...[/bold yellow]")
            success = run_headless_training(config, console)

            if success:
                console.print("[green]🎉 Training setup completed![/green]")

        except FileNotFoundError:
            print(f"❌ Config file 'config.yaml' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # GUI mode
        try:
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            ensure_project_dirs(config)
            app = TrainingApp(config)
            app.mainloop()
        except FileNotFoundError:
            print(f"❌ Config file 'config.yaml' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)