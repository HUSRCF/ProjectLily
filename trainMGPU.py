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
import torch.nn as nn # [MODIFIED] Import nn for DataParallel
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn

# --- Local Project Imports ---
# 确保这些文件与 train_gui.py 在同一目录下
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
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        # Create a shadow copy of parameters in full FP32 precision
        self.shadow = {
            k: p.data.clone().to(dtype=torch.float32)
            for k, p in model.named_parameters() if p.requires_grad
        }
    
    def update(self, model):
        with torch.no_grad():
            for k, p in model.named_parameters():
                if p.requires_grad and k in self.shadow:
                    p_fp32 = p.data.to(dtype=torch.float32)
                    self.shadow[k] = (1.0 - self.decay) * p_fp32 + self.decay * self.shadow[k]

class AdvancedLossMonitor:
    """
    Advanced loss monitoring with explosion detection, adaptive LR, and early stopping.
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
        
    def update(self, loss_value, step):
        """Update loss monitoring and return actions to take."""
        actions = {
            'reduce_lr': False,
            'stop_training': False,
            'skip_step': False,
            'message': ''
        }
        
        # Check for NaN or infinite loss
        if not torch.isfinite(torch.tensor(loss_value)):
            actions['skip_step'] = True
            actions['message'] = f"⚠️ NaN/Inf loss detected at step {step}, skipping step"
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
            'long_ma': self.long_ma
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

    missing, unexpected = model.load_state_dict(sd, strict=False)
    
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
        self.title("AudioSR Integrated Trainer - Performance Edition (DataParallel)")
        self.geometry("1000x850")

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

        self.jit_var = tk.BooleanVar(value=self.cfg['train'].get('use_jit_compile', True))
        ttk.Checkbutton(perf_frame, text="Use JIT Compilation", variable=self.jit_var).pack(side='left', padx=10)

        self.grad_checkpoint_var = tk.BooleanVar(value=self.cfg['train'].get('use_gradient_checkpointing', True))
        ttk.Checkbutton(perf_frame, text="Use Gradient Checkpointing", variable=self.grad_checkpoint_var).pack(side='left', padx=10)
        
        # Advanced Loss Monitoring Frame
        loss_frame = ttk.LabelFrame(control_frame, text="Advanced Loss Monitoring", padding=(10, 5))
        loss_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=5)
        
        # Loss explosion detection
        self.loss_explosion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(loss_frame, text="Loss Explosion Protection", variable=self.loss_explosion_var).grid(row=0, column=0, sticky='w', padx=5)
        
        ttk.Label(loss_frame, text="Explosion Threshold:").grid(row=0, column=1, sticky='w', padx=5)
        self.explosion_threshold_var = tk.StringVar(value="10.0")
        ttk.Entry(loss_frame, textvariable=self.explosion_threshold_var, width=8).grid(row=0, column=2, padx=5)
        
        # Adaptive learning rate
        self.adaptive_lr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(loss_frame, text="Adaptive Learning Rate", variable=self.adaptive_lr_var).grid(row=1, column=0, sticky='w', padx=5)
        
        ttk.Label(loss_frame, text="Reduction Factor:").grid(row=1, column=1, sticky='w', padx=5)
        self.lr_reduction_var = tk.StringVar(value="0.5")
        ttk.Entry(loss_frame, textvariable=self.lr_reduction_var, width=8).grid(row=1, column=2, padx=5)
        
        # Early stopping
        self.early_stopping_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(loss_frame, text="Early Stopping", variable=self.early_stopping_var).grid(row=2, column=0, sticky='w', padx=5)
        
        ttk.Label(loss_frame, text="Patience (steps):").grid(row=2, column=1, sticky='w', padx=5)
        self.patience_var = tk.StringVar(value="1000")
        ttk.Entry(loss_frame, textvariable=self.patience_var, width=8).grid(row=2, column=2, padx=5)

        control_frame.columnconfigure(1, weight=1)
        
        # Row 4: Layer Selection Frame (moved down)
        layer_frame = ttk.LabelFrame(control_frame, text="Selective Layer Training", padding=(10, 5))
        layer_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=10)
        
        # Training mode selection
        self.training_mode = tk.StringVar(value="full")
        ttk.Radiobutton(layer_frame, text="Full Model Training", variable=self.training_mode, value="full").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="Global Encoder Only", variable=self.training_mode, value="global_encoder").grid(row=0, column=1, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="U-Net Only (Freeze Global Encoder)", variable=self.training_mode, value="unet_only").grid(row=0, column=2, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="Custom Layer Selection", variable=self.training_mode, value="custom").grid(row=1, column=0, sticky='w', padx=5)
        
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
        
        # Row 5: Start Button (moved down)
        self.btn_t = ttk.Button(control_frame, text="Start Training", command=self.start_training_thread, style='Accent.TButton', padding=10)
        self.btn_t.grid(row=5, column=0, columnspan=4, pady=15)

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
    
    # [MODIFIED] This function now takes the model to inspect as an argument
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
            
            current_cfg['train']['pretrained_path'] = self.pretrained_path_var.get() or None
            current_cfg['train']['gradient_accumulation_steps'] = int(self.grad_accum_var.get())
            current_cfg['train']['fastboot'] = self.fastboot_var.get()
            current_cfg['train']['use_jit_compile'] = self.jit_var.get()
            current_cfg['train']['use_gradient_checkpointing'] = self.grad_checkpoint_var.get()
            
            # Advanced loss monitoring settings
            current_cfg['train']['loss_explosion_protection'] = self.loss_explosion_var.get()
            current_cfg['train']['explosion_threshold'] = float(self.explosion_threshold_var.get())
            current_cfg['train']['adaptive_lr'] = self.adaptive_lr_var.get()
            current_cfg['train']['lr_reduction_factor'] = float(self.lr_reduction_var.get())
            current_cfg['train']['early_stopping'] = self.early_stopping_var.get()
            current_cfg['train']['patience'] = int(self.patience_var.get())
            
            # Remove fp16 from config if it exists
            current_cfg['train'].pop('use_fp16', None)

            with open(CFG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(current_cfg, f, sort_keys=False, allow_unicode=True)
            
            self.console.rule("[bold cyan]🚀 Starting new training task[/bold cyan]")
            
            # --- 2. Setup ---
            set_seed(current_cfg['experiment']['seed'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            accumulation_steps = current_cfg['train']['gradient_accumulation_steps']
            use_grad_checkpoint = current_cfg['train']['use_gradient_checkpointing']
            
            self.console.print(f"Using device: [bold cyan]{device}[/]")
            self.console.print(f"Gradient Accumulation: [bold cyan]{accumulation_steps} steps[/]")
            self.console.print(f"Fastboot: [bold {'green' if self.fastboot_var.get() else 'red'}]{self.fastboot_var.get()}[/]")
            self.console.print(f"JIT Compilation: [bold {'green' if self.jit_var.get() else 'red'}]{self.jit_var.get()}[/]")
            self.console.print(f"Gradient Checkpointing: [bold {'green' if use_grad_checkpoint else 'red'}]{use_grad_checkpoint}[/]")

            # --- 3. DataLoaders ---
            train_set_full = PairedMelDataset(split='train', **current_cfg['data'], preload_to_ram=True)
            valid_set = PairedMelDataset(split='valid', **current_cfg['data'], preload_to_ram=True)
            
            train_buffer = []
            loader_thread = None
            
            if self.fastboot_var.get():
                initial_chunk_ready = threading.Event()
                loader_thread = threading.Thread(target=self._background_loader, args=(train_set_full, train_buffer, initial_chunk_ready), daemon=True)
                loader_thread.start()
                self.console.print("Fastboot enabled: Waiting for initial 10% of data...")
                
                # Wait with timeout to prevent freezing
                if initial_chunk_ready.wait(timeout=60.0):
                    self.console.print(f"[green]Initial chunk loaded ({len(train_buffer)} items). Starting training...[/green]")
                else:
                    self.console.print("[yellow]Initial loading timeout, loading synchronously instead...[/yellow]")
                    # Fallback to synchronous loading
                    for i in range(min(len(train_set_full), 100)):  # Load first 100 items as fallback
                        item = train_set_full._load_item(i)
                        if item: train_buffer.append(item)
                    loader_thread = None  # Don't try to join later
                    
                train_loader = DataLoader(train_buffer, batch_size=current_cfg['train']['batch_size'], shuffle=True)
            else:
                self.console.print("Loading full training data synchronously...")
                for i in range(len(train_set_full)):
                    item = train_set_full._load_item(i)
                    if item: train_buffer.append(item)
                self.console.print("[green]Full dataset loaded.[/green]")
                train_loader = DataLoader(train_buffer, batch_size=current_cfg['train']['batch_size'], shuffle=True)

            valid_loader = DataLoader(valid_set, batch_size=current_cfg['train']['batch_size'], shuffle=False)

            # --- 4. Model, Optimizer, Scheduler, EMA ---
            model_config = current_cfg['model'].copy()
            
            # Disable gradient checkpointing for selective training to avoid grad issues
            if self.training_mode.get() != "full" and use_grad_checkpoint:
                if 'unet_config' in model_config['params'] and 'params' in model_config['params']['unet_config']:
                    model_config['params']['unet_config']['params']['use_checkpoint'] = False
                self.console.print("[yellow]Disabled gradient checkpointing for selective training[/yellow]")
            elif use_grad_checkpoint:
                if 'unet_config' in model_config['params'] and 'params' in model_config['params']['unet_config']:
                    model_config['params']['unet_config']['params']['use_checkpoint'] = True
            
            model = instantiate_from_config(model_config).to(device)
            
            epsilon = 1e-8
            model.log_one_minus_alphas_cumprod.clamp_max_(math.log(1.0 - epsilon))
            model.sqrt_recipm1_alphas_cumprod.clamp_max_(math.sqrt(1.0 / epsilon - 1.0))

            if current_cfg['train'].get('pretrained_path'):
                load_pretrained(model, current_cfg['train']['pretrained_path'], device, self.console)

            # [MODIFIED] Wrap model with DataParallel if multiple GPUs are available
            is_dataparallel = False
            if torch.cuda.device_count() > 1:
                self.console.print(f"[bold green]✅ Using {torch.cuda.device_count()} GPUs for training via DataParallel.[/bold green]")
                model = nn.DataParallel(model)
                is_dataparallel = True

            # [MODIFIED] Get the original model from the wrapper if it exists
            unwrapped_model = model.module if is_dataparallel else model
            
            # [MODIFIED] Pass the unwrapped model to setup functions
            self._setup_selective_training(unwrapped_model, current_cfg)
            trainable_params = self._get_trainable_parameters(unwrapped_model)
            
            if not trainable_params:
                raise RuntimeError("No trainable parameters found! Check your layer selection.")
            
            # [MODIFIED] Check JIT compatibility with DataParallel
            use_jit = self.jit_var.get()
            if use_jit and is_dataparallel:
                self.console.print("[yellow]⚠️ Skipping JIT compilation: Incompatible with nn.DataParallel.[/yellow]")
                use_jit = False

            if use_jit:
                if not use_grad_checkpoint:
                    self.console.print("[cyan]Compiling model with Torch JIT...[/]")
                    # Note: JIT scripting is applied to the potentially wrapped model
                    model = torch.jit.script(model)
                    self.console.print("[green]Model compiled.[/green]")
                else:
                    self.console.print("[yellow]Skipping JIT: Incompatible with Gradient Checkpointing.[/yellow]")

            self.console.print(f"[cyan]Training mode: {self.training_mode.get()}[/]")
            
            opt = torch.optim.AdamW(trainable_params, lr=current_cfg['model']['params']['base_learning_rate'], betas=tuple(current_cfg['train']['betas']), weight_decay=current_cfg['train']['weight_decay'])
            
            num_batches = len(train_set_full)
            max_steps = (current_cfg['train']['epochs'] * num_batches) // accumulation_steps
            
            sched = WarmupCosine(opt, current_cfg['model']['params']['base_learning_rate'], warmup=current_cfg['train']['warmup_steps'], max_steps=max_steps)
            
            # [MODIFIED] Initialize EMA with the unwrapped model
            ema = EMA(unwrapped_model, decay=current_cfg['train']['ema_decay'])
            
            # Initialize advanced loss monitor
            loss_monitor = None
            if current_cfg['train'].get('loss_explosion_protection', False) or current_cfg['train'].get('adaptive_lr', False):
                loss_monitor = AdvancedLossMonitor(
                    explosion_threshold=current_cfg['train'].get('explosion_threshold', 10.0),
                    lr_reduction_factor=current_cfg['train'].get('lr_reduction_factor', 0.5),
                    patience=current_cfg['train'].get('patience', 1000) if current_cfg['train'].get('early_stopping', False) else float('inf')
                )
                self.console.print(f"[cyan]Advanced Loss Monitor enabled:[/cyan]")
                self.console.print(f"[dim]  Explosion threshold: {loss_monitor.explosion_threshold}[/dim]")
                self.console.print(f"[dim]  LR reduction factor: {loss_monitor.lr_reduction_factor}[/dim]")
                if current_cfg['train'].get('early_stopping', False):
                    self.console.print(f"[dim]  Early stopping patience: {loss_monitor.patience} steps[/dim]")

            # --- 5. Training Loop ---
            best_val_loss = float('inf')
            gstep = 0 
            outp = os.path.join(current_cfg['experiment']['out_dir'], 'checkpoints')
            ensure_dir(outp)

            # Early stopping flag
            early_stop_triggered = False

            model.train()
            
            current_loss_value = 0.0
            current_lr_value = current_cfg['model']['params']['base_learning_rate']
            
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
                TextColumn("• {task.fields[loss]}"),
                TextColumn("• {task.fields[lr]}"),
                console=self.console,
            ) as progress:
                
                main_task = progress.add_task("Overall Progress", total=current_cfg['train']['epochs'], loss="", lr="")
                # #[RESTORED] Restoring the original, more robust progress bar handling logic
                current_epoch_task = None
                epoch_tasks_to_cleanup = []

                for ep in range(current_cfg['train']['epochs']):
                    if gstep >= max_steps or early_stop_triggered: 
                        break
                    
                    if loss_monitor is not None:
                        stats = loss_monitor.get_stats()
                        if stats['steps_without_improvement'] >= loss_monitor.patience and current_cfg['train'].get('early_stopping', False):
                            self.console.print(f"[red]Early stopping: training terminated at epoch {ep+1}[/red]")
                            early_stop_triggered = True
                            break
                    
                    if self.fastboot_var.get() and ep == 1 and loader_thread is not None:
                        self.console.print("[cyan]Waiting for full dataset to load...[/cyan]")
                        try:
                            if loader_thread.is_alive():
                                loader_thread.join(timeout=60.0)
                                if loader_thread.is_alive():
                                    self.console.print("[yellow]Dataset loading timeout, continuing with current data[/yellow]")
                                    loader_thread = None
                                else:
                                    self.console.print(f"[green]Full dataset loaded ({len(train_buffer)} items). Re-initializing DataLoader.[/green]")
                                    train_loader = DataLoader(train_buffer, batch_size=current_cfg['train']['batch_size'], shuffle=True)
                            else:
                                self.console.print("[green]Background loading already completed.[/green]")
                                train_loader = DataLoader(train_buffer, batch_size=current_cfg['train']['batch_size'], shuffle=True)
                        except Exception as e:
                            self.console.print(f"[yellow]Dataset loading error: {e}, continuing with current data[/yellow]")
                            loader_thread = None

                    # #[RESTORED]
                    if current_epoch_task is not None:
                        epoch_tasks_to_cleanup.append(current_epoch_task)
                        if len(epoch_tasks_to_cleanup) > 2:
                            old_task = epoch_tasks_to_cleanup.pop(0)
                            try:
                                if (hasattr(progress, 'tasks') and old_task in progress.tasks):
                                    progress.remove_task(old_task)
                            except (KeyError, IndexError, ValueError):
                                pass
                    
                    # #[RESTORED]
                    loss_display = f"Loss: {current_loss_value:.4f}" if current_loss_value > 0 else "Loss: Starting..."
                    lr_display = f"LR: {current_lr_value:.2e}"
                    
                    stable_batch_count = len(train_loader)
                    current_epoch_task = progress.add_task(f"[cyan]Epoch {ep+1}", total=stable_batch_count, loss=loss_display, lr=lr_display)
                    opt.zero_grad()
                    
                    self.console.print(f"[green]📈 Starting epoch {ep+1}/{current_cfg['train']['epochs']} with {stable_batch_count} batches[/green]")
                    
                    for batch_idx, batch in enumerate(train_loader):
                        if batch_idx % 50 == 0:
                            self.console.print(f"[dim]Epoch {ep+1}, Batch {batch_idx+1}/{len(train_loader)}[/dim]")
                        
                        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                        loss, _ = model(batch)
                        
                        # [MODIFIED] Handle loss gathering from multiple GPUs by taking the mean
                        if is_dataparallel:
                            loss = loss.mean()
                        
                        raw_loss = loss.item()
                        
                        skip_this_step = False
                        if loss_monitor is not None:
                            actions = loss_monitor.update(raw_loss, gstep)
                            
                            if actions['message']:
                                self.console.print(f"[yellow]{actions['message']}[/yellow]")
                            
                            if actions['skip_step']:
                                skip_this_step = True
                                self.console.print(f"[red]Skipping step {gstep} due to NaN/Inf loss[/red]")
                            
                            if actions['reduce_lr'] and current_cfg['train'].get('adaptive_lr', False):
                                new_lr = loss_monitor.reduce_learning_rate(opt, self.console)
                                current_lr_value = new_lr
                            
                            if actions['stop_training'] and current_cfg['train'].get('early_stopping', False):
                                self.console.print(f"[red]Early stopping triggered at step {gstep}[/red]")
                                early_stop_triggered = True
                                break
                        
                        if skip_this_step:
                            continue
                        
                        loss = loss / accumulation_steps
                        
                        if loss.requires_grad:
                            loss.backward()
                        else:
                            raise RuntimeError(f"Loss tensor does not require grad. Check model parameters.")

                        if (batch_idx + 1) % accumulation_steps == 0:
                            # [MODIFIED] Clip gradients of the unwrapped model's parameters
                            trainable_params_for_clipping = [p for p in unwrapped_model.parameters() if p.requires_grad and p.grad is not None]
                            if trainable_params_for_clipping:
                                torch.nn.utils.clip_grad_norm_(trainable_params_for_clipping, 1.0)
                            
                            opt.step()
                            
                            sched.step()
                            # [MODIFIED] Update EMA with the unwrapped model
                            ema.update(unwrapped_model)
                            opt.zero_grad()
                            gstep += 1
                            
                            current_loss_value = raw_loss
                            current_lr_value = opt.param_groups[0]['lr']
                            
                            loss_display = f"Loss: {current_loss_value:.4f}"
                            lr_display = f"LR: {current_lr_value:.2e}"
                            
                            if gstep % current_cfg['train']['log_interval'] == 0:
                                self.console.print(f"[green]Step {gstep}: Loss = {current_loss_value:.4f}, LR = {current_lr_value:.2e}[/green]")
                                self.after(0, self.update_plot, 'train', gstep, current_loss_value)
                                
                                if loss_monitor is not None and gstep % (current_cfg['train']['log_interval'] * 4) == 0:
                                    stats = loss_monitor.get_stats()
                                    self.console.print(f"[dim]📊 Monitor Stats - Best: {stats['best_loss']:.4f}, "
                                                     f"No improvement: {stats['steps_without_improvement']}, "
                                                     f"LR reductions: {stats['lr_reductions']}, "
                                                     f"Short MA: {stats['short_ma']:.4f}, Long MA: {stats['long_ma']:.4f}[/dim]")

                            if gstep % current_cfg['train']['valid_interval_steps'] == 0:
                                model.eval()
                                val_losses = []
                                for val_batch in valid_loader:
                                    val_batch = {k: v.to(device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                                    with torch.no_grad():
                                        val_loss, _ = model(val_batch)
                                    # [MODIFIED] Average validation loss across GPUs
                                    if is_dataparallel:
                                        val_loss = val_loss.mean()
                                    val_losses.append(val_loss.item())
                                
                                avg_val_loss = sum(val_losses) / len(val_losses)
                                progress.console.print(f"📊 Validation @ step {gstep}: loss = {avg_val_loss:.4f}")
                                self.after(0, self.update_plot, 'valid', gstep, avg_val_loss)
                                model.train()

                                # [MODIFIED] Save the unwrapped model's state_dict
                                if avg_val_loss < best_val_loss:
                                    best_val_loss = avg_val_loss
                                    path = os.path.join(outp, f'best_step_{gstep}.pt')
                                    torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, path)
                                    progress.console.print(f"💾 [bold magenta]Saved best model to {path}[/bold magenta]")

                            # [MODIFIED] Save the unwrapped model's state_dict
                            if gstep % current_cfg['train']['save_interval_steps'] == 0:
                                path = os.path.join(outp, f'step_{gstep}.pt')
                                torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, path)
                                progress.console.print(f"💾 [cyan]Saved checkpoint to {path}[/cyan]")
                        
                        # #[RESTORED]
                        current_batch_loss = f"Loss: {raw_loss:.4f}" if 'raw_loss' in locals() else loss_display
                        current_lr_str = f"LR: {opt.param_groups[0]['lr']:.2e}" if len(opt.param_groups) > 0 else lr_display
                        
                        try:
                            if (current_epoch_task is not None and 
                                hasattr(progress, 'tasks') and 
                                current_epoch_task in progress.tasks):
                                
                                task = progress.tasks[current_epoch_task]
                                current_progress = task.completed
                                total_progress = task.total or len(train_loader)
                                
                                if current_progress < total_progress:
                                    progress.update(current_epoch_task, advance=1, fields={"loss": current_batch_loss, "lr": current_lr_str})
                                else:
                                    progress.update(current_epoch_task, fields={"loss": current_batch_loss, "lr": current_lr_str})
                        except (KeyError, IndexError, AttributeError):
                            pass
                        except Exception as e:
                            if batch_idx % 100 == 0:
                                self.console.print(f"[dim]Progress update error (batch {batch_idx}): {e}[/dim]")
                    
                    if early_stop_triggered:
                        break
                    
                    progress.update(main_task, advance=1)

                # #[RESTORED]
                for task_id in epoch_tasks_to_cleanup:
                    try:
                        if (hasattr(progress, 'tasks') and task_id in progress.tasks):
                            progress.remove_task(task_id)
                    except (KeyError, IndexError, ValueError):
                        pass
                        
                if (current_epoch_task is not None and 
                    hasattr(progress, 'tasks') and 
                    current_epoch_task in progress.tasks):
                    try:
                        progress.remove_task(current_epoch_task)
                    except (KeyError, IndexError, ValueError):
                        pass

            # [MODIFIED] Save final unwrapped model
            final_p = os.path.join(outp, 'final_ema.pt')
            torch.save({'state_dict': unwrapped_model.state_dict(), 'ema': ema.shadow}, final_p)
            self.after(0, self.on_training_finish, final_p)

        except Exception as e:
            error_msg = str(e)
            self.console.print_exception()
            self.after(0, lambda: self.btn_t.config(state=tk.NORMAL, text="Start Training"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("Training Error", f"Training failed: {msg}"))

if __name__ == '__main__':
    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ensure_project_dirs(config)
        app = TrainingApp(config)
        app.mainloop()
    except FileNotFoundError:
        print(f"错误: 配置文件 'config.yaml' 未找到。请确保它在脚本所在的目录中。")
        sys.exit(1)
    except Exception as e:
        print(f"启动 GUI 时发生严重错误: {e}")
        sys.exit(1)

