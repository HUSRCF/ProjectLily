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
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn

# --- Local Project Imports ---
# 确保这些文件与 train_gui.py 在同一目录下
from model import LatentDiffusion 
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
        self.title("AudioSR Integrated Trainer - Performance Edition")
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

        control_frame.columnconfigure(1, weight=1)
        
        # Row 3: Start Button
        self.btn_t = ttk.Button(control_frame, text="Start Training", command=self.start_training_thread, style='Accent.TButton', padding=10)
        self.btn_t.grid(row=3, column=0, columnspan=4, pady=15)

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

    def update_plot(self, msg_type, step, loss):
        """Thread-safe method to update plot data."""
        if msg_type == 'train':
            self.train_loss_data['steps'].append(step)
            self.train_loss_data['losses'].append(loss)
            self.train_line.set_data(self.train_loss_data['steps'], self.train_loss_data['losses'])
        elif msg_type == 'valid':
            self.valid_loss_data['steps'].append(step)
            self.valid_loss_data['losses'].append(loss)
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
            train_set_full = PairedMelDataset(split='train', **current_cfg['data'], preload_to_ram=False)
            valid_set = PairedMelDataset(split='valid', **current_cfg['data'], preload_to_ram=True)
            
            train_buffer = []
            if self.fastboot_var.get():
                initial_chunk_ready = threading.Event()
                loader_thread = threading.Thread(target=self._background_loader, args=(train_set_full, train_buffer, initial_chunk_ready))
                loader_thread.start()
                self.console.print("Fastboot enabled: Waiting for initial 10% of data...")
                initial_chunk_ready.wait()
                self.console.print(f"[green]Initial chunk loaded ({len(train_buffer)} items). Starting training...[/green]")
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
            if use_grad_checkpoint:
                if 'unet_config' in model_config['params'] and 'params' in model_config['params']['unet_config']:
                    model_config['params']['unet_config']['params']['use_checkpoint'] = True
            
            model = instantiate_from_config(model_config).to(device)
            
            epsilon = 1e-8
            model.log_one_minus_alphas_cumprod.clamp_max_(math.log(1.0 - epsilon))
            model.sqrt_recipm1_alphas_cumprod.clamp_max_(math.sqrt(1.0 / epsilon - 1.0))

            if current_cfg['train'].get('pretrained_path'):
                load_pretrained(model, current_cfg['train']['pretrained_path'], device, self.console)

            if self.jit_var.get():
                if not use_grad_checkpoint:
                    self.console.print("[cyan]Compiling model with Torch JIT...[/]")
                    model = torch.jit.script(model)
                    self.console.print("[green]Model compiled.[/green]")
                else:
                    self.console.print("[yellow]Skipping JIT: Incompatible with Gradient Checkpointing.[/yellow]")

            opt = torch.optim.AdamW(model.parameters(), lr=current_cfg['model']['params']['base_learning_rate'], betas=tuple(current_cfg['train']['betas']), weight_decay=current_cfg['train']['weight_decay'])
            
            num_batches = len(train_set_full)
            max_steps = (current_cfg['train']['epochs'] * num_batches) // accumulation_steps
            
            sched = WarmupCosine(opt, current_cfg['model']['params']['base_learning_rate'], warmup=current_cfg['train']['warmup_steps'], max_steps=max_steps)
            ema = EMA(model, decay=current_cfg['train']['ema_decay'])

            # --- 5. Training Loop ---
            best_val_loss = float('inf')
            gstep = 0 
            outp = os.path.join(current_cfg['experiment']['out_dir'], 'checkpoints')
            ensure_dir(outp)

            model.train()
            
            # --- FIX: Initialize display variables before the loop to make them persistent ---
            loss_display = "Loss: N/A"
            lr_display = "LR: N/A"
            
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
                TextColumn("• {task.fields[loss]}"),
                TextColumn("• {task.fields[lr]}"),
                console=self.console,
            ) as progress:
                
                main_task = progress.add_task("Overall Progress", total=current_cfg['train']['epochs'], loss="", lr="")

                for ep in range(current_cfg['train']['epochs']):
                    if gstep >= max_steps: break
                    
                    if self.fastboot_var.get() and ep == 1:
                        self.console.print("Waiting for full dataset to load...")
                        loader_thread.join()
                        self.console.print("[green]Full dataset loaded. Re-initializing DataLoader for subsequent epochs.[/green]")
                        train_loader = DataLoader(train_buffer, batch_size=current_cfg['train']['batch_size'], shuffle=True)

                    epoch_task = progress.add_task(f"[cyan]Epoch {ep+1}", total=len(train_loader), loss=loss_display, lr=lr_display)
                    opt.zero_grad()
                    
                    for batch_idx, batch in enumerate(train_loader):
                        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                        loss, _ = model(batch)
                        
                        # --- FIX: Display current batch loss immediately for instant feedback ---
                        loss_display = f"Batch Loss: {loss.item():.4f}"
                        
                        loss = loss / accumulation_steps
                        
                        loss.backward()

                        if (batch_idx + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            
                            opt.step()
                            
                            sched.step()
                            ema.update(model)
                            opt.zero_grad()
                            gstep += 1
                            
                            effective_loss = loss.item() * accumulation_steps
                            current_lr = opt.param_groups[0]['lr']
                            
                            # --- FIX: Update display with effective loss and LR after optimizer step ---
                            loss_display = f"[bold green]Loss: {effective_loss:.4f}[/bold green]"
                            lr_display = f"[dim]LR: {current_lr:.2e}[/dim]"
                            
                            if gstep % current_cfg['train']['log_interval'] == 0:
                                self.after(0, self.update_plot, 'train', gstep, effective_loss)

                            if gstep % current_cfg['train']['valid_interval_steps'] == 0:
                                model.eval()
                                val_losses = []
                                for val_batch in valid_loader:
                                    val_batch = {k: v.to(device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                                    with torch.no_grad():
                                        val_loss, _ = model(val_batch)
                                    val_losses.append(val_loss.item())
                                
                                avg_val_loss = sum(val_losses) / len(val_losses)
                                progress.console.print(f"📊 Validation @ step {gstep}: loss = {avg_val_loss:.4f}")
                                self.after(0, self.update_plot, 'valid', gstep, avg_val_loss)
                                model.train()

                                if avg_val_loss < best_val_loss:
                                    best_val_loss = avg_val_loss
                                    path = os.path.join(outp, f'best_step_{gstep}.pt')
                                    torch.save({'state_dict': model.state_dict(), 'ema': ema.shadow}, path)
                                    progress.console.print(f"💾 [bold magenta]Saved best model to {path}[/bold magenta]")

                            if gstep % current_cfg['train']['save_interval_steps'] == 0:
                                path = os.path.join(outp, f'step_{gstep}.pt')
                                torch.save({'state_dict': model.state_dict(), 'ema': ema.shadow}, path)
                                progress.console.print(f"💾 [cyan]Saved checkpoint to {path}[/cyan]")
                        
                        progress.update(epoch_task, advance=1, fields={"loss": loss_display, "lr": lr_display})
                    
                    progress.update(epoch_task, visible=False)
                    progress.update(main_task, advance=1)

            final_p = os.path.join(outp, 'final_ema.pt')
            torch.save({'state_dict': model.state_dict(), 'ema': ema.shadow}, final_p)
            self.after(0, self.on_training_finish, final_p)

        except Exception as e:
            self.console.print_exception()
            self.after(0, self.on_training_finish, f"Error: {e}")

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
