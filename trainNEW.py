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
from modelV1 import LatentDiffusion 
from dataset_loader import PairedMelDataset
from audiosr.latent_diffusion.util import instantiate_from_config

# --- Base Configuration ---
BASE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE, 'config.yaml')

# =====================================================================================
# SECTION: UTILITY & CORE TRAINING LOGIC
# =====================================================================================

def ensure_project_dirs(cfg: dict):
    """确保所有必要的目录都存在"""
    try:
        outp = os.path.join(BASE, cfg['experiment']['out_dir'], 'checkpoints')
        os.makedirs(outp, exist_ok=True)
        dataset_root = os.path.join(BASE, cfg['data']['dataset_root'])
        os.makedirs(os.path.join(dataset_root, 'train', cfg['data']['high_dir_name']), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'train', cfg['data']['low_dir_name']), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'valid', cfg['data']['high_dir_name']), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'valid', cfg['data']['low_dir_name']), exist_ok=True)
    except KeyError as e:
        raise KeyError(f"配置文件缺失键: {e}")

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class EMA:
    """指数移动平均"""
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: p.data.clone().to(dtype=torch.float32) for k, p in model.named_parameters() if p.requires_grad and 'discriminator' not in k}
    def update(self, model):
        with torch.no_grad():
            for k, p in model.named_parameters():
                if p.requires_grad and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(p.data.to(dtype=torch.float32), alpha=1.0 - self.decay)

class AdvancedLossMonitor:
    """高级损失监控器，用于检测爆炸、自适应学习率和早停"""
    def __init__(self, optimizer, console, **kwargs):
        self.optimizer = optimizer
        self.console = console
        self.explosion_threshold = kwargs.get('explosion_threshold', 10.0)
        self.lr_reduction_factor = kwargs.get('lr_reduction_factor', 0.5)
        self.patience = kwargs.get('patience', 1000)
        self.min_lr = 1e-7
        self.best_loss = float('inf')
        self.steps_without_improvement = 0

    def update(self, loss_value, step):
        if not torch.isfinite(torch.tensor(loss_value)):
            self.console.print(f"[bold red]警告: 步骤 {step} 损失为 NaN/Inf，跳过此步[/bold red]")
            return True # skip step

        if loss_value > self.explosion_threshold:
            self.console.print(f"[bold yellow]警告: 步骤 {step} 损失爆炸 ({loss_value:.4f})，降低学习率[/bold yellow]")
            self.reduce_learning_rate()
        
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        if self.steps_without_improvement >= self.patience:
            self.console.print(f"[bold red]早停: {self.patience} 步内无改善，停止训练[/bold red]")
            return "stop_training"
        return False

    def reduce_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.console.print(f"[yellow]学习率已降低: {old_lr:.2e} -> {new_lr:.2e}[/yellow]")

class WarmupCosine:
    """带预热的余弦学习率调度器"""
    def __init__(self, opt, base_lr, warmup, max_steps, min_ratio=0.1):
        self.opt, self.base, self.w, self.m, self.r = opt, base_lr, warmup, max_steps, min_ratio
        self.step_id = 0
    def step(self):
        if self.step_id < self.w: f = (self.step_id + 1) / max(1, self.w)
        else: t = (self.step_id - self.w) / max(1, self.m - self.w); f = self.r + 0.5 * (1 - self.r) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups: g['lr'] = self.base * f
        self.step_id += 1

def load_pretrained(model: torch.nn.Module, path: str, device: str, console: Console):
    if not path or not os.path.exists(path):
        console.print("[yellow]未找到预训练权重，从头开始训练。[/yellow]"); return
    console.print(f"[cyan]加载预训练权重: {path}[/cyan]")
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt.get("ema", ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    console.print(f"[green]权重加载完毕。[/green] 缺失键: {len(missing)}, 意外键: {len(unexpected)}")

    if len(missing) > 0:
        console.print(f"[yellow]🔍 缺失键示例 (前10个):[/yellow]")
        for key in missing[:10]:
            console.print(f"  - {key}")

    if len(unexpected) > 0:
        console.print(f"[yellow]🔍 意外键示例 (前10个):[/yellow]")
        for key in unexpected[:10]:
            console.print(f"  - {key}")

    # 检查关键组件的加载状态
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(sd.keys())

    # 分析主要组件的匹配情况
    components = {
        "diffusion_model": "扩散模型核心",
        "first_stage_model": "第一阶段模型(VAE)",
        "cond_stage_models": "条件模型",
        "discriminator": "判别器"
    }

    console.print("[cyan]🔍 组件加载分析:[/cyan]")
    for comp, desc in components.items():
        model_comp_keys = {k for k in model_keys if comp in k}
        checkpoint_comp_keys = {k for k in checkpoint_keys if comp in k}

        if model_comp_keys and checkpoint_comp_keys:
            match_ratio = len(model_comp_keys & checkpoint_comp_keys) / len(model_comp_keys)
            status = "✅" if match_ratio > 0.8 else "⚠️" if match_ratio > 0.5 else "❌"
            console.print(f"  {status} {desc}: {match_ratio*100:.1f}% 匹配")
        elif model_comp_keys:
            console.print(f"  ❌ {desc}: 检查点中完全缺失")
        elif checkpoint_comp_keys:
            console.print(f"  ⚠️ {desc}: 仅在检查点中存在")

# =====================================================================================
# SECTION: MAIN GUI APPLICATION
# =====================================================================================

class TrainingApp(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title("AudioSR GAN Trainer - Ultimate Edition")
        self.geometry("1000x850")
        self.console = Console()
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self); style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold')); style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), foreground='white', background='#0078D7'); style.configure('TCheckbutton', font=('Helvetica', 10))
        control_frame = ttk.LabelFrame(self, text="训练控制与优化", padding=(15, 10)); control_frame.pack(fill='x', padx=15, pady=10)

        # Pretrained model path
        self.pretrained_path_var = tk.StringVar(value=self.cfg['train'].get('pretrained_path', '')); ttk.Label(control_frame, text="预训练权重:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.pretrained_path_var, width=70).grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text="浏览...", command=self.select_file).grid(row=0, column=3, padx=5, pady=5)
        
        # Gradient accumulation steps
        self.grad_accum_var = tk.StringVar(value=str(self.cfg['train'].get('gradient_accumulation_steps', 1))); ttk.Label(control_frame, text="梯度累积步数:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Spinbox(control_frame, from_=1, to=128, textvariable=self.grad_accum_var, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Performance Toggles
        perf_frame = ttk.Frame(control_frame); perf_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=10)
        self.fastboot_var = tk.BooleanVar(value=self.cfg['train'].get('fastboot', True)); ttk.Checkbutton(perf_frame, text="快速启动 (异步加载)", variable=self.fastboot_var).pack(side='left', padx=10)
        self.jit_var = tk.BooleanVar(value=self.cfg['train'].get('use_jit_compile', False)); ttk.Checkbutton(perf_frame, text="使用 JIT 编译", variable=self.jit_var).pack(side='left', padx=10)
        self.grad_checkpoint_var = tk.BooleanVar(value=self.cfg['train'].get('use_gradient_checkpointing', True)); ttk.Checkbutton(perf_frame, text="使用梯度检查点", variable=self.grad_checkpoint_var).pack(side='left', padx=10)
        
        # Advanced Loss Monitoring Frame
        loss_frame = ttk.LabelFrame(control_frame, text="高级损失监控", padding=(10, 5)); loss_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=5)
        self.loss_explosion_var = tk.BooleanVar(value=self.cfg['train'].get('loss_explosion_protection', True)); ttk.Checkbutton(loss_frame, text="损失爆炸保护", variable=self.loss_explosion_var).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(loss_frame, text="爆炸阈值:").grid(row=0, column=1, sticky='w', padx=5); self.explosion_threshold_var = tk.StringVar(value=str(self.cfg['train'].get('explosion_threshold', 10.0))); ttk.Entry(loss_frame, textvariable=self.explosion_threshold_var, width=8).grid(row=0, column=2, padx=5)
        self.adaptive_lr_var = tk.BooleanVar(value=self.cfg['train'].get('adaptive_lr', True)); ttk.Checkbutton(loss_frame, text="自适应学习率", variable=self.adaptive_lr_var).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(loss_frame, text="降低因子:").grid(row=1, column=1, sticky='w', padx=5); self.lr_reduction_var = tk.StringVar(value=str(self.cfg['train'].get('lr_reduction_factor', 0.5))); ttk.Entry(loss_frame, textvariable=self.lr_reduction_var, width=8).grid(row=1, column=2, padx=5)
        self.early_stopping_var = tk.BooleanVar(value=self.cfg['train'].get('early_stopping', False)); ttk.Checkbutton(loss_frame, text="早停", variable=self.early_stopping_var).grid(row=2, column=0, sticky='w', padx=5)
        ttk.Label(loss_frame, text="耐心 (步数):").grid(row=2, column=1, sticky='w', padx=5); self.patience_var = tk.StringVar(value=str(self.cfg['train'].get('patience', 20000))); ttk.Entry(loss_frame, textvariable=self.patience_var, width=8).grid(row=2, column=2, padx=5)

        control_frame.columnconfigure(1, weight=1)
        
        # Layer Selection Frame
        layer_frame = ttk.LabelFrame(control_frame, text="选择性层训练", padding=(10, 5)); layer_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=10)
        self.training_mode = tk.StringVar(value="full"); self.training_mode.trace('w', self._update_layer_selection_state)
        ttk.Radiobutton(layer_frame, text="完整模型", variable=self.training_mode, value="full").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="仅全局编码器", variable=self.training_mode, value="global_encoder").grid(row=0, column=1, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="仅U-Net", variable=self.training_mode, value="unet_only").grid(row=0, column=2, sticky='w', padx=5)
        ttk.Radiobutton(layer_frame, text="自定义", variable=self.training_mode, value="custom").grid(row=1, column=0, sticky='w', padx=5)
        self.custom_layers_frame = ttk.Frame(layer_frame); self.custom_layers_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=5)
        self.layer_vars = {}
        for i, (key, name) in enumerate([("model.diffusion_model.global_encoder", "全局编码器"), ("model.diffusion_model.input_blocks", "U-Net输入"), ("model.diffusion_model.middle_block", "U-Net中部"), ("model.diffusion_model.output_blocks", "U-Net输出"), ("first_stage_model", "VAE"), ("cond_stage_models", "条件模型")]):
            var = tk.BooleanVar(value=True); self.layer_vars[key] = var
            ttk.Checkbutton(self.custom_layers_frame, text=name, variable=var, state='disabled').grid(row=i//3, column=i%3, sticky='w', padx=10, pady=2)
        
        # Start Button
        self.start_button = ttk.Button(control_frame, text="开始训练", command=self.start_training_thread, style='Accent.TButton', padding=10)
        self.start_button.grid(row=5, column=0, columnspan=4, pady=15)

        # 数据加载进度条
        progress_frame = ttk.LabelFrame(self, text="数据加载进度", padding=(10, 5)); progress_frame.pack(fill='x', padx=15, pady=5)
        self.loading_progress = ttk.Progressbar(progress_frame, mode='determinate'); self.loading_progress.pack(fill='x', padx=5, pady=5)
        self.loading_label = ttk.Label(progress_frame, text="数据加载状态: 待启动"); self.loading_label.pack(pady=2)

        # Plotting Frame
        plot_frame = ttk.LabelFrame(self, text="损失曲线", padding=(15, 10)); plot_frame.pack(fill='both', expand=True, padx=15, pady=10)
        plt.style.use('seaborn-v0_8-darkgrid'); self.fig = Figure(figsize=(8, 5), dpi=100); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.init_plot()

    def _update_layer_selection_state(self, *args):
        state = 'normal' if self.training_mode.get() == "custom" else 'disabled'
        for child in self.custom_layers_frame.winfo_children():
            if isinstance(child, ttk.Checkbutton): child.configure(state=state)

    def init_plot(self):
        self.ax.clear(); self.ax.set_title("生成器与判别器损失", fontsize=16); self.ax.set_xlabel("Steps", fontsize=12); self.ax.set_ylabel("Loss", fontsize=12)
        self.train_loss_data = {'steps': [], 'losses': []}; self.disc_loss_data = {'steps': [], 'losses': []}
        self.train_line, = self.ax.plot([], [], 'o-', label='Generator Loss', alpha=0.7, markersize=4)
        self.disc_line, = self.ax.plot([], [], 's-', label='Discriminator Loss', alpha=0.7, markersize=4)
        self.ax.legend(); self.ax.grid(True, which='both', linestyle='--', linewidth=0.5); self.fig.tight_layout(); self.canvas.draw()

    def update_plot(self, loss_type, step, loss_value):
        data, line = (self.train_loss_data, self.train_line) if loss_type == 'train' else (self.disc_loss_data, self.disc_line)
        data['steps'].append(step); data['losses'].append(float(loss_value)); line.set_data(data['steps'], data['losses'])
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()

    def update_loading_progress(self, loaded, total):
        """更新数据加载进度"""
        if total > 0:
            progress = (loaded / total) * 100
            self.loading_progress['value'] = progress
            self.loading_label.config(text=f"数据加载状态: {loaded}/{total} ({progress:.1f}%)")
            if loaded >= total:
                self.loading_label.config(text="数据加载状态: 完成 ✅")

    def select_file(self):
        filepath = filedialog.askopenfilename(title="选择预训练权重", filetypes=[("PyTorch Checkpoints", "*.pt *.ckpt")])
        if filepath: self.pretrained_path_var.set(filepath)

    def start_training_thread(self):
        self.start_button.config(state=tk.DISABLED, text="处理中...")
        self.init_plot()
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def _get_trainable_parameters(self, model):
        mode = self.training_mode.get()
        if mode == "full": return [p for name, p in model.named_parameters() if 'discriminator' not in name]
        
        patterns = []
        if mode == "global_encoder": patterns = ["model.diffusion_model.global_encoder"]
        elif mode == "unet_only": patterns = ["model.diffusion_model.input_blocks", "model.diffusion_model.middle_block", "model.diffusion_model.output_blocks", "model.diffusion_model.time_embed", "model.diffusion_model.out"]
        elif mode == "custom": patterns = [key for key, var in self.layer_vars.items() if var.get()]
        
        trainable_params = []
        for name, param in model.named_parameters():
            if 'discriminator' not in name and any(p in name for p in patterns):
                trainable_params.append(param)
        return trainable_params
    
    def run_training_logic(self):
        try:
            with open(CFG_PATH, 'r', encoding='utf-8') as f: self.cfg = yaml.safe_load(f)
            # 更新config
            for key, var in {
                'pretrained_path': self.pretrained_path_var, 'gradient_accumulation_steps': self.grad_accum_var,
                'fastboot': self.fastboot_var, 'use_jit_compile': self.jit_var, 'use_gradient_checkpointing': self.grad_checkpoint_var,
                'loss_explosion_protection': self.loss_explosion_var, 'explosion_threshold': self.explosion_threshold_var,
                'adaptive_lr': self.adaptive_lr_var, 'lr_reduction_factor': self.lr_reduction_var,
                'early_stopping': self.early_stopping_var, 'patience': self.patience_var
            }.items():
                val = var.get()
                # 类型转换处理
                if key in ['gradient_accumulation_steps', 'patience']:
                    self.cfg['train'][key] = int(val) if isinstance(val, str) else val
                elif key in ['explosion_threshold', 'lr_reduction_factor']:
                    self.cfg['train'][key] = float(val) if isinstance(val, str) else val
                elif key in ['fastboot', 'use_jit_compile', 'use_gradient_checkpointing', 'loss_explosion_protection', 'adaptive_lr', 'early_stopping']:
                    self.cfg['train'][key] = bool(val)
                else:
                    self.cfg['train'][key] = str(val)

            set_seed(self.cfg['experiment']['seed'])
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            acc_steps = self.cfg['train']['gradient_accumulation_steps']
            self.console.print(f"设备: [bold cyan]{device}[/], 梯度累积: {acc_steps}步")

            # 启用快速加载机制
            dataset_config = self.cfg['data'].copy()
            if self.fastboot_var.get():
                dataset_config['fast_loading'] = True
                dataset_config['initial_load_ratio'] = 0.1  # 加载10%的数据
                self.console.print("[cyan]🚀 Fast loading enabled: Will load 10% initially, rest during training[/cyan]")

            train_set = PairedMelDataset(split='train', **dataset_config, preload_to_ram=True)
            train_loader = DataLoader(train_set, batch_size=self.cfg['train']['batch_size'], shuffle=True, num_workers=self.cfg['train']['num_workers'], pin_memory=True)

            # 启动后台加载
            if hasattr(train_set, 'start_background_loading'):
                train_set.start_background_loading()

            # 初始化加载进度显示
            if hasattr(train_set, 'get_loading_progress'):
                loaded, total = train_set.get_loading_progress()
                self.after(0, self.update_loading_progress, loaded, total)
            
            model = instantiate_from_config(self.cfg['model']).to(device)
            load_pretrained(model, self.cfg['train']['pretrained_path'], device, self.console)

            gen_params = self._get_trainable_parameters(model)
            # 检查模型是否有判别器
            if hasattr(model, 'discriminator') and model.discriminator is not None:
                disc_params = list(model.discriminator.parameters())
                has_discriminator = True
            else:
                disc_params = []
                has_discriminator = False
                self.console.print("[yellow]警告: 模型没有判别器，跳过判别器训练[/yellow]")
            
            opt_g = torch.optim.AdamW(gen_params, lr=self.cfg['model']['params']['base_learning_rate'], betas=tuple(self.cfg['train']['betas']))
            if has_discriminator:
                opt_d = torch.optim.AdamW(disc_params, lr=self.cfg['train']['discriminator_lr'], betas=tuple(self.cfg['train']['discriminator_betas']))
            else:
                opt_d = None
            
            ema = EMA(model, decay=self.cfg['train']['ema_decay'])
            loss_monitor = AdvancedLossMonitor(opt_g, self.console, **self.cfg['train']) if self.cfg['train']['loss_explosion_protection'] else None
            
            gstep, outp = 0, os.path.join(self.cfg['experiment']['out_dir'], 'checkpoints')
            os.makedirs(outp, exist_ok=True); model.train()

            with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), MofNCompleteColumn(), TimeElapsedColumn(), console=self.console) as progress:
                epoch_task = progress.add_task("[yellow]总览", total=self.cfg['train']['epochs'])
                action = False  # 初始化action变量
                for ep in range(self.cfg['train']['epochs']):
                    batch_task = progress.add_task(f"[cyan]Epoch {ep+1}", total=len(train_loader))
                    for batch_idx, batch in enumerate(train_loader):
                        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                        # --- 判别器训练 ---
                        if has_discriminator and opt_d is not None:
                            opt_d.zero_grad(set_to_none=True)
                            try:
                                d_loss, d_log = model(batch, optimizer_idx=1)
                                (d_loss / acc_steps).backward()
                                if (batch_idx + 1) % acc_steps == 0: opt_d.step()
                            except Exception as e:
                                self.console.print(f"[red]判别器训练错误: {e}[/red]")
                                d_loss = torch.tensor(0.0, device=device)
                                d_log = {'loss_discriminator': d_loss}
                        else:
                            d_loss = torch.tensor(0.0, device=device)
                            d_log = {'loss_discriminator': d_loss}
                        
                        # --- 生成器训练 ---
                        opt_g.zero_grad(set_to_none=True)
                        g_loss, g_log = model(batch, optimizer_idx=0)
                        (g_loss / acc_steps).backward()

                        if (batch_idx + 1) % acc_steps == 0:
                            torch.nn.utils.clip_grad_norm_(gen_params, 1.0); opt_g.step(); ema.update(model); gstep += 1
                        
                            if loss_monitor:
                                action = loss_monitor.update(g_log['loss_total'].item(), gstep)
                                if action == "stop_training": break
                                if action: continue # skip step if NaN
                            
                            if gstep % self.cfg['train']['log_interval'] == 0:
                                log_str = f"Step:{gstep} G_Loss:{g_log['loss_total']:.4f}"
                                if has_discriminator:
                                    log_str += f" D_Loss:{d_log['loss_discriminator']:.4f}"

                                # 添加数据加载进度
                                if hasattr(train_set, 'get_loading_progress'):
                                    loaded, total = train_set.get_loading_progress()
                                    if loaded < total:
                                        progress_pct = (loaded / total) * 100
                                        log_str += f" | Data: {loaded}/{total} ({progress_pct:.1f}%)"

                                    # 更新GUI进度条
                                    try:
                                        self.after(0, self.update_loading_progress, loaded, total)
                                    except Exception as e:
                                        self.console.print(f"[yellow]加载进度GUI更新失败: {e}[/yellow]")

                                self.console.print(log_str)
                                # 线程安全的GUI更新
                                try:
                                    self.after(0, self.update_plot, 'train', gstep, g_log['loss_total'].item())
                                    if has_discriminator:
                                        self.after(0, self.update_plot, 'discriminator', gstep, d_log['loss_discriminator'].item())
                                except Exception as e:
                                    self.console.print(f"[yellow]警告: GUI更新失败: {e}[/yellow]")

                            if gstep > 0 and gstep % self.cfg['train']['save_interval_steps'] == 0:
                                torch.save({'state_dict': model.state_dict(), 'ema': ema.shadow}, os.path.join(outp, f'step_{gstep}.pt'))
                        
                        progress.update(batch_task, advance=1)
                    progress.remove_task(batch_task); progress.update(epoch_task, advance=1)
                    if action == "stop_training": break
            
            # 停止后台加载
            if hasattr(train_set, 'stop_background_loading'):
                train_set.stop_background_loading()

            final_path = os.path.join(outp, 'final_model.pt')
            torch.save({'state_dict': model.state_dict(), 'ema': ema.shadow}, final_path)
            self.after(0, self.on_training_finish, final_path)

        except Exception:
            self.console.print_exception()
            self.after(0, lambda: messagebox.showerror("训练错误", "发生严重错误，请查看控制台日志。"))
            self.after(0, self.reset_ui)
            
    def reset_ui(self): self.start_button.config(state=tk.NORMAL, text="开始训练")
    def on_training_finish(self, final_path):
        self.reset_ui()
        messagebox.showinfo("完成", f"训练已完成！\n最终模型保存在:\n{final_path}")

if __name__ == '__main__':
    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
        ensure_project_dirs(config)
        app = TrainingApp(config)
        app.mainloop()
    except FileNotFoundError: print(f"错误: 配置文件 'config.yaml' 未找到。")
    except Exception as e: print(f"启动 GUI 时发生严重错误: {e}")

