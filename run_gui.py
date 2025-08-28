import os
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import random
import shutil
import subprocess

# --- This script is the unified GUI for the 2D Spectrogram pipeline ---

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.append(BASE)

# Import the backend modules for preprocessing
import preprocess_audio as prep

CFG_PATH = os.path.join(BASE, 'config.yaml')

def ensure_dirs(cfg):
    """Ensures that all necessary directories from the config file exist."""
    # This function now uses paths from the new config structure
    dataset_root = os.path.join(BASE, cfg['data']['dataset_root'])
    # Simplified to ensure train/valid parent folders exist.
    os.makedirs(os.path.join(dataset_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(BASE, cfg['experiment']['out_dir']), exist_ok=True)

class AudioSR_GUI(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title("AudioSR 2D Spectrogram Pipeline (Desktop GUI)")
        self.geometry("800x700") # Adjusted height for new options

        style = ttk.Style(self)
        style.theme_use('clam')

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.create_preprocess_tab()
        self.create_train_tab()
        self.create_inference_tab()

    def create_preprocess_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='数据准备 / Data Prep')

        prep_frame = ttk.LabelFrame(tab, text="第一步: 生成训练数据", padding=(10, 5))
        prep_frame.pack(fill='x', padx=10, pady=5, anchor='n')
        
        self.in_dir_var = tk.StringVar(value='raw_audio')
        self.out_train_high_var = tk.StringVar(value=os.path.join(self.cfg['data']['dataset_root'], 'train', self.cfg['data']['high_dir_name']))
        self.out_train_low_var = tk.StringVar(value=os.path.join(self.cfg['data']['dataset_root'], 'train', self.cfg['data']['low_dir_name']))

        ttk.Label(prep_frame, text="原始音频目录:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(prep_frame, textvariable=self.in_dir_var, width=80).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(prep_frame, text="训练集输出 (高):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(prep_frame, textvariable=self.out_train_high_var, state='readonly').grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(prep_frame, text="训练集输出 (低):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(prep_frame, textvariable=self.out_train_low_var, state='readonly').grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        prep_frame.columnconfigure(1, weight=1)

        self.btn_p = ttk.Button(prep_frame, text="开始预处理", command=self.start_preprocess)
        self.btn_p.grid(row=4, column=0, columnspan=2, pady=10)
        
        split_frame = ttk.LabelFrame(tab, text="第二步: 划分验证集", padding=(10, 5))
        split_frame.pack(fill='x', padx=10, pady=10, anchor='n')
        
        self.valid_ratio_var = tk.DoubleVar(value=5.0)
        
        ttk.Label(split_frame, text="验证集比例 (%):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.ratio_slider = ttk.Scale(split_frame, from_=0, to=30, orient='horizontal', variable=self.valid_ratio_var, length=200)
        self.ratio_slider.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.ratio_label = ttk.Label(split_frame, text=f"{self.valid_ratio_var.get():.1f}%")
        self.ratio_label.grid(row=0, column=2, padx=10, pady=5)
        self.valid_ratio_var.trace_add('write', self.update_ratio_label)
        
        self.btn_s = ttk.Button(split_frame, text="开始划分", command=self.start_split_valid_set)
        self.btn_s.grid(row=1, column=0, columnspan=3, pady=10)
        split_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(tab, text="日志", padding=(10, 5))
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_p = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_p.pack(fill='both', expand=True)

    def update_ratio_label(self, *args):
        self.ratio_label.config(text=f"{self.valid_ratio_var.get():.1f}%")

    def create_train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='训练 / Train')

        ft_frame = ttk.LabelFrame(tab, text="微调 / 预训练设置", padding=(10, 5))
        ft_frame.pack(fill='x', padx=10, pady=10)

        self.pretrained_path_var = tk.StringVar(value=self.cfg['train'].get('pretrained_path', ''))
        self.key_filter_var = tk.StringVar(value=self.cfg['train'].get('key_filter_contains', 'model.diffusion_model.'))
        self.freeze_strategy_var = tk.StringVar(value="不冻结 (No Freeze)")
        self.freeze_custom_var = tk.StringVar(value=", ".join(self.cfg['train'].get('freeze_substrings', [])))

        ttk.Label(ft_frame, text="预训练权重路径:").grid(row=0, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(ft_frame, textvariable=self.pretrained_path_var, width=70).grid(row=0, column=1, sticky='ew', padx=5, pady=3)
        ttk.Button(ft_frame, text="浏览...", command=lambda: self.select_file(self.pretrained_path_var, "选择预训练权重")).grid(row=0, column=2, padx=5, pady=3)

        ttk.Label(ft_frame, text="权重键过滤器:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(ft_frame, textvariable=self.key_filter_var, width=70).grid(row=1, column=1, sticky='ew', padx=5, pady=3)
        
        ttk.Label(ft_frame, text="冻结策略:").grid(row=2, column=0, sticky='w', padx=5, pady=3)
        freeze_options = ["不冻结 (No Freeze)", "冻结编码器 (Freeze Encoder)", "仅微调注意力层 (Finetune Attention)", "自定义 (Custom)"]
        self.freeze_menu = ttk.OptionMenu(ft_frame, self.freeze_strategy_var, freeze_options[0], *freeze_options, command=self.on_freeze_strategy_change)
        self.freeze_menu.grid(row=2, column=1, sticky='w', padx=5, pady=3)

        self.custom_freeze_label = ttk.Label(ft_frame, text="自定义冻结层:")
        self.custom_freeze_entry = ttk.Entry(ft_frame, textvariable=self.freeze_custom_var, width=70)
        
        ft_frame.columnconfigure(1, weight=1)

        info_text = ("提示: 其他训练参数 (如模型架构、学习率等) 请直接修改 `config.yaml` 文件。")
        ttk.Label(tab, text=info_text, wraplength=750, justify='center').pack(pady=5)

        self.btn_t = ttk.Button(tab, text="开始训练", command=self.start_train, style='Accent.TButton')
        self.btn_t.pack(pady=20)

        log_frame = ttk.LabelFrame(tab, text="训练日志", padding=(10, 5))
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.log_t = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_t.pack(fill='both', expand=True)

    def on_freeze_strategy_change(self, strategy):
        if strategy == "自定义 (Custom)":
            self.custom_freeze_label.grid(row=3, column=0, sticky='w', padx=5, pady=3)
            self.custom_freeze_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=3)
        else:
            self.custom_freeze_label.grid_forget()
            self.custom_freeze_entry.grid_forget()

    def create_inference_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='推理 / Inference')
        
        info_frame = ttk.LabelFrame(tab, text="功能说明", padding=(10, 5))
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        info_text = ("推理功能正在开发中。\n\n"
                     "2D频谱工作流的模型输出的是频谱图，需要一个声码器 (Vocoder) "
                     "或使用 Griffin-Lim 算法才能将其转换回音频波形。\n\n"
                     "我们将在后续步骤中构建此功能。")
        ttk.Label(info_frame, text=info_text, wraplength=750, justify='center', font=("", 12)).pack(pady=20, padx=10, expand=True)

    def run_in_thread(self, target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    def select_file(self, string_var, title):
        filepath = filedialog.askopenfilename(title=title, filetypes=(("PyTorch Checkpoints", "*.pt *.ckpt"), ("All files", "*.*")))
        if filepath:
            string_var.set(filepath)

    def start_preprocess(self):
        self.btn_p.config(state=tk.DISABLED)
        self.log_p.delete('1.0', tk.END)
        self.log_p.insert(tk.END, "开始预处理...\n")
        self.run_in_thread(self._preprocess_task)

    def _preprocess_task(self):
        try:
            in_dir_abs = os.path.join(BASE, self.in_dir_var.get())
            out_high_abs = os.path.join(BASE, self.out_train_high_var.get())
            out_low_abs = os.path.join(BASE, self.out_train_low_var.get())
            processed = prep.preprocess_dir(in_dir_abs, out_high_abs, out_low_abs, sr=self.cfg['data']['sample_rate'], low_sr=12000)
            log_msg = '\n'.join([f'Processed: {x}' for x in processed]) or 'No files found.'
            self.log_p.insert(tk.END, log_msg + "\n预处理完成。\n")
        except Exception as e:
            self.log_p.insert(tk.END, f"[ERROR] {repr(e)}\n")
        finally:
            self.btn_p.config(state=tk.NORMAL)

    def start_split_valid_set(self):
        self.btn_s.config(state=tk.DISABLED)
        self.log_p.insert(tk.END, "\n开始划分验证集...\n")
        self.run_in_thread(self._split_valid_set_task)

    def _split_valid_set_task(self):
        try:
            ratio = self.valid_ratio_var.get()
            if ratio <= 0:
                self.log_p.insert(tk.END, "划分比例为0，无需操作。\n")
                return

            train_high_path = os.path.join(BASE, self.out_train_high_var.get())
            train_low_path = os.path.join(BASE, self.out_train_low_var.get())
            valid_high_path = os.path.join(BASE, self.cfg['data']['dataset_root'], 'valid', self.cfg['data']['high_dir_name'])
            valid_low_path = os.path.join(BASE, self.cfg['data']['dataset_root'], 'valid', self.cfg['data']['low_dir_name'])
            
            os.makedirs(valid_high_path, exist_ok=True)
            os.makedirs(valid_low_path, exist_ok=True)

            if not os.path.exists(train_high_path):
                raise FileNotFoundError(f"训练集目录不存在: {train_high_path}")

            files = [f for f in os.listdir(train_high_path) if f.lower().endswith(('.wav', '.flac', '.ogg', '.mp3'))]
            if not files:
                self.log_p.insert(tk.END, "训练集中没有找到音频文件，无法划分。\n")
                return
            
            num_to_move = int(len(files) * (ratio / 100.0))
            if num_to_move == 0 and len(files) > 0:
                 self.log_p.insert(tk.END, f"文件总数太少 ({len(files)})，按比例计算需要移动0个文件。\n")
                 return

            files_to_move = random.sample(files, num_to_move)
            moved_count = 0
            for f in files_to_move:
                src_high, dst_high = os.path.join(train_high_path, f), os.path.join(valid_high_path, f)
                src_low, dst_low = os.path.join(train_low_path, f), os.path.join(valid_low_path, f)
                if os.path.exists(src_high) and os.path.exists(src_low):
                    shutil.move(src_high, dst_high)
                    shutil.move(src_low, dst_low)
                    self.log_p.insert(tk.END, f"  移动: {f}\n")
                    moved_count += 1
            
            self.log_p.insert(tk.END, f"\n划分完成！共移动 {moved_count} 对文件到验证集。\n")
        except Exception as e:
            self.log_p.insert(tk.END, f"[ERROR] {repr(e)}\n")
        finally:
            self.btn_s.config(state=tk.NORMAL)

    def start_train(self):
        self.btn_t.config(state=tk.DISABLED)
        self.log_t.delete('1.0', tk.END)
        self.log_t.insert(tk.END, "准备启动训练...\n")
        self.run_in_thread(self._train_task)

    def _train_task(self):
        try:
            with open(CFG_PATH, 'r', encoding='utf-8') as f:
                current_cfg = yaml.safe_load(f)
            
            current_cfg['train']['pretrained_path'] = self.pretrained_path_var.get() or None
            current_cfg['train']['key_filter_contains'] = self.key_filter_var.get() or None

            strategy = self.freeze_strategy_var.get()
            freeze_list = []
            if strategy == "冻结编码器 (Freeze Encoder)":
                freeze_list = ["input_blocks."]
            elif strategy == "仅微调注意力层 (Finetune Attention)":
                self.log_t.insert(tk.END, "注意: '仅微调注意力层'策略会冻结所有非注意力块。\n")
                # This freezes all ResBlocks and convs, leaving AttnBlocks trainable
                freeze_list = ["input_blocks.0", "input_blocks.1.0", "input_blocks.2.0", "input_blocks.3.0", "input_blocks.4.0", "input_blocks.5.0", "input_blocks.6.0", "input_blocks.7.0", "input_blocks.8.0", "input_blocks.9.0", "input_blocks.10.0", "input_blocks.11.0", "middle_block.0", "middle_block.2", "output_blocks.", "out."]
            elif strategy == "自定义 (Custom)":
                freeze_str = self.freeze_custom_var.get()
                if freeze_str:
                    freeze_list = [s.strip() for s in freeze_str.split(',') if s.strip()]
            current_cfg['train']['freeze_substrings'] = freeze_list

            with open(CFG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(current_cfg, f, sort_keys=False, allow_unicode=True)

            self.log_t.insert(tk.END, f"配置文件已更新。\n使用预训练权重: {current_cfg['train']['pretrained_path']}\n冻结策略: '{strategy}' -> {freeze_list}\n")
            self.log_t.insert(tk.END, f"启动 2D 频谱管线...\n")

            cmd = [sys.executable, os.path.join(BASE, 'train.py'), '--config', CFG_PATH]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in iter(process.stdout.readline, ''):
                self.log_t.insert(tk.END, line)
                self.log_t.see(tk.END)
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                self.log_t.insert(tk.END, f"\n--- 训练进程异常结束，返回码: {return_code} ---\n")
            else:
                self.log_t.insert(tk.END, f"\n--- 训练进程正常结束 ---\n")
        except Exception as e:
            self.log_t.insert(tk.END, f"\n[GUI ERROR] 启动训练失败: {repr(e)}\n")
        finally:
            self.btn_t.config(state=tk.NORMAL)

if __name__ == '__main__':
    try:
        with open(CFG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ensure_dirs(config)
        app = AudioSR_GUI(config)
        app.mainloop()
    except FileNotFoundError:
        print(f"错误: 配置文件 'config.yaml' 未找到。请确保它在脚本所在的目录中。")
    except Exception as e:
        print(f"启动时发生错误: {e}")
