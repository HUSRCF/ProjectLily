import torch
import argparse
from typing import Dict, Any
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading

def inspect_checkpoint(ckpt_path: str, log_widget: scrolledtext.ScrolledText):
    """
    加载一个 PyTorch 检查点文件 (.pt, .ckpt) 并将其内部结构
    打印到指定的 Tkinter ScrolledText 组件中。
    """
    def log(message):
        log_widget.insert(tk.END, message + "\n")
        log_widget.see(tk.END)

    log(f"--- 正在检查权重文件: {ckpt_path} ---")

    try:
        # 加载到 CPU 以避免占用显存
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        log(f"\n[错误] 无法加载文件: {e}")
        return

    if not isinstance(ckpt, dict):
        log("\n[错误] 权重文件不是一个字典 (dictionary)。")
        return

    log("\n--- 文件顶层键 (Top-Level Keys) ---")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            log(f"- {key} (包含 {len(ckpt[key])} 个子键)")
        else:
            log(f"- {key} (类型: {type(ckpt[key])})")

    def print_state_dict(prefix: str, sd: Dict[str, Any]):
        log(f"\n--- 权重字典详情: '{prefix}' ---")
        if not sd:
            log("  (此字典为空)")
            return
        
        max_key_len = max(len(k) for k in sd.keys()) if sd else 0
        
        for name, param in sd.items():
            if isinstance(param, torch.Tensor):
                shape_str = str(list(param.shape))
                log(f"  {name:<{max_key_len}} | 形状: {shape_str}")
            else:
                log(f"  {name:<{max_key_len}} | (非张量, 类型: {type(param)})")

    # 递归地查找并打印所有 state_dict
    def find_and_print_sds(data, prefix=""):
        if isinstance(data, dict):
            is_state_dict = all(isinstance(v, torch.Tensor) for v in data.values()) and data
            
            if is_state_dict:
                print_state_dict(prefix if prefix else "root", data)
            else:
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    find_and_print_sds(value, new_prefix)

    find_and_print_sds(ckpt)
    log("\n--- 检查完毕 ---")


class CheckpointInspectorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyTorch Checkpoint Inspector")
        self.geometry("900x700")

        # --- UI Elements ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="选择权重文件", padding="10")
        file_frame.pack(fill="x", pady=5)

        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(file_frame, textvariable=self.path_var, width=80)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.browse_button = ttk.Button(file_frame, text="浏览...", command=self.browse_file)
        self.browse_button.pack(side="left")

        # Inspect button
        self.inspect_button = ttk.Button(main_frame, text="开始检查", command=self.start_inspection)
        self.inspect_button.pack(pady=10)

        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="检查结果", padding="10")
        log_frame.pack(fill="both", expand=True)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_widget.pack(fill="both", expand=True)

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=(("PyTorch Checkpoints", "*.pt *.ckpt"), ("All files", "*.*"))
        )
        if filepath:
            self.path_var.set(filepath)

    def start_inspection(self):
        ckpt_path = self.path_var.get()
        if not ckpt_path:
            self.log_widget.delete('1.0', tk.END)
            self.log_widget.insert(tk.END, "[错误] 请先选择一个文件路径。")
            return

        self.log_widget.delete('1.0', tk.END)
        self.inspect_button.config(state="disabled")
        
        # Run inspection in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=self._run_inspection_task, args=(ckpt_path,), daemon=True)
        thread.start()

    def _run_inspection_task(self, ckpt_path):
        try:
            inspect_checkpoint(ckpt_path, self.log_widget)
        except Exception as e:
            self.log_widget.insert(tk.END, f"\n[致命错误] 发生意外: {repr(e)}")
        finally:
            self.inspect_button.config(state="normal")


if __name__ == "__main__":
    app = CheckpointInspectorGUI()
    app.mainloop()
