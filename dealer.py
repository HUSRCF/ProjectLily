import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from pydub import AudioSegment
import threading
import multiprocessing
import queue
import time
from tqdm import tqdm

# --- Core Audio Processing Logic (for worker processes) ---

def worker_process_folder(task_args):
    """
    This function runs in a separate process. It processes a single folder and
    displays its own tqdm progress bar.

    Args:
        task_args (tuple): A tuple containing (input_folder, output_file, position).
                           'position' is for placing the tqdm bar correctly.

    Returns:
        tuple: A tuple containing (status, message).
    """
    input_folder, output_file, position = task_args
    folder_name = os.path.basename(input_folder)
    pid = os.getpid()
    
    supported_formats = ('.wav', '.flac')
    
    try:
        audio_files = sorted(
            [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
        )

        if not audio_files:
            return ('SKIPPED', f"文件夹 '{folder_name}' 中无音频文件。")

        combined_audio = None
        
        # Create a tqdm progress bar for this specific worker process
        progress_bar = tqdm(
            total=len(audio_files),
            desc=f"PID {pid} | {folder_name[:25]:<25}", # Truncate and pad folder name
            position=position + 1, # Position 0 is for the main progress bar
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )

        for filename in audio_files:
            filepath = os.path.join(input_folder, filename)
            file_format = os.path.splitext(filename)[1][1:].lower()
            try:
                audio_segment = AudioSegment.from_file(filepath, format=file_format)
                if combined_audio is None:
                    combined_audio = audio_segment
                else:
                    combined_audio += audio_segment
            except Exception:
                continue
            finally:
                progress_bar.update(1)
        
        progress_bar.close()

        if combined_audio:
            combined_audio.export(output_file, format="wav")
            result = ('SUCCESS', f"已处理 '{folder_name}' -> {os.path.basename(output_file)}")
        else:
            result = ('FAILURE', f"未能从 '{folder_name}' 加载任何音频。")

    except Exception as e:
        result = ('ERROR', f"处理 '{folder_name}' 时出错: {e}")

    return result


# --- Process Management (runs in a separate thread) ---

def start_processing_manager(is_batch_mode, input_path, output_path, update_queue):
    """
    Manages the processing. For batch mode, it sets up a multiprocessing Pool
    and a main tqdm progress bar.
    """
    print("\n" + "="*80)
    print("--- [Manager] Processing Manager Thread Started ---")
    start_time = time.time()

    if not is_batch_mode:
        print(f"--- [Manager] Running in Single Folder Mode for: {input_path}")
        # Single mode doesn't need a worker position
        status, message = worker_process_folder((input_path, output_path, 0))
        if status == 'SUCCESS':
            update_queue.put(('STATUS', f"成功！音频已拼接并保存至: {output_path}"))
            update_queue.put(('DONE_SINGLE', output_path))
        else:
            update_queue.put(('STATUS', f"错误: {message}"))
            update_queue.put(('DONE_ERROR', message))
    else:
        print("--- [Manager] Running in Batch Processing Mode ---")
        update_queue.put(('STATUS', "正在扫描子文件夹..."))
        
        raw_tasks = []
        for subdir, _, _ in os.walk(input_path):
            if subdir == input_path: continue
            if any(f.lower().endswith(('.wav', '.flac')) for f in os.listdir(subdir)):
                folder_name = os.path.basename(subdir)
                output_filename = f"{folder_name}.wav"
                output_filepath = os.path.join(output_path, output_filename)
                raw_tasks.append((subdir, output_filepath))
        
        if not raw_tasks:
            update_queue.put(('STATUS', "错误: 未在任何子文件夹中找到可处理的音频文件。"))
            update_queue.put(('DONE_ERROR', "无可处理文件")); return

        num_processes = multiprocessing.cpu_count()
        # Assign a position to each task for its progress bar
        tasks_with_pos = [(task[0], task[1], i % num_processes) for i, task in enumerate(raw_tasks)]
        
        print(f"--- [Manager] Found {len(tasks_with_pos)} tasks. Creating a Pool with {num_processes} workers. ---")
        update_queue.put(('MAX_PROGRESS', len(tasks_with_pos)))
        update_queue.put(('STATUS', f"开始使用 {num_processes} 个CPU核心进行并行处理..."))
        
        results = []
        # Create a main progress bar for overall progress
        with tqdm(total=len(tasks_with_pos), desc="Overall Progress", position=0) as main_pbar:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap_unordered to get results as they complete
                for result in pool.imap_unordered(worker_process_folder, tasks_with_pos):
                    results.append(result)
                    main_pbar.update(1)
        
        print("\n--- [Manager] All worker processes have finished. Final Summary: ---")
        for res in results: print(f"  - Status: {res[0]}, Details: {res[1]}")
        
        success_count = sum(1 for r in results if r[0] == 'SUCCESS')
        update_queue.put(('FINAL_RESULTS', (success_count, len(tasks_with_pos), output_path)))

    end_time = time.time()
    print(f"--- [Manager] Processing Finished. Total time: {end_time - start_time:.2f} seconds. ---")
    print("="*80 + "\n")


# --- GUI Application Class ---

class AudioSplicerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音频拼接工具 (多进程加速版)")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # Variables
        self.input_folder_path = tk.StringVar(value="尚未选择文件夹")
        self.output_path = tk.StringVar(value="尚未选择输出路径")
        self.batch_mode = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="欢迎使用音频拼接工具！")
        self.cpu_count = multiprocessing.cpu_count()
        self.update_queue = queue.Queue()

        # --- GUI Layout ---
        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input/Output frames
        self.create_io_widgets(main_frame)

        # Batch mode checkbox and CPU info
        batch_frame = tk.Frame(main_frame)
        batch_frame.pack(fill=tk.X, pady=10)
        batch_check = tk.Checkbutton(batch_frame, text="启用批量处理模式 (并行处理子文件夹)", variable=self.batch_mode, command=self.toggle_batch_mode)
        batch_check.pack(side=tk.LEFT)
        self.cpu_label = tk.Label(batch_frame, text=f"CPU核心数: {self.cpu_count}", fg="grey")
        self.cpu_label.pack(side=tk.RIGHT)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Start Button
        self.start_button = tk.Button(main_frame, text="开始拼接", command=self.start_splicing_thread, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white")
        self.start_button.pack(pady=20, ipadx=20, ipady=5)

        # Status Bar
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_io_widgets(self, parent):
        input_frame = tk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)
        tk.Button(input_frame, text="选择根文件夹", command=self.select_input_folder, width=15).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(input_frame, textvariable=self.input_folder_path, relief=tk.SUNKEN, bg="white", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        output_frame = tk.Frame(parent)
        output_frame.pack(fill=tk.X, pady=5)
        self.output_btn = tk.Button(output_frame, text="选择输出文件", command=self.select_output_path, width=15)
        self.output_btn.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(output_frame, textvariable=self.output_path, relief=tk.SUNKEN, bg="white", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    def toggle_batch_mode(self):
        if self.batch_mode.get():
            self.output_btn.config(text="选择输出文件夹")
            self.output_path.set("尚未选择输出文件夹")
            self.status_var.set("批量处理模式已启用。")
        else:
            self.output_btn.config(text="选择输出文件")
            self.output_path.set("尚未选择输出路径")
            self.status_var.set("批量处理模式已禁用。")

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择包含音频的根文件夹")
        if folder: self.input_folder_path.set(folder)

    def select_output_path(self):
        if self.batch_mode.get():
            path = filedialog.askdirectory(title="选择保存拼接文件的文件夹")
        else:
            path = filedialog.asksaveasfilename(title="选择输出文件", defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if path: self.output_path.set(path)

    def start_splicing_thread(self):
        input_p, output_p = self.input_folder_path.get(), self.output_path.get()
        is_batch = self.batch_mode.get()
        
        # --- Input Validation ---
        if not os.path.isdir(input_p):
            messagebox.showerror("输入错误", "请输入一个有效的根文件夹路径。"); return
        if is_batch and not os.path.isdir(output_p):
            messagebox.showerror("输出错误", "批量模式下，请选择一个有效的输出文件夹。"); return
        if not is_batch and (not output_p or "尚未选择" in output_p):
            messagebox.showerror("输出错误", "请选择一个有效的输出文件路径。"); return

        self.start_button.config(state=tk.DISABLED, text="正在处理...")
        self.progress_bar['value'] = 0

        # --- Start the manager thread ---
        self.processing_thread = threading.Thread(
            target=start_processing_manager,
            args=(is_batch, input_p, output_p, self.update_queue)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # --- Start polling the queue for updates ---
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """ Periodically checks the queue for messages from the processing thread/processes. """
        try:
            message_type, data = self.update_queue.get_nowait()
            
            if message_type == 'STATUS':
                self.status_var.set(data)
            elif message_type == 'MAX_PROGRESS':
                self.progress_bar['maximum'] = data
                self.progress_bar['value'] = 0 # Reset progress on new task
            elif message_type == 'UPDATE_PROGRESS':
                self.progress_bar['value'] = data
            elif message_type == 'DONE_SINGLE':
                self.progress_bar['value'] = 1
                self.progress_bar['maximum'] = 1
                messagebox.showinfo("完成", f"音频拼接完成！\n文件已保存到:\n{data}")
                self.reset_ui()
            elif message_type == 'DONE_ERROR':
                messagebox.showerror("错误", data)
                self.reset_ui()
            elif message_type == 'FINAL_RESULTS':
                success_count, total_tasks, output_path = data
                self.progress_bar['value'] = self.progress_bar['maximum']
                final_message = f"批量处理完成！\n成功处理了 {success_count} / {total_tasks} 个子文件夹。"
                self.status_var.set(final_message)
                messagebox.showinfo("批量完成", f"{final_message}\n文件已保存到:\n{output_path}")
                self.reset_ui()

        except queue.Empty:
            pass # No new messages
        
        if self.processing_thread.is_alive():
            # Update GUI progress bar from main thread
            if self.batch_mode.get() and self.progress_bar['maximum'] > 0:
                 # This is tricky as results come unordered. We just count completed tasks.
                 # The terminal provides the real-time feedback.
                 pass
            self.root.after(100, self.process_queue)
        else:
            if self.start_button['state'] == tk.DISABLED:
                 try:
                    # Check one last time for any remaining messages
                    while not self.update_queue.empty():
                        self.process_queue()
                 except queue.Empty:
                    self.reset_ui()

    def reset_ui(self):
        """Resets the button and progress bar to their initial state."""
        self.start_button.config(state=tk.NORMAL, text="开始拼接")
        self.progress_bar['value'] = 0


# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    main_root = tk.Tk()
    app = AudioSplicerApp(main_root)
    main_root.mainloop()
