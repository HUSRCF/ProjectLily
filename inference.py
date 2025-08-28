import os
import sys
import yaml
import threading
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.signal import butter, lfilter
import traceback # Import the standard traceback module

# Set offline mode to avoid network issues with transformers
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1' 
os.environ['HF_DATASETS_OFFLINE'] = '1'

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Plotting Imports ---
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Local Project Imports ---
# Make sure model.py and audiosr folder are in the same directory
from model import LatentDiffusion
from audiosr.latent_diffusion.util import instantiate_from_config
from audiosr.utils import _locate_cutoff_freq
from audiosr.lowpass import lowpass
# --- HiFi-GAN Imports ---
# Note: The Vocoder class is already defined in your model.py inside AutoencoderKL,
# so we don't need to import it separately if the weights are in the main checkpoint.

# =====================================================================================
# SECTION: CORE INFERENCE AND AUDIO PROCESSING LOGIC
# =====================================================================================

class InferenceEngine:
    """
    Handles the core logic for audio processing, model inference, and file I/O.
    """
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.model = self.load_model()
        # The vocoder is part of the first_stage_model and is loaded with it.
        # No separate setup_transforms needed for vocoder if it's integrated.
        self.setup_mel_transform()

    def load_model(self):
        """Instantiates the model from the config."""
        model = instantiate_from_config(self.cfg['model']).to(self.device)
        print(f"模型已在设备上实例化: {self.device}")
        return model

    def load_weights(self, weights_path):
        """Loads weights into the model and handles EMA."""
        print(f"正在从以下路径加载权重: {weights_path}")
        ckpt = torch.load(weights_path, map_location=self.device)
        
        # Try to load AudioSR checkpoint with vocoder weights if available
        audiosr_checkpoint_path = "/home/husrcf/.cache/huggingface/hub/models--haoheliu--audiosr_basic/snapshots/74a47f49061a1e788e968cc43ad45c0b6243f37d/pytorch_model.bin"
        audiosr_ckpt = None
        if os.path.exists(audiosr_checkpoint_path):
            print("正在加载 AudioSR 检查点以获取 vocoder 权重...")
            # Load on CPU first to avoid memory issues, then move only vocoder weights to device
            audiosr_ckpt = torch.load(audiosr_checkpoint_path, map_location='cpu')
        
        if "ema" in ckpt and ckpt["ema"]:
            print("正在加载 EMA 权重...")
            ema_shadow_params = ckpt["ema"]
            
            new_ema_state_dict = {}
            for name, p_ema in ema_shadow_params.items():
                if name.startswith('model.'):
                    internal_name = name[len('model.'):]
                    buffer_name = internal_name.replace('.', '')
                    new_ema_state_dict[buffer_name] = p_ema

            if not new_ema_state_dict:
                print("警告: 在检查点中找到EMA权重，但模型的EMA模块中没有匹配的参数。将回退到标准权重。")
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                missing, unexpected = self.model.model_ema.load_state_dict(new_ema_state_dict, strict=False)
                print(f"EMA 权重已加载。缺失键: {len(missing)}, 意外键: {len(unexpected)}")
                self.model.model_ema.copy_to(self.model.model)

        elif "state_dict" in ckpt:
            print("正在加载标准 state_dict...")
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            print("正在加载原始权重...")
            self.model.load_state_dict(ckpt, strict=False)
        
        # Load AudioSR vocoder weights if available
        if audiosr_ckpt is not None:
            print("正在加载 AudioSR vocoder 权重...")
            audiosr_state_dict = audiosr_ckpt["state_dict"] if "state_dict" in audiosr_ckpt else audiosr_ckpt
            
            # Extract vocoder weights and move to device
            vocoder_weights = {}
            for key, value in audiosr_state_dict.items():
                if key.startswith('first_stage_model.vocoder.'):
                    new_key = key.replace('first_stage_model.vocoder.', 'first_stage_model.vocoder.')
                    vocoder_weights[new_key] = value.to(self.device)
            
            if vocoder_weights:
                missing_vocoder, unexpected_vocoder = self.model.load_state_dict(vocoder_weights, strict=False)
                print(f"AudioSR vocoder 权重已加载。缺失键: {len(missing_vocoder)}, 意外键: {len(unexpected_vocoder)}")
                # Clear the large checkpoint from memory
                del audiosr_ckpt, audiosr_state_dict, vocoder_weights
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Remove weight norm from vocoder (required for HiFi-GAN)
                if hasattr(self.model.first_stage_model, 'vocoder') and hasattr(self.model.first_stage_model.vocoder, 'remove_weight_norm'):
                    try:
                        self.model.first_stage_model.vocoder.remove_weight_norm()
                        print("已移除 vocoder 的 weight normalization")
                    except ValueError as e:
                        if "weight_norm" in str(e):
                            print(f"警告: 部分层没有 weight normalization，跳过移除: {e}")
                        else:
                            raise e
            else:
                print("警告: 未在 AudioSR 检查点中找到 vocoder 权重")
        
        self.model.eval()
        print("权重加载成功，模型已进入评估模式。")
    
    def _safe_remove_weight_norm(self, module):
        """
        Safely remove weight normalization from modules where it exists.
        """
        from torch.nn.utils import remove_weight_norm
        
        for name, child in module.named_modules():
            # Check each parameter to see if weight_norm is applied
            for param_name, _ in child.named_parameters(recurse=False):
                if param_name == 'weight_g' or param_name == 'weight_v':
                    # This indicates weight normalization is applied
                    try:
                        remove_weight_norm(child, 'weight')
                        print(f"已移除 {name} 的 weight normalization")
                        break  # Only one weight norm per module
                    except ValueError:
                        continue  # Skip if weight norm not found

    def setup_mel_transform(self):
        """Sets up AudioSR-exact mel spectrogram transformation."""
        # Use AudioSR's exact mel processing
        from audiosr.utils import mel_spectrogram_train, spectral_normalize_torch
        
        # Store the functions for use in preprocessing
        self.mel_spectrogram_train = mel_spectrogram_train
        self.spectral_normalize_torch = spectral_normalize_torch
        
        print("🎵 Using AudioSR-exact mel processing pipeline")

    def _preprocess_audio(self, audio_path):
        """Loads, resamples, and creates a low-pass version of the audio."""
        sr = self.cfg['data']['sample_rate']
        audio, orig_sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if orig_sr != sr:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)
        
        # CRITICAL FIX: Normalize audio to handle silent/very quiet inputs
        current_rms = torch.sqrt(torch.mean(audio**2))
        if current_rms < 0.001:  # Very quiet audio
            print(f"⚠️  检测到安静音频 (RMS: {current_rms:.6f})，正在标准化...")
            target_rms = 0.05
            scale_factor = target_rms / (current_rms + 1e-8)
            scale_factor = min(scale_factor, 1000.0)  # Cap scaling
            audio = audio * scale_factor
            # Prevent clipping
            if audio.abs().max() > 0.95:
                audio = audio * (0.95 / audio.abs().max())
            new_rms = torch.sqrt(torch.mean(audio**2))
            print(f"   标准化完成: RMS {current_rms:.6f} -> {new_rms:.6f}")
        
        window = torch.hann_window(2048, device=self.device).to(torch.float32)
        
        stft_for_cutoff = torch.stft(audio.to(self.device).squeeze(0), n_fft=2048, hop_length=480, win_length=2048, window=window, return_complex=True, center=True)
        
        cutoff_freq = (_locate_cutoff_freq(stft_for_cutoff.abs(), percentile=0.985) / 1024) * (sr / 2)
        
        nyquist = sr / 2
        cutoff_freq = np.clip(cutoff_freq, 10.0, nyquist - 1.0)
        
        low_quality_waveform = lowpass(audio.cpu().numpy().squeeze(), highcut=cutoff_freq, fs=sr, order=8, _type="butter")
        
        return torch.from_numpy(low_quality_waveform.copy()).to(self.device).unsqueeze(0), audio.squeeze()

    @torch.no_grad()
    def _ddim_sample(self, start_noise, cond, steps=100, eta=1.0):
        """Performs DDIM sampling to generate audio from noise."""
        shape = start_noise.shape
        b = shape[0]
        t_total = self.model.num_timesteps
        
        timesteps = torch.linspace(t_total - 1, 0, steps, device=self.device).long()
        x_t = start_noise
        
        for i, t_curr in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < steps - 1 else torch.tensor(-1, device=self.device)
            model_output = self.model.apply_model(x_t, t_curr.expand(b), cond)
            
            if self.model.parameterization == "v":
                sqrt_alpha_prod = self.model.sqrt_alphas_cumprod[t_curr]
                sqrt_one_minus_alpha_prod = self.model.sqrt_one_minus_alphas_cumprod[t_curr]
                x0_pred = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * model_output
            else:
                x0_pred = (x_t - self.model.sqrt_one_minus_alphas_cumprod[t_curr] * model_output) / self.model.sqrt_alphas_cumprod[t_curr]
            
            x0_pred.clamp_(-1., 1.)
            
            if t_prev < 0:
                x_t = x0_pred
                continue
            
            alpha_prod_t_prev = self.model.alphas_cumprod_prev[t_curr]
            dir_xt = (1. - alpha_prod_t_prev - (eta * self.model.posterior_variance[t_curr])).sqrt() * model_output
            x_prev = alpha_prod_t_prev.sqrt() * x0_pred + dir_xt
            x_t = x_prev
            
        return x_t

    def run_inference(self, audio_path, output_dir, progress_callback, **kwargs):
        """Main inference function with chunking, stitching, and HiFi-GAN vocoder."""
        try:
            # --- 1. Preprocessing ---
            print("\n--- 步骤 1: 预处理音频 ---")
            sr = self.cfg['data']['sample_rate']
            low_quality_waveform, original_waveform = self._preprocess_audio(audio_path)
            print("音频已加载并预处理。")
            
            # --- 2. Chunking Setup ---
            print("\n--- 步骤 2: 音频分块 ---")
            chunk_seconds = self.cfg['data']['segment_seconds']
            chunk_samples = int(chunk_seconds * sr)
            overlap_samples = int(0.2 * sr)
            fade_samples = overlap_samples
            
            chunks = []
            current_pos = 0
            total_len = low_quality_waveform.shape[1]
            step = chunk_samples - overlap_samples

            while current_pos + chunk_samples <= total_len:
                chunk = low_quality_waveform[:, current_pos : current_pos + chunk_samples]
                chunks.append(chunk)
                current_pos += step
            
            if current_pos < total_len:
                last_chunk = low_quality_waveform[:, -chunk_samples:]
                chunks.append(last_chunk)

            print(f"音频被分割成 {len(chunks)} 个块。")

            processed_chunks = []
            
            # --- 3. Inference over Chunks ---
            print("\n--- 步骤 3: 运行模型推理 ---")
            
            for i, chunk in enumerate(chunks):
                progress_callback(i + 1, len(chunks)) # Update GUI progress
                print(f"\n{'='*20} 正在处理块 {i+1}/{len(chunks)} {'='*20}")
                
                with torch.no_grad():
                    # Use AudioSR's exact mel processing pipeline
                    # mel_spectrogram_train expects (batch, samples) format
                    log_mel_spec, stft = self.mel_spectrogram_train(chunk.to(torch.float32))
                    
                    # AudioSR applies spectral normalization (dynamic range compression)
                    # This creates the proper log mel format that the VAE was trained on
                    log_mel = log_mel_spec.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, n_mels, time]
                    
                    print(f"   AudioSR mel: {log_mel.shape}, range: [{log_mel.min():.3f}, {log_mel.max():.3f}]")
                    
                    cond_tensor = self.model.cond_stage_models[0](log_mel)
                    cond = {"concat": cond_tensor}
                    
                    start_noise = torch.randn_like(cond_tensor)
                    latent = self._ddim_sample(start_noise, cond, steps=kwargs.get('ddim_steps', 100))
                    
                    # Use the integrated HiFi-GAN vocoder via decode_to_waveform
                    waveform_chunk = self.model.first_stage_model.decode_to_waveform(latent)
                    
                    # Move to CPU immediately to free GPU memory and convert to numpy
                    processed_chunks.append(waveform_chunk.squeeze().detach().cpu().numpy())
                    
                    # --- Clean up GPU memory ---
                    del log_mel_spec, stft, log_mel, cond_tensor, cond, start_noise, latent, waveform_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # --- 4. Stitching with Crossfade ---
            print("\n--- 步骤 4: 拼接音频块 ---")
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            final_audio = np.zeros(len(original_waveform) + chunk_samples)
            current_pos = 0
            
            for i, chunk in enumerate(processed_chunks):
                chunk_len = chunk.shape[0]
                if i == 0:
                    final_audio[current_pos : current_pos + chunk_len] += chunk
                else:
                    chunk[:fade_samples] *= fade_in
                    final_audio[current_pos : current_pos + fade_samples] *= fade_out
                    final_audio[current_pos : current_pos + chunk_len] += chunk
                current_pos += chunk_len - overlap_samples

            final_audio = final_audio[:len(original_waveform)]
            print("音频块拼接成功。")
            
            # --- 5. Advanced Post-processing ---
            print("\n--- 步骤 5: 后期处理 ---")
            if kwargs.get('use_clipping', False):
                final_audio = self._frequency_clipping(original_waveform.cpu().numpy(), final_audio, sr, 12000)
            if kwargs.get('use_filter', False):
                print("正在应用 12kHz 低通滤波器...")
                b, a = butter(8, 12000, btype='low', fs=sr)
                final_audio = lfilter(b, a, final_audio)
            
            max_abs_val = np.max(np.abs(final_audio))
            if max_abs_val > 0:
                print("正在标准化音频...")
                final_audio /= max_abs_val
            else:
                print("警告: 生成的音频为完全静音，跳过标准化。")

            
            # --- 6. Save Outputs ---
            print("\n--- 步骤 6: 保存输出 ---")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(output_dir, f"{basename}_enhanced.wav")
            sf.write(output_path, final_audio, sr)
            print(f"增强后的音频已保存至: {output_path}")
            spec_path = os.path.join(output_dir, f"{basename}_spectrogram.png")
            self._save_spectrogram(final_audio, sr, spec_path)
            print(f"频谱图已保存至: {spec_path}")

            return output_path, spec_path
        except Exception as e:
            raise e

    def _frequency_clipping(self, original_wav, generated_wav, sr, cutoff_freq):
        """Combines low frequencies from original audio with high frequencies from generated audio."""
        print(f"正在应用频率剪切，分界线为 {cutoff_freq} Hz...")
        n_fft = self.cfg['data']['n_fft']
        min_len = min(len(original_wav), len(generated_wav))
        original_wav, generated_wav = original_wav[:min_len], generated_wav[:min_len]

        stft_orig = torch.stft(torch.from_numpy(original_wav), n_fft=n_fft, return_complex=True)
        stft_gen = torch.stft(torch.from_numpy(generated_wav), n_fft=n_fft, return_complex=True)
        
        freq_bins = torch.fft.rfftfreq(n_fft, d=1.0/sr)
        cutoff_bin = torch.where(freq_bins >= cutoff_freq)[0][0]
        
        mask = torch.ones_like(stft_orig)
        mask[cutoff_bin:, :] = 0
        
        stft_combined = stft_orig * mask + stft_gen * (1 - mask)
        
        combined_wav = torch.istft(stft_combined, n_fft=n_fft)
        return combined_wav.cpu().numpy()

    def _save_spectrogram(self, waveform, sr, save_path):
        """Generates and saves a spectrogram plot."""
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.specgram(waveform, Fs=sr, cmap='viridis', NFFT=1024, noverlap=512)
        ax.set_title("Generated Audio Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

# =====================================================================================
# SECTION: TKINTER GUI APPLICATION
# =====================================================================================

class App(tk.Tk):
    def __init__(self, cfg_path):
        super().__init__()
        self.title("Audio Super-Resolution Inference GUI")
        self.geometry("800x700")
        
        self.cfg_path = cfg_path
        self.engine = None
        self.thread = None

        # --- Style ---
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), foreground='white', background='#0078D7')
        style.configure('TLabel', font=('Helvetica', 10))
        style.configure('TFrame', padding=10)

        # --- Variables ---
        self.input_audio_var = tk.StringVar(value="/home/husrcf/Code/Python/ProjectLily_Z_III/data/valid/low/02 - 共同渡过.wav")
        self.weights_path_var = tk.StringVar(value="/home/husrcf/Code/Python/ProjectLily_Z_III/outputs/audiosr_ldm_train/checkpoints/step_30000.pt")
        self.output_dir_var = tk.StringVar(value="/home/husrcf/Code/Python/ProjectLily_Z_III/output")
        self.use_filter_var = tk.BooleanVar(value=True)
        self.use_clipping_var = tk.BooleanVar(value=True)
        self.ddim_steps_var = tk.IntVar(value=100)

        # --- Main Frame ---
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._create_io_widgets(main_frame)
        self._create_options_widgets(main_frame)
        self._create_control_widgets(main_frame)
        self._create_output_widgets(main_frame)

    def _create_io_widgets(self, parent):
        io_frame = ttk.LabelFrame(parent, text="输入 / 输出", padding=15)
        io_frame.pack(fill=tk.X, pady=5)

        ttk.Label(io_frame, text="输入音频:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.input_audio_var, width=60).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(io_frame, text="浏览...", command=lambda: self._browse_file(self.input_audio_var, "选择音频文件", (("Audio Files", "*.wav *.flac *.mp3"), ("All files", "*.*")))).grid(row=0, column=2, padx=5)

        ttk.Label(io_frame, text="模型权重:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.weights_path_var, width=60).grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(io_frame, text="浏览...", command=lambda: self._browse_file(self.weights_path_var, "选择模型权重", (("PyTorch Checkpoints", "*.pt *.ckpt"), ("All files", "*.*")))).grid(row=1, column=2, padx=5)

        ttk.Label(io_frame, text="输出目录:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.output_dir_var, width=60).grid(row=2, column=1, sticky='ew', padx=5)
        ttk.Button(io_frame, text="浏览...", command=lambda: self._browse_dir(self.output_dir_var, "选择输出目录")).grid(row=2, column=2, padx=5)

        io_frame.columnconfigure(1, weight=1)

    def _create_options_widgets(self, parent):
        options_frame = ttk.LabelFrame(parent, text="推理选项", padding=15)
        options_frame.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(options_frame, text="对输出应用 12kHz 低通滤波器", variable=self.use_filter_var).pack(anchor='w', padx=5)
        ttk.Checkbutton(options_frame, text="使用频率剪切 (低频来自原始音频, 高频来自AI)", variable=self.use_clipping_var).pack(anchor='w', padx=5, pady=5)
        ddim_frame = ttk.Frame(options_frame)
        ddim_frame.pack(anchor='w', padx=5, pady=5)
        ttk.Label(ddim_frame, text="DDIM 步数:").pack(side=tk.LEFT)
        ttk.Spinbox(ddim_frame, from_=10, to=1000, textvariable=self.ddim_steps_var, width=8).pack(side=tk.LEFT, padx=5)

    def _create_control_widgets(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        self.start_button = ttk.Button(control_frame, text="开始推理", command=self.start_inference, style='Accent.TButton', padding=10)
        self.start_button.pack(pady=10)
        self.progress_bar = ttk.Progressbar(control_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True, pady=5)
        self.status_label = ttk.Label(control_frame, text="状态: 空闲", anchor='center')
        self.status_label.pack(fill=tk.X, expand=True)

    def _create_output_widgets(self, parent):
        output_frame = ttk.LabelFrame(parent, text="输出频谱图", padding=15)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#f0f0f0")
        self.ax.tick_params(axis='x', colors='gray'); self.ax.tick_params(axis='y', colors='gray')
        self.ax.spines['bottom'].set_color('gray'); self.ax.spines['top'].set_color('gray') 
        self.ax.spines['right'].set_color('gray'); self.ax.spines['left'].set_color('gray')
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=output_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _browse_file(self, var, title, filetypes):
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filepath: var.set(filepath)

    def _browse_dir(self, var, title):
        dirpath = filedialog.askdirectory(title=title)
        if dirpath: var.set(dirpath)

    def start_inference(self):
        if not self.input_audio_var.get() or not self.weights_path_var.get() or not self.output_dir_var.get():
            messagebox.showerror("错误", "请指定输入音频、模型权重和输出目录。")
            return
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        self.start_button.config(state=tk.DISABLED, text="处理中...")
        self.progress_bar['value'] = 0
        self.update_status("状态: 初始化...")
        self.thread = threading.Thread(target=self._inference_thread_func, daemon=True)
        self.thread.start()

    def _inference_thread_func(self):
        try:
            print("正在初始化引擎...")
            if self.engine is None:
                self.update_status("状态: 加载配置并实例化模型...")
                with open(self.cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                self.engine = InferenceEngine(cfg)
            
            self.update_status("状态: 加载模型权重...")
            self.engine.load_weights(self.weights_path_var.get())
            
            self.update_status("状态: 开始推理流程...")
            output_path, spec_path = self.engine.run_inference(
                audio_path=self.input_audio_var.get(), output_dir=self.output_dir_var.get(),
                progress_callback=self.update_progress, use_filter=self.use_filter_var.get(),
                use_clipping=self.use_clipping_var.get(), ddim_steps=self.ddim_steps_var.get()
            )
            self.after(0, self.on_inference_complete, output_path, spec_path)
        except BaseException:
            print("在推理线程中捕获到错误。正在向主线程报告...")
            exc_info = sys.exc_info()
            self.after(0, self.on_inference_error, exc_info)

    def update_progress(self, current, total):
        self.after(0, self._update_progress_gui, current, total)

    def _update_progress_gui(self, current, total):
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = current
        self.update_status(f"状态: 正在处理块 {current} / {total}...")

    def update_status(self, text):
        self.status_label.config(text=text)

    def on_inference_complete(self, output_path, spec_path):
        self.start_button.config(state=tk.NORMAL, text="开始推理")
        self.update_status(f"状态: 完成！音频已保存至 {output_path}")
        print(f"\n✅ 推理完成！音频已保存至: {output_path}")
        messagebox.showinfo("成功", f"推理完成！\n输出已保存至:\n{output_path}")
        
        img = plt.imread(spec_path)
        self.ax.clear()
        self.ax.imshow(img, aspect='auto', origin='lower')
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

    def on_inference_error(self, exc_info):
        self.start_button.config(state=tk.NORMAL, text="开始推理")
        self.update_status(f"状态: 发生错误！详情请查看控制台。")
        print("\n" + "="*50)
        print("推理过程中发生错误:")
        # Use the standard traceback module to print the exception
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        print("="*50 + "\n")

# =====================================================================================
# SECTION: MAIN EXECUTION
# =====================================================================================

if __name__ == '__main__':
    CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if not os.path.exists(CFG_PATH):
        tk.Tk().withdraw()
        messagebox.showerror("配置错误", f"错误: 在脚本目录中未找到 'config.yaml'。\n请确保该文件存在。")
        sys.exit(1)
    app = App(CFG_PATH)
    app.mainloop()
