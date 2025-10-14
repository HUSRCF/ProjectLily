# 项目结构

项目`/home/husrcf/Code/Python/ProjectLily_Z_III/`的目录结构（已排除如`node_modules`等目录）：
```
/home/husrcf/Code/Python/ProjectLily_Z_III//
├── app.py
├── config.yaml
├── dataset_loader.py
├── infer.py
├── model.py
├── package.py
├── preprocess_audio.py
├── run_gui.py
└── train.py
```

---

# 代码内容

## 文件: `preprocess_audio.py`

```python

import os
import argparse
from typing import Optional, List
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

def load_audio_mono(path: str):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr

def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    x = torch.tensor(audio, dtype=torch.float32)[None, None, :]
    ratio = sr_out / sr_in
    new_len = int(round(x.shape[-1] * ratio))
    y = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
    return y[0,0].numpy()

def windowed_sinc_lowpass(cutoff_hz: float, sr: int, kernel_size: int = 127, eps: float = 1e-8):
    """
    Simple windowed-sinc low-pass filter kernel for 1D conv.
    cutoff_hz: pass-band edge
    """
    # normalized cutoff in [0,1], 1 -> Nyquist (sr/2)
    wc = cutoff_hz / (sr / 2 + eps)
    n = torch.arange(kernel_size) - (kernel_size - 1) / 2
    # sinc
    h = torch.where(n == 0, torch.tensor(1.0), torch.sin(torch.tensor(np.pi) * wc * n) / (torch.tensor(np.pi) * wc * n))
    # Hann window
    w = 0.5 * (1 - torch.cos(2 * torch.tensor(np.pi) * (torch.arange(kernel_size) / (kernel_size - 1))))
    h = h * w
    # normalize
    h = h / h.sum()
    return h.float()

def apply_lowpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    kernel = windowed_sinc_lowpass(cutoff_hz, sr, kernel_size=127)
    x = torch.tensor(audio, dtype=torch.float32)[None, None, :]
    k = kernel[None, None, :]
    y = F.conv1d(x, k, padding=kernel.shape[0]//2)
    return y[0,0].numpy()

def make_low_res(audio: np.ndarray, sr: int, target_sr_low: int = 12000, cutoff_hz: Optional[float] = None) -> np.ndarray:
    """
    Create a low-resolution version by (optional) LPF -> downsample -> upsample back to sr.
    """
    if cutoff_hz is None:
        cutoff_hz = min(target_sr_low // 2 - 1000, 6000)  # leave guard band
        cutoff_hz = max(2000, cutoff_hz)
    y = apply_lowpass(audio, sr=sr, cutoff_hz=cutoff_hz)
    y = resample(y, sr_in=sr, sr_out=target_sr_low)
    y = resample(y, sr_in=target_sr_low, sr_out=sr)  # bring back to original sr
    return y

def process_file(path_in: str, out_high: str, out_low: str, target_sr: int = 48000, target_sr_low: int = 12000, mp3_lpf_hz: Optional[float] = 16000):
    audio, sr = load_audio_mono(path_in)
    # resample to target_sr
    audio = resample(audio, sr_in=sr, sr_out=target_sr)

    # optional low-pass to attenuate MP3 artifacts (typically above ~16k for many bitrates)
    if mp3_lpf_hz is not None:
        audio = apply_lowpass(audio, sr=target_sr, cutoff_hz=mp3_lpf_hz)

    # synthesize low-res degraded input
    low = make_low_res(audio, sr=target_sr, target_sr_low=target_sr_low)

    os.makedirs(os.path.dirname(out_high), exist_ok=True)
    os.makedirs(os.path.dirname(out_low), exist_ok=True)
    sf.write(out_high, audio, target_sr)
    sf.write(out_low,  low,  target_sr)
    return out_high, out_low

def preprocess_dir(in_dir: str, out_high: str, out_low: str, sr: int = 48000, low_sr: int = 12000, mp3_lpf_hz: Optional[float] = 16000) -> List[str]:
    """Batch preprocess a directory. Returns list of processed basenames."""
    exts = ('.wav', '.flac', '.ogg', '.mp3')
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(exts)]
    processed = []
    for f in files:
        in_path = os.path.join(in_dir, f)
        high_path = os.path.join(out_high, os.path.splitext(f)[0] + '.wav')
        low_path  = os.path.join(out_low,  os.path.splitext(f)[0] + '.wav')
        process_file(in_path, high_path, low_path, target_sr=sr, target_sr_low=low_sr, mp3_lpf_hz=mp3_lpf_hz)
        processed.append(f)
    return processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory containing wav/flac/ogg/mp3')
    parser.add_argument('--out_high', type=str, required=True, help='Output directory for 48k high-quality wav')
    parser.add_argument('--out_low',  type=str, required=True, help='Output directory for degraded low-res wav')
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--low_sr', type=int, default=12000, help='Intermediate bottleneck sample rate to simulate bandwidth loss')
    parser.add_argument('--mp3_lpf_hz', type=float, default=16000.0, help='Apply pre LPF at this cutoff to reduce MP3 artifacts; set -1 to disable')
    args = parser.parse_args()

    mp3_lpf = None if args.mp3_lpf_hz < 0 else float(args.mp3_lpf_hz)
    processed = preprocess_dir(args.in_dir, args.out_high, args.out_low, sr=args.sr, low_sr=args.low_sr, mp3_lpf_hz=mp3_lpf)
    for f in processed:
        print('Processed:', f)

if __name__ == '__main__':
    main()

```

## 文件: `app.py`

```python

import os
import sys
import yaml
import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.append(BASE)

import preprocess_audio as prep
import train as train_mod
import infer as infer_mod

CFG_PATH = os.path.join(BASE, 'config.yaml')

def ensure_dirs(cfg):
    os.makedirs(cfg['data']['train_dir_high'], exist_ok=True)
    os.makedirs(cfg['data']['train_dir_low'],  exist_ok=True)
    os.makedirs(cfg['data']['valid_dir_high'], exist_ok=True)
    os.makedirs(cfg['data']['valid_dir_low'],  exist_ok=True)
    os.makedirs(cfg['experiment']['out_dir'],   exist_ok=True)

def gui_preprocess(in_dir, out_high, out_low, sr, low_sr, mp3_lpf):
    try:
        mp3_lpf = None if mp3_lpf is None or mp3_lpf < 0 else float(mp3_lpf)
        processed = prep.preprocess_dir(in_dir, out_high, out_low, sr=int(sr), low_sr=int(low_sr), mp3_lpf_hz=mp3_lpf)
        log = '\n'.join([f'Processed: {x}' for x in processed]) or 'No files found.'
        return log
    except Exception as e:
        return f'[ERROR] {repr(e)}'

def gui_train():
    try:
        ckpt = train_mod.run_training(CFG_PATH)
        return f'[DONE] Final EMA checkpoint: {ckpt}'
    except Exception as e:
        return f'[ERROR] {repr(e)}'

def gui_infer(low_path, ckpt_path):
    try:
        out_path = os.path.join(BASE, 'sr_output.wav')
        out = infer_mod.run_inference(CFG_PATH, ckpt_path, low_path, out_path)
        return out, f'Saved: {out}'
    except Exception as e:
        return None, f'[ERROR] {repr(e)}'

def app():
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    ensure_dirs(cfg)

    with gr.Blocks(title='AudioSR Mini Pipeline', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# AudioSR Mini Pipeline (GUI)')
        with gr.Tab('预处理 / Preprocess'):
            in_dir   = gr.Textbox(value='raw_audio', label='输入目录（包含 wav/flac/ogg/mp3）')
            out_high = gr.Textbox(value=cfg['data']['train_dir_high'], label='输出高质量目录 (48kHz)')
            out_low  = gr.Textbox(value=cfg['data']['train_dir_low'],  label='输出低质量目录 (与高质量同名)')
            with gr.Row():
                sr      = gr.Number(value=cfg['data']['sample_rate'], label='目标采样率', precision=0)
                low_sr  = gr.Number(value=12000, label='低带宽采样率（中转）', precision=0)
                mp3_lpf = gr.Number(value=16000, label='MP3 预低通截止频率（-1 关闭）', precision=0)
            btn_p = gr.Button('开始预处理')
            log_p = gr.Textbox(label='日志', lines=12)
            btn_p.click(gui_preprocess, inputs=[in_dir, out_high, out_low, sr, low_sr, mp3_lpf], outputs=log_p)

        with gr.Tab('训练 / Train'):
            gr.Markdown('使用 config.yaml 中的配置直接训练（v-pred + EMA + STFT Loss）。')
            btn_t = gr.Button('开始训练')
            log_t = gr.Textbox(label='训练日志（只显示最后结果；详细日志见终端）', lines=6)
            btn_t.click(gui_train, outputs=log_t)

        with gr.Tab('推理 / Inference (已集成 OLA)'):
            gr.Markdown('选择低质量音频与权重，自动使用配置中的采样步数、CFG、OLA 分块尺寸。')
            low_file  = gr.Audio(source='upload', type='filepath', label='低质量输入音频')
            ckpt_file = gr.Textbox(value=os.path.join(cfg['experiment']['out_dir'], 'checkpoints', 'final_ema.pt'), label='Checkpoint 路径（EMA *.pt）')
            btn_i = gr.Button('开始推理')
            out_audio = gr.Audio(label='还原结果', type='filepath')
            out_log   = gr.Textbox(label='日志', lines=4)
            btn_i.click(gui_infer, inputs=[low_file, ckpt_file], outputs=[out_audio, out_log])

        gr.Markdown('—— 启动：`python app.py`（无需命令行参数） ——')

    demo.launch(server_name='0.0.0.0', server_port=7860, show_api=False)

if __name__ == '__main__':
    app()

```

## 文件: `run_gui.py`

```python
from app import app
if __name__ == '__main__':
    app()

```

## 文件: `dataset_loader.py`

```python
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")

def load_audio_mono(path: str):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)

def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    x = torch.tensor(audio, dtype=torch.float32)[None, None, :]
    new_len = int(round(x.shape[-1] * (sr_out / sr_in)))
    y = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return y[0, 0].numpy()

class MultiDomainPairedDataset(Dataset):
    """
    扫描 dataset_root/{category}/{high_dir_name, low_dir_name} 构造 (low, high) 配对。
    通过 valid_ratio + split_seed 做确定性 train/valid 划分。

    读取策略：
    - low(12k) 与 high(48k) 均重采样到 sample_rate（默认 48k）
    - 两端长度不齐则右侧补零对齐
    - 训练/验证：基于起点裁取 segment_seconds（建议 6s），并带空白检测与后移重试逻辑
    返回 (cond, x0)，其中 cond 为 低带宽->重采样回 sample_rate 的条件，x0 为目标高质量。
    """

    def __init__(
        self,
        dataset_root: str,
        categories: List[str],
        high_dir_name: str,
        low_dir_name: str,
        sample_rate: int,
        segment_seconds: float,
        split: str = "train",
        valid_ratio: float = 0.05,
        split_seed: int = 1337,
        rms_target: Optional[float] = None,
        blank_thr: float = 1e-4,              # 振幅阈值（|x| < thr 视为空白）
        blank_ratio_max: float = 0.30,        # 允许的空白占比上限
        blank_hop_seconds: float = 1.0,       # 失败后起点右移秒数
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.segment_len = int(round(segment_seconds * sample_rate))
        self.rms_target = rms_target
        self.is_train = (split == "train")
        self.blank_thr = float(blank_thr)
        self.blank_ratio_max = float(blank_ratio_max)
        self.blank_hop = max(1, int(round(blank_hop_seconds * self.sample_rate)))

        pairs: List[Tuple[str, str]] = []  # (low_path, high_path)
        for cat in categories:
            high_dir = os.path.join(dataset_root, cat, high_dir_name)
            low_dir  = os.path.join(dataset_root, cat, low_dir_name)
            if not (os.path.isdir(high_dir) and os.path.isdir(low_dir)):
                continue
            high_names = {f for f in os.listdir(high_dir) if f.lower().endswith(AUDIO_EXTS)}
            low_names  = {f for f in os.listdir(low_dir)  if f.lower().endswith(AUDIO_EXTS)}
            inter = sorted(list(high_names & low_names))
            for name in inter:
                pairs.append((os.path.join(low_dir, name), os.path.join(high_dir, name)))

        rng = random.Random(split_seed)
        rng.shuffle(pairs)
        n_total = len(pairs)
        n_valid = int(round(n_total * float(valid_ratio)))
        valid_set = pairs[:n_valid]
        train_set = pairs[n_valid:]
        self.pairs = train_set if self.is_train else valid_set
        if len(self.pairs) == 0:
            raise RuntimeError("No paired audio found under dataset_root. Check folder names and files.")

    def __len__(self):
        return len(self.pairs)

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.rms_target is None:
            return x
        rms = torch.sqrt(torch.mean(x**2, dim=(-1,)))[:, None]
        scale = torch.clamp(self.rms_target / (rms + 1e-8), 0.1, 10.0)
        return x * scale

    @staticmethod
    def _blank_ratio(arrs: List[np.ndarray], thr: float) -> float:
        # 同时满足所有通道/来源都低于阈值才算空白
        mask = np.ones_like(arrs[0], dtype=bool)
        for a in arrs:
            mask &= (np.abs(a) < thr)
        return float(mask.mean())

    def _pick_start_with_blank_skip(self, high: np.ndarray, low: np.ndarray, seg: int, prefer_center: bool) -> int:
        n = len(high)
        if n <= seg:
            return 0
        # 初始起点：训练随机 / 验证中心
        if prefer_center:
            start = max(0, (n - seg) // 2)
        else:
            start = random.randint(0, n - seg)

        best_start = start
        best_ratio = 1.0
        tried = set()
        while True:
            if start in tried:
                break
            tried.add(start)
            h = high[start:start+seg]
            l = low[start:start+seg]
            r = self._blank_ratio([h, l], self.blank_thr)
            if r < best_ratio:
                best_ratio = r
                best_start = start
            if r < self.blank_ratio_max:
                return start
            # 向后移动
            if start + self.blank_hop <= n - seg:
                start = start + self.blank_hop
            else:
                # 无法再移动，返回最优候选
                return best_start

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]
        low,  sr_l = load_audio_mono(low_path)
        high, sr_h = load_audio_mono(high_path)

        # 重采样到统一 SR（如 48k）
        low  = resample(low,  sr_l, self.sample_rate)
        high = resample(high, sr_h, self.sample_rate)

        # 右侧补零对齐（处理压缩/解码边缘差异）
        Lh, Ll = len(high), len(low)
        if Lh != Ll:
            if Lh > Ll:
                low = np.pad(low, (0, Lh - Ll))
            else:
                high = np.pad(high, (0, Ll - Lh))

        seg = self.segment_len
        if len(high) < seg:
            pad = seg - len(high)
            high = np.pad(high, (0, pad))
            low  = np.pad(low,  (0, pad))

        # 选择起点 + 空白检测（训练随机 / 验证中心，都启用跳空逻辑）
        prefer_center = not self.is_train
        start = self._pick_start_with_blank_skip(high, low, seg, prefer_center)

        high = high[start:start + seg]
        low  = low[start:start + seg]

        x = torch.tensor(high, dtype=torch.float32)[None, :]
        c = torch.tensor(low,  dtype=torch.float32)[None, :]
        x = self._rms_normalize(x)
        c = self._rms_normalize(c)
        return c, x
```

## 文件: `config.yaml`

```yaml
experiment:
  name: audiosr-mini
  seed: 1337
  out_dir: outputs/audiosr-mini

data:
  train_dir_high: data/train/high
  train_dir_low:  data/train/low
  valid_dir_high: data/valid/high
  valid_dir_low:  data/valid/low
  sample_rate: 48000
  mono: true
  segment_seconds: 3.0
  rms_target: 0.06

model:
  in_channels: 1
  cond_channels: 1
  base_channels: 128
  channel_mults: [1, 2, 4, 4]
  num_res_blocks: 2
  dropout: 0.0
  use_attention_scales: [2]
  attn_num_heads: 8
  attn_head_dim: 32
  time_embed_dim: 512
  prediction_type: v

diffusion:
  steps: 1000
  schedule: cosine
  cfg_drop_prob: 0.15
  aux_recon_weight: 0.1

train:
  batch_size: 8
  num_workers: 4
  epochs: 50
  lr: 2.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
  ema_decay: 0.999
  grad_clip: 1.0
  warmup_steps: 2000
  max_steps: null
  log_interval: 100
  save_interval_steps: 2000
  valid_interval_steps: 2000

loss:
  stft_scales:
    - [2048, 512, 2048]
    - [1024, 256, 1024]
    - [512, 128, 512]
  l1_weight: 1.0
  stft_weight: 0.5

inference:
  sampler: ddim
  sample_steps: 24
  guidance_scale: 1.5
  dynamic_clip_percentile: 0.999
  chunk_seconds: 5.0
  chunk_hop_seconds: 4.0
```

## 文件: `model.py`

```python
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x): return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        device = t.device; half_dim = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half_dim, device=device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0,1)); return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__(); self.pos = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
    def forward(self, t): return self.mlp(self.pos(t))

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout=0.0, up=False, down=False):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch; self.up, self.down = up, down
        self.norm1 = nn.GroupNorm(8, in_ch); self.act1 = nn.SiLU(); self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(tdim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch); self.act2 = nn.SiLU(); self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch or up or down: self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else: self.skip = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) if up else None
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2) if down else None
    def forward(self, x, t_emb):
        h = x
        if exists(self.downsample): x = self.downsample(x); h = x
        x = self.norm1(x); x = self.act1(x); x = self.conv1(x)
        x = x + self.time_proj(t_emb)[:, :, None]
        x = self.norm2(x); x = self.act2(x); x = self.dropout(x); x = self.conv2(x)
        if exists(self.upsample): x = self.upsample(x); h = self.upsample(h)
        return x + self.skip(h)

class SpatialTransformer1D(nn.Module):
    def __init__(self, channels, num_heads=8, head_dim=32):
        super().__init__(); self.channels = channels; self.inner_dim = num_heads * head_dim
        self.norm = nn.GroupNorm(8, channels); self.proj_in = nn.Conv1d(channels, self.inner_dim, 1)
        self.attn = nn.MultiheadAttention(embed_dim=self.inner_dim, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv1d(self.inner_dim, channels, 1)
    def forward(self, x, context=None):
        b, c, t = x.shape; h = self.norm(x); h = self.proj_in(h); h = h.transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False); h = h.transpose(1, 2); h = self.proj_out(h); return x + h

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, base_channels=128, channel_mults=(1,2,4,4),
                 num_res_blocks=2, time_embed_dim=512, dropout=0.0, use_attention_scales=None,
                 attn_num_heads=8, attn_head_dim=32, prediction_type='v'):
        super().__init__(); self.prediction_type = prediction_type
        self.in_channels = in_channels; self.cond_channels = cond_channels
        self.input_conv = nn.Conv1d(in_channels + cond_channels, base_channels, 3, padding=1)
        self.time_mlp = TimeEmbedding(base_channels * 4, time_embed_dim)
        ch = base_channels; self.downs = nn.ModuleList(); use_attention_scales = set(use_attention_scales or [])
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock1D(ch, out_ch, time_embed_dim, dropout=dropout, down=False)); ch = out_ch
                if i in use_attention_scales: self.downs.append(SpatialTransformer1D(ch, attn_num_heads, attn_head_dim))
            if i < len(channel_mults) - 1: self.downs.append(ResBlock1D(ch, ch, time_embed_dim, down=True))
        self.mid1 = ResBlock1D(ch, ch, time_embed_dim, dropout=dropout); self.mid_attn = SpatialTransformer1D(ch, attn_num_heads, attn_head_dim); self.mid2 = ResBlock1D(ch, ch, time_embed_dim, dropout=dropout)
        self.ups = nn.ModuleList()
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock1D(ch + out_ch, out_ch, time_embed_dim, dropout=dropout, up=False)); ch = out_ch
                if i in use_attention_scales: self.ups.append(SpatialTransformer1D(ch, attn_num_heads, attn_head_dim))
            if i > 0: self.ups.append(ResBlock1D(ch, ch, time_embed_dim, up=True))
        self.out_norm = nn.GroupNorm(8, ch); self.out_act = nn.SiLU(); self.out_conv = nn.Conv1d(ch, in_channels, 3, padding=1)
    def forward(self, x, t, cond=None, cond_drop_mask=None):
        b, c, tlen = x.shape
        if cond is None:
            cond = torch.zeros(b, self.cond_channels, tlen, device=x.device, dtype=x.dtype)
        else:
            if cond.shape[-1] != tlen:
                cond = F.interpolate(cond, size=tlen, mode='linear', align_corners=False)
        if cond_drop_mask is not None:
            mask = cond_drop_mask[:, None, None].to(cond.dtype); cond = cond * (1.0 - mask)
        x = torch.cat([x, cond], dim=1); x = self.input_conv(x)
        t_emb = self.time_mlp(t)
        skips = []; i = 0
        while i < len(self.downs):
            layer = self.downs[i]
            if isinstance(layer, ResBlock1D):
                x = layer(x, t_emb); skips.append(x); i += 1
            elif isinstance(layer, SpatialTransformer1D):
                x = layer(x); i += 1
            else:
                x = layer(x, t_emb); i += 1
        x = self.mid1(x, t_emb); x = self.mid_attn(x); x = self.mid2(x, t_emb)
        j = len(skips) - 1; i = 0
        while i < len(self.ups):
            layer = self.ups[i]
            if isinstance(layer, ResBlock1D):
                if layer.upsample is None and layer.downsample is None:
                    x = torch.cat([x, skips[j]], dim=1); j -= 1
                x = layer(x, t_emb); i += 1
            elif isinstance(layer, SpatialTransformer1D):
                x = layer(x); i += 1
            else:
                x = layer(x, t_emb); i += 1
        x = self.out_norm(x); x = self.out_act(x); x = self.out_conv(x); return x
```

## 文件: `train.py`

```python

import os
import math
import random
import argparse
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Local imports ----
from model import UNet1D
try:
    from dataset_loader import MultiDomainPairedDataset
except Exception:
    MultiDomainPairedDataset = None

# ========== Utils ==========

def set_seed(seed: int):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rms_normalize(x: torch.Tensor, target_rms: Optional[float] = None):
    if target_rms is None:
        return x
    rms = torch.sqrt(torch.mean(x**2, dim=(-1,)))
    rms = rms[:, None]
    scale = torch.clamp(target_rms / (rms + 1e-8), 0.1, 10.0)
    return x * scale


# ========== Legacy fallback dataset (high/low folders) ==========

import soundfile as sf
import numpy as np

def _load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        x = torch.tensor(audio, dtype=torch.float32)[None, None, :]
        ratio = target_sr / sr
        new_len = int(round(x.shape[-1] * ratio))
        x = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
        audio = x[0,0].numpy()
    return audio.astype(np.float32)

class PairWaveDataset(Dataset):
    """
    Fallback dataset: two folders (high_dir, low_dir) with matching filenames.
    Includes blank-segment skipping as requested.
    """
    def __init__(
        self,
        high_dir: str,
        low_dir: str,
        sample_rate: int,
        segment_seconds: float,
        rms_target: Optional[float] = None,
        split: str = 'train',
        blank_thr: float = 1e-4,
        blank_ratio_max: float = 0.30,
        blank_hop_seconds: float = 1.0
    ):
        super().__init__()
        self.high_dir = high_dir
        self.low_dir = low_dir
        self.sample_rate = int(sample_rate)
        self.segment_len = int(round(segment_seconds * sample_rate))
        self.rms_target = rms_target
        self.split = split
        self.blank_thr = float(blank_thr)
        self.blank_ratio_max = float(blank_ratio_max)
        self.blank_hop = max(1, int(round(blank_hop_seconds * self.sample_rate)))
        high_names = set([f for f in os.listdir(high_dir) if f.lower().endswith(('.wav', '.flac', '.ogg'))])
        low_names  = set([f for f in os.listdir(low_dir) if f.lower().endswith(('.wav', '.flac', '.ogg'))])
        self.names = sorted(list(high_names & low_names))
        if len(self.names) == 0:
            raise RuntimeError('No matching audio filenames found in high/ and low/.')

    @staticmethod
    def _blank_ratio(h: np.ndarray, l: np.ndarray, thr: float) -> float:
        m = (np.abs(h) < thr) & (np.abs(l) < thr)
        return float(m.mean())

    def _pick_start(self, high: np.ndarray, low: np.ndarray, seg: int, prefer_center: bool) -> int:
        n = len(high)
        if n <= seg:
            return 0
        start = max(0, (n - seg) // 2) if prefer_center else random.randint(0, n - seg)
        best_start, best_ratio = start, 1.0
        tried = set()
        while True:
            if start in tried:
                break
            tried.add(start)
            h = high[start:start+seg]; l = low[start:start+seg]
            r = self._blank_ratio(h, l, self.blank_thr)
            if r < best_ratio:
                best_ratio, best_start = r, start
            if r < self.blank_ratio_max:
                return start
            if start + self.blank_hop <= n - seg:
                start += self.blank_hop
            else:
                return best_start

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        high = _load_audio_mono(os.path.join(self.high_dir, name), self.sample_rate)
        low  = _load_audio_mono(os.path.join(self.low_dir, name),  self.sample_rate)

        H = len(high); L = len(low); seg = self.segment_len
        if H != L:
            if H > L:
                low = np.pad(low, (0, H - L))
            else:
                high = np.pad(high, (0, L - H))
            H = L = len(high)
        if H < seg:
            pad = seg - H
            high = np.pad(high, (0, pad))
            low  = np.pad(low,  (0, pad))
            H = L = len(high)

        prefer_center = (self.split != 'train')
        start = self._pick_start(high, low, seg, prefer_center)
        high = high[start:start+seg]; low  = low[start:start+seg]

        x = torch.tensor(high, dtype=torch.float32)[None, :]
        c = torch.tensor(low,  dtype=torch.float32)[None, :]
        x = rms_normalize(x, self.rms_target); c = rms_normalize(c, self.rms_target)
        return c, x


# ========== Diffusion utilities ==========

def cosine_alpha_cumprod(n_steps: int):
    s = 0.008
    t = torch.linspace(0, n_steps, n_steps+1)
    f = torch.cos(((t / n_steps) + s) / (1+s) * math.pi / 2) ** 2
    a_bar = f / f[0]
    return a_bar[1:]

class NoiseSchedule:
    def __init__(self, steps: int, schedule: str = 'cosine', device='cpu'):
        self.steps = steps
        if schedule == 'cosine':
            a_bar = cosine_alpha_cumprod(steps).to(device)
        else:
            betas = torch.linspace(1e-4, 0.02, steps, device=device)
            a_bar = torch.cumprod(1.0 - betas, dim=0)
        self.alpha_bars = a_bar
        self.alphas = torch.sqrt(a_bar)
        self.sigmas = torch.sqrt(1.0 - a_bar)

    def sample_timesteps(self, b: int, device):
        t_idx = torch.randint(0, self.steps, (b,), device=device)
        a = self.alphas[t_idx]; s = self.sigmas[t_idx]
        return t_idx, a, s

    def alpha_sigma_at(self, t_idx: torch.Tensor):
        a = self.alphas[t_idx]; s = self.sigmas[t_idx]
        return a, s


# ========== Losses ==========

def stft_loss(x, y, stft_scales: List[Tuple[int,int,int]]):
    loss_sc = 0.0; loss_mag = 0.0
    for n_fft, hop, win in stft_scales:
        x_spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, return_complex=True, center=True)
        y_spec = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, return_complex=True, center=True)
        x_mag = torch.clamp(x_spec.abs(), 1e-7, None); y_mag = torch.clamp(y_spec.abs(), 1e-7, None)
        sc = torch.mean(torch.norm(y_mag - x_mag, dim=-1) / torch.norm(y_mag, dim=-1).clamp(min=1e-7))
        mag = torch.mean(torch.abs(y_mag - x_mag))
        loss_sc = loss_sc + sc; loss_mag = loss_mag + mag
    loss_sc = loss_sc / len(stft_scales); loss_mag = loss_mag / len(stft_scales)
    return loss_sc + loss_mag


# ========== EMA ==========

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay; self.shadow = {}; self.register(model)
    def register(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad: self.shadow[name] = p.data.clone()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[name])


# ========== Warmup + Cosine LR Scheduler ==========

class WarmupCosine:
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.ws = max(1, int(warmup_steps))
        self.ms = max(1, int(max_steps))
        self.min_ratio = float(min_lr_ratio)
        self.step_id = 0
    def get_factor(self, step: int) -> float:
        if step < self.ws:
            return (step + 1) / self.ws
        t = (step - self.ws) / max(1, self.ms - self.ws)
        return self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * t))
    def step(self):
        f = self.get_factor(self.step_id)
        for g in self.opt.param_groups:
            g['lr'] = self.base_lr * f
        self.step_id += 1
        return f


# ========== Training core ==========

@dataclass
class TrainConfig:
    cfg: dict

def forward_diffusion_sample(x0, a, s, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    while a.ndim < x0.ndim:
        a = a[:, None]; s = s[:, None]
    xt = a * x0 + s * noise; return xt, noise

def v_from(x0, eps, a, s): return a * eps - s * x0
def x0_from_v(xt, v, a, s): return a * xt - s * v

def train_one_step(model, batch, nsched, optimizer, device, cfg_drop_prob, loss_conf, pred_type):
    model.train()
    cond, x0 = batch; cond = cond.to(device); x0 = x0.to(device)
    b = x0.shape[0]; t_idx, a, s = nsched.sample_timesteps(b, device=device)
    t_norm = (t_idx.float() + 0.5) / nsched.steps
    xt, eps = forward_diffusion_sample(x0, a, s)
    drop = (torch.rand(b, device=device) < cfg_drop_prob)
    target = v_from(x0, eps, a, s) if pred_type == 'v' else eps
    pred = model(xt, t_norm, cond=cond, cond_drop_mask=drop)
    loss_main = F.smooth_l1_loss(pred, target)
    x0_hat = x0_from_v(xt, pred, a[:, None], s[:, None]) if pred_type == 'v' else (xt - s[:, None] * pred) / a[:, None]
    l1 = torch.mean(torch.abs(x0_hat - x0))
    stft = stft_loss(x0_hat, x0, loss_conf['stft_scales'])
    loss = loss_main + loss_conf['stft_weight'] * stft + loss_conf['l1_weight'] * l1 * loss_conf.get('aux_recon_weight', 0.1)
    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
    return loss.item(), {'l_main': loss_main.item(), 'l_l1': l1.item(), 'l_stft': stft.item()}

@torch.no_grad()
def validate(model, loader, nsched, device, loss_conf, pred_type):
    model.eval(); losses = []; import numpy as np
    for batch in loader:
        cond, x0 = batch; cond = cond.to(device); x0 = x0.to(device)
        b = x0.shape[0]; t_idx = torch.full((b,), nsched.steps - 1, device=device, dtype=torch.long)
        a, s = nsched.alpha_sigma_at(t_idx); t_norm = (t_idx.float() + 0.5) / nsched.steps
        xt, eps = forward_diffusion_sample(x0, a, s)
        target = v_from(x0, eps, a, s) if pred_type == 'v' else eps
        pred = model(xt, t_norm, cond=cond, cond_drop_mask=None)
        loss = F.smooth_l1_loss(pred, target); losses.append(loss.item())
    return float(np.mean(losses))

def run_training(config_path: str) -> str:
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)

    # enforce default 6.0s if missing
    if 'data' in cfg and 'segment_seconds' not in cfg['data']:
        cfg['data']['segment_seconds'] = 6.0

    set_seed(cfg['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(cfg['experiment']['out_dir']); ensure_dir(os.path.join(cfg['experiment']['out_dir'], 'checkpoints')); ensure_dir(os.path.join(cfg['experiment']['out_dir'], 'logs'))

    # === dataset selection & blank-skip params ===
    data_conf = cfg['data']
    blank_thr = float(data_conf.get('blank_thr', 1e-4))
    blank_ratio_max = float(data_conf.get('blank_ratio_max', 0.30))
    blank_hop_seconds = float(data_conf.get('blank_hop_seconds', 1.0))

    if data_conf.get('dataset_root') and MultiDomainPairedDataset is not None:
        common_kwargs = dict(
            dataset_root=data_conf['dataset_root'],
            categories=list(data_conf.get('categories', [])),
            high_dir_name=data_conf.get('high_dir_name', '48k'),
            low_dir_name=data_conf.get('low_dir_name', '12k'),
            sample_rate=data_conf['sample_rate'],
            segment_seconds=data_conf['segment_seconds'],
            valid_ratio=float(data_conf.get('valid_ratio', 0.05)),
            split_seed=int(data_conf.get('split_seed', 1337)),
            rms_target=data_conf.get('rms_target'),
            blank_thr=blank_thr,
            blank_ratio_max=blank_ratio_max,
            blank_hop_seconds=blank_hop_seconds,
        )
        train_set = MultiDomainPairedDataset(split='train', **common_kwargs)
        valid_set = MultiDomainPairedDataset(split='valid', **common_kwargs)
    else:
        train_set = PairWaveDataset(
            high_dir=data_conf['train_dir_high'],
            low_dir=data_conf['train_dir_low'],
            sample_rate=data_conf['sample_rate'],
            segment_seconds=data_conf['segment_seconds'],
            rms_target=data_conf.get('rms_target'),
            split='train',
            blank_thr=blank_thr,
            blank_ratio_max=blank_ratio_max,
            blank_hop_seconds=blank_hop_seconds,
        )
        valid_set = PairWaveDataset(
            high_dir=data_conf['valid_dir_high'],
            low_dir=data_conf['valid_dir_low'],
            sample_rate=data_conf['sample_rate'],
            segment_seconds=data_conf['segment_seconds'],
            rms_target=data_conf.get('rms_target'),
            split='valid',
            blank_thr=blank_thr,
            blank_ratio_max=blank_ratio_max,
            blank_hop_seconds=blank_hop_seconds,
        )

    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=max(1, cfg['train']['batch_size']//2), shuffle=False, num_workers=cfg['train']['num_workers'], drop_last=False)

    # === model ===
    mcfg = cfg['model']
    model = UNet1D(
        in_channels=mcfg['in_channels'],
        cond_channels=mcfg['cond_channels'],
        base_channels=mcfg['base_channels'],
        channel_mults=tuple(mcfg['channel_mults']),
        num_res_blocks=mcfg['num_res_blocks'],
        time_embed_dim=mcfg['time_embed_dim'],
        dropout=mcfg['dropout'],
        use_attention_scales=mcfg['use_attention_scales'],
        attn_num_heads=mcfg['attn_num_heads'],
        attn_head_dim=mcfg['attn_head_dim'],
        prediction_type=mcfg.get('prediction_type','v')
    ).to(device)

    # === optimizer + scheduler ===
    opt_conf = cfg['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf['lr'], weight_decay=opt_conf['weight_decay'], betas=tuple(opt_conf['betas']))

    steps_per_epoch = max(1, len(train_loader))
    max_steps = opt_conf.get('max_steps') or (opt_conf['epochs'] * steps_per_epoch)
    scheduler = WarmupCosine(
        optimizer,
        base_lr=opt_conf['lr'],
        warmup_steps=int(opt_conf.get('warmup_steps', 2000)),
        max_steps=int(max_steps),
        min_lr_ratio=float(opt_conf.get('min_lr_ratio', 0.1))
    )

    # === EMA & noise schedule ===
    ema = EMA(model, decay=opt_conf['ema_decay'])
    nsched = NoiseSchedule(steps=cfg['diffusion']['steps'], schedule=cfg['diffusion']['schedule'], device=device)

    # === training loop ===
    global_step = 0; best_val = float('inf')
    for epoch in range(cfg['train']['epochs']):
        for batch in train_loader:
            global_step += 1
            lr_factor = scheduler.step()  # update LR each step

            loss_value, d = train_one_step(
                model, batch, nsched, optimizer, device,
                cfg_drop_prob=cfg['diffusion']['cfg_drop_prob'],
                loss_conf={
                    'stft_scales': [tuple(x) for x in cfg['loss']['stft_scales']],
                    'stft_weight': cfg['loss']['stft_weight'],
                    'l1_weight': cfg['loss']['l1_weight'],
                    'aux_recon_weight': cfg['diffusion'].get('aux_recon_weight', 0.1)
                },
                pred_type=cfg['model'].get('prediction_type','v')
            )
            ema.update(model)

            if global_step % cfg['train']['log_interval'] == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"[step {global_step}] lr={cur_lr:.2e} loss={loss_value:.4f} l_main={d['l_main']:.4f} l1={d['l_l1']:.4f} stft={d['l_stft']:.4f}")

            if global_step % cfg['train']['valid_interval_steps'] == 0:
                val = validate(model, valid_loader, nsched, device, cfg['loss'], pred_type=cfg['model'].get('prediction_type','v'))
                print(f"[valid @ step {global_step}] val_loss={val:.4f}")
                if val < best_val:
                    best_val = val
                    ckpt = os.path.join(cfg['experiment']['out_dir'], 'checkpoints', f'best_step{global_step}.pt')
                    torch.save({'model': model.state_dict(), 'ema': ema.shadow, 'cfg': cfg}, ckpt)
                    print(f"Saved best checkpoint: {ckpt}")

            if global_step % cfg['train']['save_interval_steps'] == 0:
                ckpt = os.path.join(cfg['experiment']['out_dir'], 'checkpoints', f'ema_step{global_step}.pt')
                torch.save({'ema': ema.shadow, 'cfg': cfg}, ckpt)
                print(f"Saved EMA checkpoint: {ckpt}")

    ckpt = os.path.join(cfg['experiment']['out_dir'], 'checkpoints', f'final_ema.pt')
    torch.save({'ema': ema.shadow, 'cfg': cfg}, ckpt)
    print(f"Training finished. Saved {ckpt}")
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    run_training(args.config)

if __name__ == '__main__':
    main()

```

## 文件: `package.py`

```python
#!/usr/bin/env python3
import os
import re
import sys
import fnmatch

# 移除 from pathlib import Path，后续用 os 替代
# from pathlib import Path

# --- 配置区 ---
# 目标目录，使用 os 获取当前脚本所在目录
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

# 输出的Markdown文件
OUTPUT_FILE = "codebase_prompt.md"

# ==============================================================================
# 新增：仅包含（白名单）模式 - 用于筛选【文件内容】
# ==============================================================================
INCLUDE_ONLY_PATTERNS = [
    "*.py", "*.yaml"
]

# ==============================================================================
# 排除（黑名单）模式
# ==============================================================================
# 要排除的目录。此规则对【项目结构图】和【文件内容】两部分都生效。
EXCLUDE_DIRS = [
    "*/node_modules/*", "*/.git/*", "*/dist/*", "*/build/*", "*/.vscode/*", "*/.idea/*",
    "*/__pycache__/*", "*/venv/*", "*/.nuxt/*", "*/diaries/*", "*/runs/*", "*/output*/*"
]

# 要排除的文件类型或文件。
EXCLUDE_FILES = [
    "*.log", "*.tmp", "*.lock", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.DS_Store", "*.sqlite3", "*.db", "*.png", "*.ico", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.pth", "*.npy", "tokenizer.json", "alphagenome_pytorch"
]

# --- END 配置区 ---

def main():
    # 检查 SOURCE_DIR 是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源目录 '{SOURCE_DIR}' 不存在。")
        sys.exit(1)

    # 清空或创建输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('')

    # --- 1. 生成项目结构图 ---
    generate_project_structure()

    # --- 2. 拼接代码内容 ---
    generate_code_content()

    print(f"Done! 代码已拼接至 '{OUTPUT_FILE}'")

def generate_project_structure():
    """生成项目结构的Markdown文档"""
    # 预处理要排除的目录名
    exclude_names_pattern = []
    for dir_pattern in EXCLUDE_DIRS:
        dir_name_part = re.sub(r'/\*$', '', dir_pattern)
        dir_name_part = os.path.basename(dir_name_part)
        if dir_name_part and dir_name_part != '*' and dir_name_part != '.':
            exclude_names_pattern.append(dir_name_part)
    
    exclude_names_regex = '|'.join(exclude_names_pattern) if exclude_names_pattern else ''

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 项目结构\n\n")
        f.write(f"项目`{SOURCE_DIR}`的目录结构（已排除如`node_modules`等目录）：\n")
        f.write("```\n")
        f.write(f"{SOURCE_DIR}/\n")
        
        # 生成目录树
        generate_tree(SOURCE_DIR, '', exclude_names_regex, f)
        
        f.write("```\n\n")
        f.write("---\n\n")

def generate_tree(dir_path, prefix, exclude_regex, file_obj):
    """递归生成目录树"""
    # 获取当前目录下一级的所有文件和目录，并排序
    items = sorted(os.listdir(dir_path))
    
    for i, item in enumerate(items):
        # 检查是否需要排除这个文件或目录
        if exclude_regex and re.search(exclude_regex, item):
            continue
            
        item_path = os.path.join(dir_path, item)
        
        # 判断连接符
        connector = "├── "
        new_prefix = "│   "
        if i == len(items) - 1:
            connector = "└── "
            new_prefix = "    "
        
        # 判断是目录还是文件，并输出
        if os.path.isdir(item_path):
            file_obj.write(f"{prefix}{connector}{item}/\n")
            # 递归调用
            generate_tree(item_path, prefix + new_prefix, exclude_regex, file_obj)
        else:
            file_obj.write(f"{prefix}{connector}{item}\n")

def generate_code_content():
    """生成代码内容的Markdown文档"""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 代码内容\n\n")
        
        # 构建文件搜索条件
        if INCLUDE_ONLY_PATTERNS:
            print(f"模式：仅拼接匹配 {' '.join(INCLUDE_ONLY_PATTERNS)} 的文件内容")
            include_patterns = INCLUDE_ONLY_PATTERNS
        else:
            print("模式：拼接所有文件，除了黑名单中的文件和目录")
            include_patterns = None
        
        # 遍历所有文件
        for root, dirs, files in os.walk(SOURCE_DIR):
            # 排除目录
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), EXCLUDE_DIRS)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # 排除文件
                if should_exclude(file_path, EXCLUDE_DIRS + [f"*/{pattern}" for pattern in EXCLUDE_FILES]):
                    continue
                
                # 检查是否在包含列表中
                if include_patterns and not any(fnmatch.fnmatch(file_path, pattern) for pattern in include_patterns):
                    continue
                
                # 写入文件内容
                relative_path = os.path.relpath(file_path, SOURCE_DIR)
                f.write(f"## 文件: `{relative_path}`\n\n")
                
                # 获取文件扩展名作为代码块语言
                file_ext = os.path.splitext(file_path)[1].lstrip('.')
                f.write(f"```{'python' if file_ext == 'py' else file_ext}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"无法读取文件: {file_path} ({str(e)})\n")
                
                f.write("\n```\n\n")

def should_exclude(path, patterns):
    """检查路径是否应被排除"""
    path_str = str(path)
    return any(re.search(pattern.replace('*', '.*'), path_str) for pattern in patterns)

if __name__ == "__main__":
    main()
```

## 文件: `infer.py`

```python

import os
import math
import argparse
import yaml
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from model import UNet1D
from train import NoiseSchedule, x0_from_v, v_from

def load_audio_mono(path: str, target_sr: int):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        x = torch.tensor(audio, dtype=torch.float32)[None, None, :]
        ratio = target_sr / sr
        new_len = int(round(x.shape[-1] * ratio))
        x = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
        audio = x[0,0].numpy()
    return audio.astype(np.float32)

@torch.no_grad()
def sample_guided(model, nsched, cond, steps=24, guidance_scale=1.5, sampler='ddim', dynamic_clip_percentile=0.999, device='cpu'):
    T = cond.shape[-1]
    x = torch.randn(1, 1, T, device=device)
    t_idxs = torch.linspace(nsched.steps - 1, 0, steps, device=device)
    t_idxs = torch.round(t_idxs).long()
    t_idxs = torch.clamp(t_idxs, 0, nsched.steps - 1)
    for i, t_idx in enumerate(t_idxs):
        a, s = nsched.alpha_sigma_at(t_idx[None])
        t_norm = (t_idx.float() + 0.5) / nsched.steps
        t_norm = t_norm[None].repeat(x.shape[0])
        pred_c = model(x, t_norm, cond=cond, cond_drop_mask=None)
        pred_u = model(x, t_norm, cond=cond, cond_drop_mask=torch.ones(x.shape[0], device=device, dtype=torch.bool))
        pred = pred_u + guidance_scale * (pred_c - pred_u)
        if model.prediction_type == 'v':
            x0 = x0_from_v(x, pred, a[:, None], s[:, None])
            eps = (x - a[:, None] * x0) / s[:, None]
        else:
            eps = pred
            x0 = (x - s[:, None] * eps) / a[:, None]
        if i < len(t_idxs) - 1:
            t_next = t_idxs[i+1]
            a_next, s_next = nsched.alpha_sigma_at(t_next[None])
            if sampler == 'ddim':
                x = a_next[:, None] * x0 + s_next[:, None] * eps
            elif sampler in ('ddim_ancestral', 'euler_ancestral'):
                a_t = a[:, None]; a_nt = a_next[:, None]
                sigma = torch.sqrt(torch.clamp((1 - a_nt**2)/(1 - a_t**2) * (1 - (a_t**2)/(a_nt**2)), min=0.0))
                noise = torch.randn_like(x)
                x = a_next[:, None] * x0 + torch.sqrt(torch.clamp(1 - a_next[:, None]**2 - sigma**2, min=0.0)) * eps + sigma * noise
            else:
                raise ValueError(f'Unknown sampler: {sampler}')
        else:
            x = x0
        if dynamic_clip_percentile is not None:
            p = dynamic_clip_percentile
            lo = torch.quantile(x, q=1.0-p); hi = torch.quantile(x, q=p)
            x = x.clamp(lo.item(), hi.item())
    return x

@torch.no_grad()
def sample_guided_chunked(model, nsched, cond, steps, guidance_scale, sampler,
                          dynamic_clip_percentile, sr, chunk_seconds, chunk_hop_seconds, device):
    B, Cc, T = cond.shape
    win = int(round(chunk_seconds * sr))
    hop = int(round(chunk_hop_seconds * sr))
    win = max(win, 1); hop = max(hop, 1)
    w = torch.hann_window(win, periodic=False, device=device).sqrt().view(1,1,win)
    out  = torch.zeros(1, 1, T, device=device)
    sumw = torch.zeros(1, 1, T, device=device)
    for start in range(0, T, hop):
        end = min(start + win, T)
        seg = cond[..., start:end]
        L = end - start
        if L < win:
            seg = F.pad(seg, (0, win - L))
        yseg = sample_guided(model, nsched, seg, steps=steps, guidance_scale=guidance_scale, sampler=sampler, dynamic_clip_percentile=dynamic_clip_percentile, device=device)
        out[..., start:start+win]  += yseg * w
        sumw[..., start:start+win] += w
        if end == T: break
    out = out / sumw.clamp_min(1e-8)
    return out

def run_inference(config_path: str, ckpt_path: str, low_path: str, out_path: str) -> str:
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr = cfg['data']['sample_rate']
    low = load_audio_mono(low_path, sr)
    cond = torch.tensor(low, dtype=torch.float32)[None, None, :]
    mcfg = cfg['model']
    model = UNet1D(mcfg['in_channels'], mcfg['cond_channels'], mcfg['base_channels'], tuple(mcfg['channel_mults']), mcfg['num_res_blocks'], mcfg['time_embed_dim'], mcfg['dropout'], mcfg['use_attention_scales'], mcfg['attn_num_heads'], mcfg['attn_head_dim'], mcfg.get('prediction_type','v')).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema' in ckpt:
        shadow = ckpt['ema']; missing, unexpected = model.load_state_dict(shadow, strict=False); print('Loaded EMA. missing:', missing, 'unexpected:', unexpected)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        raise RuntimeError('Invalid checkpoint content.')
    model.eval(); cond = cond.to(device)
    nsched = NoiseSchedule(steps=cfg['diffusion']['steps'], schedule=cfg['diffusion']['schedule'], device=device)
    steps = cfg['inference']['sample_steps']; gscale = cfg['inference']['guidance_scale']; sampler = cfg['inference']['sampler']; clip_p = cfg['inference']['dynamic_clip_percentile']; cs = cfg['inference']['chunk_seconds']; hs = cfg['inference']['chunk_hop_seconds']
    if cs is not None and cs > 0 and int(cs*sr) < cond.shape[-1]:
        y = sample_guided_chunked(model, nsched, cond, steps, gscale, sampler, clip_p, sr=sr, chunk_seconds=cs, chunk_hop_seconds=hs, device=device)
    else:
        y = sample_guided(model, nsched, cond, steps, gscale, sampler, clip_p, device)
    y = y.detach().cpu().numpy()[0,0]; sf.write(out_path, y, sr); return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--low', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    out = run_inference(args.config, args.ckpt, args.low, args.out); print('Wrote:', out)

if __name__ == '__main__':
    main()

```

