
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
