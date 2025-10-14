import os
import random
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from rich.progress import track

# --- AudioSR lowpass filtering utility ---
from audiosr.lowpass import lowpass
from audiosr.utils import _locate_cutoff_freq

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")

def load_audio_mono(path: str, target_sr: int):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        
    return audio.squeeze(0).numpy().astype(np.float32)

def create_lowpass_version(waveform, sr):
    """Creates a low-pass filtered version of the audio, simulating a low-quality input."""
    waveform = waveform.to(torch.float32)
    window = torch.hann_window(2048, device=waveform.device).to(torch.float32)
    
    stft_for_cutoff = torch.stft(waveform, n_fft=2048, hop_length=480, win_length=2048, window=window, return_complex=True, center=True)
    
    cutoff_freq = (_locate_cutoff_freq(stft_for_cutoff.abs(), percentile=0.985) / 1024) * (sr / 2)
    if cutoff_freq < 1000:
        cutoff_freq = sr / 2

    filtered_waveform = lowpass(
        waveform.cpu().numpy(),
        highcut=cutoff_freq,
        fs=sr,
        order=8,
        _type="butter",
    )
    return torch.from_numpy(filtered_waveform.copy()).to(torch.float32)

class PairedMelDataset(Dataset):
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
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        n_mels: int = 256,
        fmin: float = 20.0,
        fmax: Optional[float] = 24000.0,
        preload_to_ram: bool = False, # New parameter
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.segment_len = int(round(segment_seconds * sample_rate))
        self.is_train = (split == 'train')
        self.preload_to_ram = preload_to_ram
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0,
            norm='slaney',
            mel_scale='slaney'
        )

        pairs: List[str] = []
        for cat in categories:
            high_dir = os.path.join(dataset_root, cat, high_dir_name)
            if not os.path.isdir(high_dir):
                print(f"Warning: Directory not found {high_dir}")
                continue
            
            files = {f for f in os.listdir(high_dir) if f.lower().endswith(AUDIO_EXTS)}
            for name in sorted(files):
                pairs.append(os.path.join(high_dir, name))
        
        rng = random.Random(split_seed)
        rng.shuffle(pairs)
        n_valid = int(round(len(pairs) * float(valid_ratio)))
        self.files = pairs[n_valid:] if self.is_train else pairs[:n_valid]
        if not self.files:
            raise RuntimeError('No audio files found.')
        print(f"Found {len(self.files)} files for {split} split.")

        self.data_buffer = []
        if self.preload_to_ram:
            print(f"Pre-loading {len(self.files)} files into RAM for '{split}' split...")
            for idx in track(range(len(self.files)), description=f"Loading {split} data..."):
                self.data_buffer.append(self._load_item(idx))
            print("Pre-loading complete.")

    def __len__(self):
        return len(self.files)

    def _load_item(self, idx: int):
        filepath = self.files[idx]
        
        try:
            high_quality_wav = load_audio_mono(filepath, self.sample_rate)
            
            if len(high_quality_wav) < self.segment_len:
                pad_amount = self.segment_len - len(high_quality_wav)
                high_quality_wav = np.pad(high_quality_wav, (0, pad_amount), 'constant')
            elif len(high_quality_wav) > self.segment_len:
                start = random.randint(0, len(high_quality_wav) - self.segment_len) if self.is_train else 0
                high_quality_wav = high_quality_wav[start : start + self.segment_len]

            high_quality_tensor = torch.from_numpy(high_quality_wav).to(torch.float32)
            low_quality_tensor = create_lowpass_version(high_quality_tensor, self.sample_rate)
            
            with torch.no_grad():
                high_mel = self.mel_transform(high_quality_tensor)
                low_mel = self.mel_transform(low_quality_tensor)
            
            high_mel = torch.log(torch.clamp(high_mel, min=1e-5))
            low_mel = torch.log(torch.clamp(low_mel, min=1e-5))

            return {
                "fbank": high_mel.unsqueeze(0),
                "lowpass_mel": low_mel.unsqueeze(0)
            }
        except Exception as e:
            print(f"Error loading file {filepath}: {e}, skipping.")
            return None

    def __getitem__(self, idx: int):
        if self.preload_to_ram:
            item = self.data_buffer[idx]
            if item is None:
                # Handle cases where a file failed to load during preloading
                return self.__getitem__((idx + 1) % len(self.files))
            return item
        
        item = self._load_item(idx)
        if item is None:
            return self.__getitem__((idx + 1) % len(self.files))
        return item
