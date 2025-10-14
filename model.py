import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from contextlib import contextmanager
from functools import partial
import numpy as np
import warnings
import sys
import os
import torchaudio

# --- Official AudioSR Imports (or their equivalents) ---
from audiosr.latent_diffusion.util import (
    exists,
    default,
    instantiate_from_config,
)
from audiosr.latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from audiosr.latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from audiosr.latent_diffusion.modules.attention import SpatialTransformer
from audiosr.latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder # VAE components

# ==============================================================================
# SECTION: SPECTRAL LOSS FUNCTIONS (直接集成在此文件中)
# ==============================================================================

def spectral_convergence_loss(x, y):
    """计算频谱收敛损失 (Spectral Convergence Loss)"""
    return torch.norm(torch.abs(y) - torch.abs(x), p="fro") / torch.norm(torch.abs(y), p="fro")

def log_stft_magnitude_loss(x, y):
    """计算对数STFT幅值损失 (Log STFT Magnitude Loss)"""
    x_mag = torch.abs(x) + 1e-8
    y_mag = torch.abs(y) + 1e-8
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

class STFTLoss(nn.Module):
    """单一分辨率的STFT损失模块"""
    def __init__(self, fft_size, hop_size, win_length, window):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
    
    def forward(self, x_wav, y_wav):
        self.window = self.window.to(x_wav.device)
        x_stft = torch.stft(x_wav.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=self.window, return_complex=True)
        y_stft = torch.stft(y_wav.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=self.window, return_complex=True)
        sc_loss = spectral_convergence_loss(x_stft, y_stft)
        mag_loss = log_stft_magnitude_loss(x_stft, y_stft)
        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率STFT损失"""
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window"):
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl, window))

    def forward(self, x_wav, y_wav):
        sc_loss, mag_loss = 0.0, 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x_wav, y_wav)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss

class MelSpectrogramLoss(nn.Module):
    """梅尔频谱图的L1损失"""
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, f_min=f_min, f_max=f_max,
            power=1.0, norm='slaney'
        )

    def forward(self, x_wav, y_wav):
        x_mel = self.mel_transform(x_wav.squeeze(1))
        y_mel = self.mel_transform(y_wav.squeeze(1))
        return F.l1_loss(torch.log(torch.clamp(x_mel, min=1e-5)), 
                         torch.log(torch.clamp(y_mel, min=1e-5)))

# ==============================================================================
# SECTION: MODEL DEFINITIONS
# ==============================================================================


# === GLOBAL ENCODER FOR WHOLE AUDIO PROCESSING ===
class GlobalAudioEncoder(nn.Module):
    """
    Lightweight CNN+Transformer Global Encoder that processes whole audio
    and extracts critical global information to condition the diffusion process.
    """
    def __init__(self, 
                 input_channels=256,      # Mel spectrogram channels
                 hidden_dim=128,          # Hidden dimension for efficiency
                 num_heads=4,             # Multi-head attention
                 num_layers=4,            # Transformer layers
                 context_dim=64,          # Output context dimension
                 max_seq_len=1024):       # Maximum sequence length
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # === CNN Feature Extractor ===
        # Efficiently downsample and extract features from mel spectrogram
        self.cnn_encoder = nn.Sequential(
            # First conv block - reduce frequency dimension
            nn.Conv2d(1, 32, kernel_size=(8, 3), stride=(4, 1), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second conv block - further downsample
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third conv block - final feature extraction
            nn.Conv2d(64, hidden_dim, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # === Positional Encoding ===
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, context_dim)
        )
        
        # === Global Context Pooling ===
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(context_dim, context_dim)
        
    def forward(self, x):
        """
        Forward pass of Global Encoder
        Args:
            x: Input mel spectrogram [batch_size, 1, mel_bins, time_steps]
        Returns:
            global_context: Global context features [batch_size, context_dim]
            sequence_context: Sequence context features [batch_size, seq_len, context_dim]
        """
        batch_size = x.shape[0]
        
        # === CNN Feature Extraction ===
        # x: [B, 1, mel_bins, time] -> [B, hidden_dim, reduced_mel, time]
        features = self.cnn_encoder(x)  # [B, hidden_dim, ~16, time]
        
        # Reshape for transformer: [B, hidden_dim, reduced_mel, time] -> [B, seq_len, hidden_dim]
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # [B, time, hidden_dim, reduced_mel]
        features = features.view(B, W, C * H)  # [B, time, hidden_dim * reduced_mel]
        
        # Project to hidden_dim
        if C * H != self.hidden_dim:
            if not hasattr(self, '_adaptive_proj'):
                self._adaptive_proj = nn.Linear(C * H, self.hidden_dim).to(features.device)
            features = self._adaptive_proj(features)
        
        # === Add Positional Encoding ===
        seq_len = features.shape[1]
        if seq_len <= self.pos_encoding.shape[1]:
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            # If sequence is longer, interpolate positional encoding
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        features = features + pos_enc
        
        # === Transformer Processing ===
        # Self-attention to capture global dependencies
        transformed_features = self.transformer(features)  # [B, seq_len, hidden_dim]
        
        # === Output Projections ===
        sequence_context = self.output_proj(transformed_features)  # [B, seq_len, context_dim]
        
        # Global context via pooling
        global_features = self.global_pool(sequence_context.transpose(1, 2)).squeeze(-1)  # [B, context_dim]
        global_context = self.global_proj(global_features)  # [B, context_dim]
        
        return global_context, sequence_context

class GlobalConditionedSpatialTransformer(SpatialTransformer):
    """
    Enhanced SpatialTransformer that incorporates global audio context
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None, global_context_dim=64):
        super().__init__(in_channels, n_heads, d_head, depth, dropout, context_dim)
        self.global_context_dim = global_context_dim
        
        if global_context_dim is not None:
            # Project global context to match transformer dimensions
            self.global_proj = nn.Linear(global_context_dim, n_heads * d_head)
            
            # Global context attention
            self.global_attn = nn.MultiheadAttention(
                embed_dim=n_heads * d_head,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
    
    def forward(self, x, context=None, global_context=None):
        """
        Forward pass with global context integration
        Args:
            x: Input features [B, C, H, W]
            context: Local context from other conditions
            global_context: Global audio context [B, global_context_dim] or [B, seq_len, global_context_dim]
        """
        if global_context is not None and hasattr(self, 'global_proj'):
            # Process global context
            if global_context.dim() == 2:
                # Single global vector: [B, global_context_dim] -> [B, 1, n_heads * d_head]
                global_features = self.global_proj(global_context).unsqueeze(1)
            else:
                # Sequence of global vectors: [B, seq_len, global_context_dim] -> [B, seq_len, n_heads * d_head]
                global_features = self.global_proj(global_context)
            
            # Reshape input for attention
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
            
            # Apply global context attention
            attended_x, _ = self.global_attn(x_flat, global_features, global_features)
            
            # Reshape back
            x += attended_x.transpose(1, 2).view(B, C, H, W)
        
        # Call parent forward with potentially modified x
        return super().forward(x, context)

# === EXACT AUDIOSR VOCODER ===
class AudioSRVocoder(nn.Module):
    """
    Exact AudioSR HiFi-GAN vocoder implementation matching output size perfectly
    """
    def __init__(self):
        super().__init__()
        
        # Pre-conv
        self.conv_pre = nn.Conv1d(256, 1536, 7, 1, padding=3)
        
        # Upsampling - EXACT AudioSR configuration
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(1536, 768, 12, 6, padding=3),
            nn.ConvTranspose1d(768, 384, 10, 5, padding=2, output_padding=1),
            nn.ConvTranspose1d(384, 192, 8, 4, padding=2),
            nn.ConvTranspose1d(192, 96, 4, 2, padding=1),
            nn.ConvTranspose1d(96, 48, 4, 2, padding=1),
        ])
        
        # ResBlocks - EXACT AudioSR structure (20 total)
        self.resblocks = nn.ModuleList()
        
        channels_list = [768, 768, 768, 768, 384, 384, 384, 384, 192, 192, 192, 192, 96, 96, 96, 96, 48, 48, 48, 48]
        kernels_list = [3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15]
        
        for channels, kernel_size in zip(channels_list, kernels_list):
            self.resblocks.append(AudioSRVocoderResBlock(channels, kernel_size))
        
        # Post conv
        self.conv_post = nn.Conv1d(48, 1, 7, 1, padding=3)
        
        # Don't apply weight norm during initialization - will be applied after weight loading
        self._weight_norm_applied = False
    
    def apply_weight_norm(self):
        """Apply weight normalization to all conv layers"""
        from torch.nn.utils import weight_norm
        
        self.conv_pre = weight_norm(self.conv_pre)
        
        for up in self.ups:
            up = weight_norm(up)
        
        for resblock in self.resblocks:
            resblock.apply_weight_norm()
        
        self.conv_post = weight_norm(self.conv_post)
    
    def forward(self, x):
        x = self.conv_pre(x)
        
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Apply corresponding resblocks
            xs = None
            for j in range(4):  # 4 resblocks per upsampling stage
                resblock_idx = i * 4 + j
                if xs is None:
                    xs = self.resblocks[resblock_idx](x)
                else:
                    xs += self.resblocks[resblock_idx](x)
            x = xs / 4
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for compatibility"""
        from torch.nn.utils import remove_weight_norm
        
        try:
            remove_weight_norm(self.conv_pre)
        except ValueError:
            pass
            
        for up in self.ups:
            try:
                remove_weight_norm(up)
            except ValueError:
                pass
        
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        
        try:
            remove_weight_norm(self.conv_post)
        except ValueError:
            pass

class AudioSRVocoderResBlock(nn.Module):
    """HiFi-GAN ResBlock for vocoder"""
    def __init__(self, channels, kernel_size, dilation=(1, 3, 5)):
        super().__init__()
        
        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     padding=get_padding(kernel_size, d), dilation=d)
            for d in dilation
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     padding=get_padding(kernel_size, 1), dilation=1)
            for _ in dilation
        ])
    
    def apply_weight_norm(self):
        """Apply weight normalization to all conv layers"""
        from torch.nn.utils import weight_norm
        
        for i in range(len(self.convs1)):
            self.convs1[i] = weight_norm(self.convs1[i])
            self.convs2[i] = weight_norm(self.convs2[i])
    
    def remove_weight_norm(self):
        """Remove weight normalization"""
        from torch.nn.utils import remove_weight_norm
        
        for i in range(len(self.convs1)):
            try:
                remove_weight_norm(self.convs1[i])
            except ValueError:
                pass
            try:
                remove_weight_norm(self.convs2[i])
            except ValueError:
                pass
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

# --- Import AudioSR's HiFi-GAN Vocoder ---
# We'll use the original AudioSR HiFi-GAN implementation
try:
    import sys
    import os
    sys.path.append('/home/husrcf/Code/Python/projectlily Z')
    from audiosr.hifigan.models import Generator as HiFiGANGenerator
    from audiosr.utilities.model import get_vocoder_config_48k, vocoder_infer
    from audiosr.hifigan import AttrDict
    AUDIOSR_AVAILABLE = True
except ImportError:
    print("Warning: AudioSR not available, using fallback vocoder")
    AUDIOSR_AVAILABLE = False
    
    # Fallback simplified vocoder for when AudioSR is not available
    class HiFiGANGenerator(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.conv_pre = nn.Conv1d(256, 512, 7, 1, padding=3)
            self.conv_post = nn.Conv1d(32, 1, 7, 1, padding=3)
        def forward(self, x):
            return torch.tanh(self.conv_post(F.leaky_relu(self.conv_pre(x))))
        def remove_weight_norm(self): pass

# === EXACT AUDIOSR CLAP WITH ALL MISSING MODULES ===
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # MISSING MODULES ADDED:
        self.attn_drop = nn.Dropout(0.0)  # Attention dropout
        self.proj_drop = nn.Dropout(0.0)  # Projection dropout
        self.softmax = nn.Softmax(dim=-1)  # Softmax activation
        
        window_size = 8
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        self.relative_position_index = nn.Parameter(torch.zeros(64, 64), requires_grad=False)
    
    def forward(self, x):
        # Simplified forward pass - full implementation would be more complex
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention calculation
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # MISSING MODULE ADDED:
        self.drop_path = DropPath(0.1)  # DropPath for stochastic depth
        
        self.mlp = Mlp(dim, dim * 4, dim)
        self.register_parameter("attn_mask", None)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(4 * in_channels)

class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, num_blocks, num_heads, mask_shape=None):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim, num_heads) for _ in range(num_blocks)])
        if mask_shape is not None:
            for i in range(num_blocks):
                if i % 2 != 0:
                    self.blocks[i].attn_mask = nn.Parameter(torch.zeros(mask_shape), requires_grad=False)
        if out_dim is not None:
            self.downsample = PatchMerging(dim, out_dim)
        else:
            self.downsample = None

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(128)

class ClapAudioBranch(nn.Module):
    """CLAP Audio Branch with ALL missing modules"""
    def __init__(self):
        super().__init__()
        # Existing modules...
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = nn.Module()
        self.spectrogram_extractor.stft.conv_real = nn.Conv1d(1, 513, 1024, bias=False)
        self.spectrogram_extractor.stft.conv_imag = nn.Conv1d(1, 513, 1024, bias=False)
        
        self.logmel_extractor = nn.Module()
        self.logmel_extractor.melW = nn.Parameter(torch.zeros(513, 64))
        
        self.bn0 = nn.BatchNorm1d(64)
        
        # Patch embedding
        self.patch_embed = PatchEmbed()
        
        # Layers with proper blocks
        self.layers = nn.ModuleList([
            BasicLayer(128, 256, 2, 4, mask_shape=(64, 64, 64)),
            BasicLayer(256, 512, 2, 8, mask_shape=(16, 64, 64)),
            BasicLayer(512, 1024, 12, 16, mask_shape=(4, 64, 64)),
            BasicLayer(1024, None, 2, 32) 
        ])
        
        self.norm = nn.LayerNorm(1024)
        self.tscam_conv = nn.Conv2d(1024, 527, kernel_size=(2,3), padding=(0,1))
        self.head = nn.Linear(527, 527)
        
        # CRITICAL MISSING MODULE:
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(50265, 768)
        self.position_embeddings = nn.Embedding(514, 768)
        self.token_type_embeddings = nn.Embedding(1, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(514).expand((1, -1)))

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)

class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = nn.Module()
        self.output.dense = nn.Linear(768, 768)
        self.output.LayerNorm = nn.LayerNorm(768, eps=1e-12)

class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 3072)

class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(12)])

class BertPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)

class ClapTextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()
        self.pooler = BertPooler()

class ClapWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.logit_scale_a = nn.Parameter(torch.tensor(0.0))
        self.model.logit_scale_t = nn.Parameter(torch.tensor(0.0))
        self.model.audio_branch = ClapAudioBranch()
        self.model.text_branch = ClapTextBranch()
        
        self.model.text_projection = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 512))
        self.model.audio_projection = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512))
        
        # Transform modules without .sequential nesting to match checkpoint
        # Create transform modules with .sequential structure to match AudioSR
        self.model.text_transform = nn.Module()
        self.model.text_transform.sequential = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        self.model.audio_transform = nn.Module()
        self.model.audio_transform.sequential = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        self.mel_transform = nn.Module()
        self.mel_transform.spectrogram = nn.Module()
        self.mel_transform.spectrogram.window = nn.Parameter(torch.zeros(1024))
        self.mel_transform.mel_scale = nn.Module()
        self.mel_transform.mel_scale.fb = nn.Parameter(torch.zeros(513, 64))

# --- Custom EMA Handler ---
class CustomLitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.m_name2s_name = {name: name.replace('.', '') for name, p in model.named_parameters()}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int64) if use_num_updates else torch.tensor(-1, dtype=torch.int64))
        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = self.m_name2s_name[name]
                self.register_buffer(s_name, p.clone().detach().data)
        self.collected_params = []
    @torch.no_grad()
    def forward(self, model):
        self.num_updates += 1
        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = self.m_name2s_name[name]
                s_param = self.get_buffer(s_name)
                s_param.sub_((1 - self.decay) * (s_param - p.data))
    def store(self, params): self.collected_params = [p.clone() for p in params]
    def restore(self, params):
        for p_old, p_new in zip(params, self.collected_params): p_old.data.copy_(p_new.data)
        self.collected_params = []
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.get_buffer(self.m_name2s_name[name]))

# --- Building Blocks for UNet (ResBlock, Upsample, etc.) ---
class TimestepBlock(nn.Module):
    @staticmethod
    def forward(x, emb): raise NotImplementedError
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, global_context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock): 
                x = layer(x, emb)
            elif isinstance(layer, GlobalConditionedSpatialTransformer): 
                x = layer(x, context, global_context)
            elif isinstance(layer, SpatialTransformer): 
                x = layer(x, context)
            else: 
                x = layer(x)
        return x
class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels, self.out_channels, self.use_conv, self.dims = channels, out_channels or channels, use_conv, dims
        if use_conv: self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv: x = self.conv(x)
        return x
class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels, self.out_channels, self.use_conv, self.dims = channels, out_channels or channels, use_conv, dims
        stride = 2
        if use_conv: self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else: self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
class AudioSRResBlock(TimestepBlock):
    """
    Exact ResBlock implementation matching AudioSR with h_upd modules
    """
    def __init__(self, channels, emb_channels, dropout, out_channels=None, dims=2, 
                 use_checkpoint=False, use_scale_shift_norm=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Standard ResBlock components
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        
        # CRITICAL: Add the missing h_upd module
        self.h_upd = nn.Identity()  # This matches AudioSR's h_upd structure
    
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h

# Keep original ResBlock for compatibility
ResBlock = AudioSRResBlock

# --- Main U-Net Model (Corrected Architecture) ---
class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, num_heads=-1, num_head_channels=-1, use_scale_shift_norm=False, resblock_updown=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, use_global_encoder=True, global_context_dim=64,
                 # --- 1. 新增参数，并提供默认值以保持向后兼容 ---
                 global_encoder_hidden_dim=128,
                 global_encoder_heads=4,
                 global_encoder_layers=2,
                 **kwargs):
        super().__init__()
        self.image_size, self.in_channels, self.model_channels, self.out_channels, self.num_res_blocks, self.attention_resolutions, self.dropout, self.channel_mult, self.conv_resample, self.num_classes, self.use_checkpoint, self.num_heads, self.num_head_channels = image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, num_classes, use_checkpoint, num_heads, num_head_channels
        
        # Global Encoder Integration
        self.use_global_encoder = use_global_encoder
        self.global_context_dim = global_context_dim
        if use_global_encoder:
            # --- 2. 使用从config传入的参数来实例化 GlobalAudioEncoder ---
            self.global_encoder = GlobalAudioEncoder(
                input_channels=256,
                hidden_dim=global_encoder_hidden_dim,
                num_heads=global_encoder_heads,
                num_layers=global_encoder_layers,
                context_dim=global_context_dim,
                max_seq_len=1024
            )     
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        if self.num_classes is not None: self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size, input_block_chans, ch, ds = model_channels, [model_channels], model_channels, 1
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    if self.use_global_encoder:
                        layers.append(GlobalConditionedSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, global_context_dim=self.global_context_dim))
                    else:
                        layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        num_heads = ch // num_head_channels
        dim_head = num_head_channels
        if self.use_global_encoder:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), 
                GlobalConditionedSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, global_context_dim=self.global_context_dim), 
                GlobalConditionedSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, global_context_dim=self.global_context_dim), 
                ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
            )
        else:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), 
                SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim), 
                SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim), 
                ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
            )
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    if self.use_global_encoder:
                        layers.append(GlobalConditionedSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, global_context_dim=self.global_context_dim))
                        layers.append(GlobalConditionedSpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, global_context_dim=self.global_context_dim))
                    else:
                        layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                        layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)))
    def forward(self, x, timesteps, context=None, y=None, full_audio=None, **kwargs):
        hs, t_emb = [], timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None: emb = emb + self.label_emb(y)
        
        # Extract global context if Global Encoder is enabled
        global_context = None
        if self.use_global_encoder and full_audio is not None:
            global_context_vec, sequence_context = self.global_encoder(full_audio)
            global_context = global_context_vec  # Use global context for simplicity
        
        h = x
        for module in self.input_blocks:
            if self.use_global_encoder:
                h = module(h, emb, context, global_context)
            else:
                h = module(h, emb, context)
            hs.append(h)
        
        if self.use_global_encoder:
            h = self.middle_block(h, emb, context, global_context)
        else:
            h = self.middle_block(h, emb, context)
        
        for module in self.output_blocks:
            if self.use_global_encoder:
                h = module(torch.cat([h, hs.pop()], dim=1), emb, context, global_context)
            else:
                h = module(torch.cat([h, hs.pop()], dim=1), emb, context)
        
        return self.out(h)

# === COMPLETE AUDIOSR-EXACT AUTOENCODER ===
class AudioSRAutoEncoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # Use EXACT AudioSR vocoder
        self.vocoder = AudioSRVocoder()
        
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
    
    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
    
    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))
    
    def decode_to_waveform(self, z):
        """Decode to waveform using exact AudioSR vocoder"""
        mel_spec = self.decode(z)
        
        batch_size = mel_spec.shape[0]
        
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)
            
            # Correct reshaping for AudioSR
            if mel_spec.shape[1] == 512 and mel_spec.shape[2] == 512:
                total_elements = mel_spec.shape[1] * mel_spec.shape[2]
                time_frames = total_elements // 256  # n_mels = 256
                mel_spec = mel_spec.reshape(batch_size, 256, time_frames)
        
        # CRITICAL: Apply AudioSR's spectral denormalization 
        from audiosr.utils import spectral_de_normalize_torch
        mel_spec_denormalized = spectral_de_normalize_torch(mel_spec)
        
        waveform = self.vocoder(mel_spec_denormalized)
        
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        return waveform

# Keep original AutoencoderKL for compatibility
class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig) 
        
        # Initialize AudioSR's HiFi-GAN vocoder
        n_mels = ddconfig.get("mel_bins", 256)
        self.n_mels = n_mels
        
        # Use EXACT AudioSR vocoder
        self.vocoder = AudioSRVocoder()
            
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
    
    def encode(self, x): 
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
    
    def decode(self, z): 
        return self.decoder(self.post_quant_conv(z))
    
    def decode_to_waveform(self, z):
        """
        Decode latent to mel spectrogram and then to waveform using AudioSR's HiFi-GAN
        Args:
            z: Latent tensor of shape (batch, embed_dim, h, w)
        Returns:
            waveform: Audio waveform tensor of shape (batch, 1, time_samples)
        """
        # First decode to mel spectrogram
        mel_spec = self.decode(z)  # Shape: (batch, 1, h, w)
        
        batch_size = mel_spec.shape[0]
        
        if mel_spec.dim() == 4:  # (batch, channels, height, width)
            # Remove channel dimension: (batch, height, width)  
            mel_spec = mel_spec.squeeze(1)
            
            # Reshape to correct mel spectrogram format
            if mel_spec.shape[1] == 512 and mel_spec.shape[2] == 512:
                # Reshape 512x512 to get 256 mel channels
                total_elements = mel_spec.shape[1] * mel_spec.shape[2]  # 512*512 = 262144
                time_frames = total_elements // self.n_mels  # 262144/256 = 1024
                mel_spec = mel_spec.reshape(batch_size, self.n_mels, time_frames)
            elif mel_spec.shape[1] != self.n_mels:
                if mel_spec.shape[2] == self.n_mels:
                    # Transpose to get correct shape
                    mel_spec = mel_spec.transpose(1, 2)  # (batch, n_mels, time_frames)
                else:
                    # General reshape case
                    total_elements = mel_spec.shape[1] * mel_spec.shape[2]
                    time_frames = total_elements // self.n_mels
                    mel_spec = mel_spec.reshape(batch_size, self.n_mels, time_frames)
        
        # Ensure we have the right shape
        assert mel_spec.shape[1] == self.n_mels, f"Expected {self.n_mels} mel channels, got {mel_spec.shape[1]}"
        
        if AUDIOSR_AVAILABLE:
            # Use AudioSR's vocoder directly (not vocoder_infer which has dimension issues)
            # The vocoder expects (batch, n_mels, time) format, which is what we have
            with torch.no_grad():
                # CRITICAL: VAE decoder outputs need to be de-normalized like in AudioSR
                from audiosr.utils import spectral_de_normalize_torch
                
                # Apply AudioSR's spectral denormalization (dynamic range decompression)
                mel_spec_denormalized = spectral_de_normalize_torch(mel_spec)
                
                # HiFi-GAN vocoder expects the denormalized mel spectrograms
                waveform = self.vocoder(mel_spec_denormalized)
                
                # Ensure correct output shape (batch, 1, time_samples)
                if waveform.dim() == 3 and waveform.shape[1] == 1:
                    # Shape is already (batch, 1, time_samples)
                    pass
                elif waveform.dim() == 2:
                    # Shape is (batch, time_samples), add channel dimension
                    waveform = waveform.unsqueeze(1)
                elif waveform.dim() == 1:
                    # Shape is (time_samples), add batch and channel dimensions
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
        else:
            # Fallback: use the vocoder directly when AudioSR is not available
            # Apply dynamic range decompression for consistency
            try:
                from audiosr.utils import spectral_de_normalize_torch
                mel_spec_denormalized = spectral_de_normalize_torch(mel_spec)
            except:
                # Fallback if AudioSR utils not available
                mel_spec_denormalized = torch.exp(mel_spec)
            waveform = self.vocoder(mel_spec_denormalized)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
        
        return waveform
class VAEFeatureExtract(nn.Module):
    def __init__(self, first_stage_config,**kwargs):
        super().__init__()
        self.vae = instantiate_from_config(first_stage_config)
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
    def forward(self, batch):
        with torch.no_grad(): vae_embed = self.vae.encode(batch).sample()
        return vae_embed.detach()

# --- Main LDM Class ---
class LatentDiffusion(nn.Module):
    def __init__(self, first_stage_config, cond_stage_config, unet_config, beta_schedule="linear", timesteps=1000, loss_type="l2", parameterization="v", scale_factor=1.0, scale_by_std=False, use_ema=True, data_config=None, stft_loss_weight=0.0, mel_loss_weight=0.0, **kwargs):
        super().__init__()
        self.scale_by_std, self.parameterization, self.model = scale_by_std, parameterization, DiffusionWrapper(unet_config)
        if use_ema: self.model_ema = CustomLitEma(self.model)
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.cond_stage_models = nn.ModuleDict()
        for key, cfg in cond_stage_config.items():
            # 我们只关心模型本身，其他元数据可以直接从cfg获取
            self.cond_stage_models[key] = instantiate_from_config(cfg)
        self.clap = ClapWrapper()
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps, scale_factor=scale_factor)
        self.loss_type = loss_type

        # --- 新增：实例化频谱损失函数 ---
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        if (self.stft_loss_weight > 0 or self.mel_loss_weight > 0) and data_config is not None:
            if self.stft_loss_weight > 0:
                self.stft_loss = MultiResolutionSTFTLoss()
                print("STFT Loss enabled.")
            if self.mel_loss_weight > 0:
                self.mel_loss = MelSpectrogramLoss(
                    sample_rate=data_config.get('sample_rate', 48000),
                    n_fft=data_config.get('n_fft', 2048),
                    hop_length=data_config.get('hop_length', 480),
                    win_length=data_config.get('win_length', 2048),
                    n_mels=data_config.get('n_mels', 256),
                    f_min=data_config.get('fmin', 20.0),
                    f_max=data_config.get('fmax', 24000.0)
                )
                print("Mel Loss enabled.")
        else:
             print("Spectral losses are disabled (weight is 0 or data_config not provided).")
        # --- 新增结束 ---


    def register_schedule(self, beta_schedule, timesteps, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, scale_factor=1.0):
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        # --- FIX: Clip alphas_cumprod to prevent log(0) and division by zero warnings ---
        alphas_cumprod = np.clip(alphas_cumprod, a_min=0.0, a_max=1.0 - 1e-8)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas)); self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)); self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod))); self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod))); self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        posterior_variance = (1 - self.scale_by_std) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance)); self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('logvar', torch.zeros(timesteps))
    def instantiate_cond_stage(self, config):
        for key, cfg in config.items():
            model = instantiate_from_config(cfg)
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[key] = {"model_idx": len(self.cond_stage_models) - 1, "cond_stage_key": cfg.get("cond_stage_key"), "conditioning_key": cfg.get("conditioning_key")}
    @contextmanager
    def ema_scope(self, context=None):
        if hasattr(self, 'model_ema'):
            self.model_ema.store([p.data for p in self.model.parameters()]); self.model_ema.copy_to(self.model)
        try: yield None
        finally:
            if hasattr(self, 'model_ema'): self.model_ema.restore([p.data for p in self.model.parameters()])
    @torch.no_grad()
    def get_input(self, batch):
        x, c_concat_data = batch["fbank"], batch["lowpass_mel"]
        z = self.get_first_stage_encoding(self.first_stage_model.encode(x))
        cond = {}
        for key, model in self.cond_stage_models.items():
            # 我们假设所有条件模型都是 concat 类型，这符合当前配置
            cond[key] = model(c_concat_data)
        return z, cond
    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample() if isinstance(encoder_posterior, DiagonalGaussianDistribution) else encoder_posterior
        return self.scale_factor * z
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    def get_v(self, x, noise, t): return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x)
    def apply_model(self, x_noisy, t, cond, full_audio=None): return self.model(x_noisy, t, cond, full_audio=full_audio)
    def get_loss(self, pred, target):
        if self.loss_type == 'l1': return F.l1_loss(pred, target)
        elif self.loss_type == 'l2': return F.mse_loss(pred, target)
        else: raise NotImplementedError()
    def p_losses(self, x_start, cond, t, noise=None, full_audio=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, full_audio=full_audio)
        # 1. 计算潜在空间损失
        target = self.get_v(x_start, noise, t) if self.parameterization == "v" else noise
        latent_loss = self.get_loss(model_output, target)
        total_loss = latent_loss
        log_dict = {"loss/latent": latent_loss.clone().detach()}
        # 2. 计算频谱损失（如果启用）
        if self.stft_loss_weight > 0 or self.mel_loss_weight > 0:
            # a. 重建预测的干净潜在表示 x_start_pred
            if self.parameterization == "v":
                sqrt_alpha_prod = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_noisy.shape)
                sqrt_one_minus_alpha_prod = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
                x_start_pred = sqrt_alpha_prod * x_noisy - sqrt_one_minus_alpha_prod * model_output
            else: # "epsilon"
                sqrt_recip_alphas_cumprod = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_noisy.shape)
                sqrt_recipm1_alphas_cumprod = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_noisy.shape)
                x_start_pred = sqrt_recip_alphas_cumprod * x_noisy - sqrt_recipm1_alphas_cumprod * model_output

            # b. 解码为音频波形
            with torch.no_grad(): # 解码过程不计算梯度，只用于计算损失
                 target_wav = self.first_stage_model.decode_to_waveform(x_start)
            
            # 预测波形的解码需要梯度以反向传播
            pred_wav = self.first_stage_model.decode_to_waveform(x_start_pred)
            
            # c. 计算损失
            if self.mel_loss_weight > 0 and hasattr(self, 'mel_loss'):
                mel_loss = self.mel_loss(pred_wav, target_wav)
                total_loss += self.mel_loss_weight * mel_loss
                log_dict["loss/mel"] = mel_loss.detach()
            
            if self.stft_loss_weight > 0 and hasattr(self, 'stft_loss'):
                sc_loss, mag_loss = self.stft_loss(pred_wav, target_wav)
                stft_loss = sc_loss + mag_loss
                total_loss += self.stft_loss_weight * stft_loss
                log_dict["loss/stft"] = stft_loss.detach()
            
        log_dict["loss/total"] = total_loss.clone().detach()

        return total_loss, log_dict
    def forward(self, batch):
        z, c = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=z.device).long()
        
        # Pass the full audio (original high-quality mel) for global context
        full_audio = batch.get("fbank", None)  # Full audio for global encoder
        
        return self.p_losses(z, c, t, full_audio=full_audio)
    
    def load_checkpoint(self, checkpoint_path, strict=False):
        """
        Custom checkpoint loading function that handles missing keys gracefully
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            warnings.warn(f"Missing keys in checkpoint: {missing_keys[:10]}... (showing first 10)")
        if unexpected_keys:
            warnings.warn(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}... (showing first 10)")
        
        return missing_keys, unexpected_keys

class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)
    def forward(self, x, t, cond_dict={}, full_audio=None):
        """
        xc, context = x, None
        for key, value in cond_dict.items():
            if "concat" in key: 
                # Ensure spatial dimensions match before concatenation
                if value.shape[-2:] != x.shape[-2:]:
                    # Interpolate conditioning tensor to match input spatial dimensions
                    value = F.interpolate(value, size=x.shape[-2:], mode='bilinear', align_corners=False)
                xc = torch.cat([x, value], dim=1)
        return self.diffusion_model(xc, t, context=context, full_audio=full_audio)
        """
        xc, context = x, None
        for key, value in cond_dict.items():
            if "concat" in key: 
                # Ensure spatial dimensions match before concatenation
                if value.shape[-2:] != x.shape[-2:]:
                    # Interpolate conditioning tensor to match input spatial dimensions
                    value = F.interpolate(value, size=x.shape[-2:], mode='bilinear', align_corners=False)
                xc = torch.cat([x, value], dim=1)
        #
        # --- 修改建议 1: 结束 ---

        return self.diffusion_model(xc, t, context=context, full_audio=full_audio)