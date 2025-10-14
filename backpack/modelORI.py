# This file is adapted from the official AudioSR repository to be compatible with the project structure.
# It now integrates the LatentDiffusion model, VAE, and UNet in one place with the correct naming hierarchy.
#
# --- MODIFICATIONS ---
# 1. Replaced placeholder CLAP and Vocoder classes with detailed architectural skeletons
#    that precisely match the keys in the pre-trained checkpoint file.
# 2. Registered 'scale_factor' as a buffer in the main LatentDiffusion class.
# 3. Kept the custom EMA handler and the corrected UNet architecture from previous versions.
# 4. Corrected the shape of 'logit_scale_a' and 'logit_scale_t' to be scalars to match the checkpoint.
# 5. Fully built out the ClapWrapper's audio_branch with a SwinTransformer skeleton to match all checkpoint keys.
# 6. Set bias=False for STFT conv layers as per the checkpoint's structure.
# 7. Corrected a SyntaxError in the UNetModel's __init__ method.
# 8. Corrected an UnboundLocalError in the LatentDiffusion's register_schedule method.
# 9. Corrected the SwinTransformer's window_size and attn_mask shapes to match the checkpoint.
# 10. Re-instated the 'text_branch' and added the correct 'text_transform' and 'audio_transform' modules.
# 11. FINAL FIX: Added the '.sequential' nesting to the transform modules to match the checkpoint keys.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from contextlib import contextmanager
from functools import partial
import numpy as np

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

# --- Detailed Placeholder for Vocoder ---
class ResBlockModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dilation=1),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dilation=3),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dilation=5),
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
        ])

class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv1d(256, 1536, 7, 1, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(1536, 768, 12, 6, padding=3),
            nn.ConvTranspose1d(768, 384, 10, 5, padding=2, output_padding=1),
            nn.ConvTranspose1d(384, 192, 8, 4, padding=2),
            nn.ConvTranspose1d(192, 96, 4, 2, padding=1),
            nn.ConvTranspose1d(96, 48, 4, 2, padding=1),
        ])
        self.resblocks = nn.ModuleList([
            ResBlockModule(768, 768, 3), ResBlockModule(768, 768, 7), ResBlockModule(768, 768, 11), ResBlockModule(768, 768, 15),
            ResBlockModule(384, 384, 3), ResBlockModule(384, 384, 7), ResBlockModule(384, 384, 11), ResBlockModule(384, 384, 15),
            ResBlockModule(192, 192, 3), ResBlockModule(192, 192, 7), ResBlockModule(192, 192, 11), ResBlockModule(192, 192, 15),
            ResBlockModule(96, 96, 3), ResBlockModule(96, 96, 7), ResBlockModule(96, 96, 11), ResBlockModule(96, 96, 15),
            ResBlockModule(48, 48, 3), ResBlockModule(48, 48, 7), ResBlockModule(48, 48, 11), ResBlockModule(48, 48, 15),
        ])
        self.conv_post = nn.Conv1d(48, 1, 7, 1, padding=3)

# --- Detailed Placeholders for CLAP ---
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
        window_size = 8
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        self.relative_position_index = nn.Parameter(torch.zeros(64, 64), requires_grad=False)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4, dim)
        self.register_parameter("attn_mask", None)

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
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = nn.Module()
        self.spectrogram_extractor.stft.conv_real = nn.Conv1d(1, 513, 1024, bias=False)
        self.spectrogram_extractor.stft.conv_imag = nn.Conv1d(1, 513, 1024, bias=False)
        self.logmel_extractor = nn.Module()
        self.logmel_extractor.melW = nn.Parameter(torch.zeros(513, 64))
        self.bn0 = nn.BatchNorm1d(64)
        self.patch_embed = PatchEmbed()
        self.layers = nn.ModuleList([
            BasicLayer(128, 256, 2, 4, mask_shape=(64, 64, 64)),
            BasicLayer(256, 512, 2, 8, mask_shape=(16, 64, 64)),
            BasicLayer(512, 1024, 12, 16, mask_shape=(4, 64, 64)),
            BasicLayer(1024, None, 2, 32) 
        ])
        self.norm = nn.LayerNorm(1024)
        self.tscam_conv = nn.Conv2d(1024, 527, kernel_size=(2,3), padding=(0,1))
        self.head = nn.Linear(527, 527)

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
        
        # --- FIX: Added the '.sequential' nesting to match the checkpoint keys ---
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
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock): x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer): x = layer(x, context)
            else: x = layer(x)
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
class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, dims=2, use_checkpoint=False, use_scale_shift_norm=False):
        super().__init__()
        self.channels, self.emb_channels, self.dropout, self.out_channels, self.use_checkpoint, self.use_scale_shift_norm = channels, emb_channels, dropout, out_channels or channels, use_checkpoint, use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        self.skip_connection = nn.Identity() if self.out_channels == channels else conv_nd(dims, channels, self.out_channels, 1)
    def forward(self, x, emb): return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

# --- Main U-Net Model (Corrected Architecture) ---
class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, num_heads=-1, num_head_channels=-1, use_scale_shift_norm=False, resblock_updown=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, **kwargs):
        super().__init__()
        self.image_size, self.in_channels, self.model_channels, self.out_channels, self.num_res_blocks, self.attention_resolutions, self.dropout, self.channel_mult, self.conv_resample, self.num_classes, self.use_checkpoint, self.num_heads, self.num_head_channels = image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, num_classes, use_checkpoint, num_heads, num_head_channels
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
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
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
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim), SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
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
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)))
    def forward(self, x, timesteps, context=None, y=None, **kwargs):
        hs, t_emb = [], timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None: emb = emb + self.label_emb(y)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb, context)
        return self.out(h)

# --- VAE and Conditioner definitions ---
class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.encoder, self.decoder, self.vocoder = Encoder(**ddconfig), Decoder(**ddconfig), Vocoder()
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
    def encode(self, x): return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
    def decode(self, z): return self.decoder(self.post_quant_conv(z))
class VAEFeatureExtract(nn.Module):
    def __init__(self, first_stage_config):
        super().__init__()
        self.vae = instantiate_from_config(first_stage_config)
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
    def forward(self, batch):
        with torch.no_grad(): vae_embed = self.vae.encode(batch).sample()
        return vae_embed.detach()

# --- Main LDM Class ---
class LatentDiffusion(nn.Module):
    def __init__(self, first_stage_config, cond_stage_config, unet_config, beta_schedule="linear", timesteps=1000, loss_type="l2", parameterization="v", scale_factor=1.0, scale_by_std=False, use_ema=True, **kwargs):
        super().__init__()
        self.scale_by_std, self.parameterization, self.model = scale_by_std, parameterization, DiffusionWrapper(unet_config)
        if use_ema: self.model_ema = CustomLitEma(self.model)
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.cond_stage_models, self.cond_stage_model_metadata = nn.ModuleList(), {}
        self.instantiate_cond_stage(cond_stage_config)
        self.clap = ClapWrapper()
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps, scale_factor=scale_factor)
        self.loss_type = loss_type
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
        for key, meta in self.cond_stage_model_metadata.items():
            if meta["conditioning_key"] == "concat": cond[key] = self.cond_stage_models[meta["model_idx"]](c_concat_data)
        return z, cond
    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample() if isinstance(encoder_posterior, DiagonalGaussianDistribution) else encoder_posterior
        return self.scale_factor * z
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    def get_v(self, x, noise, t): return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x)
    def apply_model(self, x_noisy, t, cond): return self.model(x_noisy, t, cond)
    def get_loss(self, pred, target):
        if self.loss_type == 'l1': return F.l1_loss(pred, target)
        elif self.loss_type == 'l2': return F.mse_loss(pred, target)
        else: raise NotImplementedError()
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        target = self.get_v(x_start, noise, t) if self.parameterization == "v" else noise
        loss = self.get_loss(model_output, target)
        return loss, {"loss": loss}
    def forward(self, batch):
        z, c = self.get_input(batch)
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=z.device).long()
        return self.p_losses(z, c, t)
class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)
    def forward(self, x, t, cond_dict={}):
        xc, context = x, None
        for key, value in cond_dict.items():
            if "concat" in key: xc = torch.cat([x, value], dim=1)
        return self.diffusion_model(xc, t, context=context)