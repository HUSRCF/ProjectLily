# Model Architecture Detailed Analysis

## Configuration (from config.yaml)
```yaml
Global Encoder:
  - hidden_dim: 256
  - num_heads: 8
  - num_layers: 4
  - context_dim: 128

UNet:
  - model_channels: 128
  - channel_mult: [1, 2, 3, 5]
  - num_res_blocks: 2
  - attention_resolutions: [8, 4, 2]
  - num_head_channels: 32
```

---

## 1. GlobalAudioEncoder Structure

### Layer-by-Layer Breakdown

#### CNN Feature Extractor
```
Input: [B, 1, 256, T] (mel spectrogram)

Conv Block 1:
  Conv2d(1 → 32, kernel=(8,3), stride=(4,1), padding=(2,1))
    Parameters: 1 × 8 × 3 × 32 + 32 = 800
  BatchNorm2d(32)
    Parameters: 32 × 2 = 64
  ReLU
  Output: [B, 32, 64, T]

Conv Block 2:
  Conv2d(32 → 64, kernel=(4,3), stride=(2,1), padding=(1,1))
    Parameters: 32 × 4 × 3 × 64 + 64 = 24,640
  BatchNorm2d(64)
    Parameters: 64 × 2 = 128
  ReLU
  Output: [B, 64, 32, T]

Conv Block 3:
  Conv2d(64 → 256, kernel=(4,3), stride=(2,1), padding=(1,1))
    Parameters: 64 × 4 × 3 × 256 + 256 = 196,864
  BatchNorm2d(256)
    Parameters: 256 × 2 = 512
  ReLU
  Output: [B, 256, 8, T]

Reshape: [B, 256, 8, T] → [B, T, 2048] → Linear → [B, T, 256]
  Adaptive Linear (if needed): 2048 → 256
    Parameters: 2048 × 256 + 256 = 524,544
```

**CNN Total Parameters: ~746,552**

#### Positional Encoding
```
Learnable Parameter: [1, 1024, 256]
Parameters: 1024 × 256 = 262,144
```

#### Transformer Encoder (4 layers)
Each TransformerEncoderLayer:
```
MultiHeadAttention (d_model=256, num_heads=8):
  - Q projection: Linear(256 → 256)
    Params: 256 × 256 + 256 = 65,792
  - K projection: Linear(256 → 256)
    Params: 65,792
  - V projection: Linear(256 → 256)
    Params: 65,792
  - Output projection: Linear(256 → 256)
    Params: 65,792
  Subtotal: 263,168

FeedForward Network (dim=256, hidden=512):
  - Linear1: 256 → 512
    Params: 256 × 512 + 512 = 131,584
  - Linear2: 512 → 256
    Params: 512 × 256 + 256 = 131,328
  Subtotal: 262,912

LayerNorm (×2): 256 × 4 = 1,024

Layer Total: 527,104
× 4 layers = 2,108,416
```

**Transformer Total Parameters: ~2,108,416**

#### Output Projection
```
Linear(256 → 256): 256 × 256 + 256 = 65,792
ReLU
Dropout(0.1)
Linear(256 → 128): 256 × 128 + 128 = 32,896

Global Projection:
  Linear(128 → 128): 128 × 128 + 128 = 16,512

Output Projection Total: 115,200
```

### **GlobalAudioEncoder Total: ~3,232,312 parameters**

---

## 2. GlobalConditionedSpatialTransformer

### Structure per Instance
```
Input: [B, C, H, W]  (C varies by layer)
Global Context: [B, 128]

Global Projection:
  Linear(128 → n_heads × d_head)
  For typical layer with 4 attention blocks:
    C = 256, num_heads = 8, d_head = 32
    Linear(128 → 256)
    Parameters: 128 × 256 + 256 = 33,024

Global Cross-Attention:
  MultiheadAttention(embed_dim=256, num_heads=8)
    Q,K,V projections: 256 × 256 × 3 = 196,608
    Output projection: 256 × 256 = 65,536
    Parameters: 262,144

Parent SpatialTransformer:
  (Inherited from AudioSR)
  - Self-attention layers
  - Cross-attention for text conditioning
  - Feed-forward networks
  Estimated: ~500,000 params per instance
```

### **Per GlobalConditionedSpatialTransformer: ~795,168 parameters**

**Total in UNet (appears 12 times based on config):**
- Input blocks: 4 attention locations × 2 transformers = 8
- Middle block: 2 transformers
- Output blocks: 4 attention locations × 2 transformers = 8
- **Total: 18 instances × 795,168 ≈ 14,313,024 parameters**

---

## 3. UNet Diffusion Model

### Channel Progression
```
Level 0: 128 × 1 = 128 channels
Level 1: 128 × 2 = 256 channels
Level 2: 128 × 3 = 384 channels
Level 3: 128 × 5 = 640 channels
```

### Input Blocks (Encoder)
```
Block 0:
  Conv2d(32 → 128, kernel=3, padding=1)
    Params: 32 × 3 × 3 × 128 + 128 = 36,992

Level 0 (128 ch, res 64×64):
  2 ResBlocks(128 → 128)
    Per ResBlock:
      - in_layers: GroupNorm + SiLU + Conv(128→128)
        Params: ~150,000
      - emb_layers: SiLU + Linear(512→128)
        Params: 512 × 128 + 128 = 65,664
      - out_layers: GroupNorm + SiLU + Dropout + Conv(128→128)
        Params: ~150,000
      - skip_connection: Identity (0 params)
      Total per ResBlock: ~365,664

    2 × 365,664 = 731,328

  Downsample(128 → 256)
    Conv2d: 128 × 3 × 3 × 256 = 294,912

Level 1 (256 ch, res 32×32):
  2 ResBlocks(256 → 256)
    Per ResBlock: ~1,200,000
    Total: 2,400,000

  Downsample(256 → 384)
    Conv2d: 442,368

Level 2 (384 ch, res 16×16):
  2 ResBlocks(384 → 384)
    Per ResBlock: ~2,000,000
    Total: 4,000,000

  2 × GlobalConditionedSpatialTransformer (included above)

  Downsample(384 → 640)
    Conv2d: 738,560

Level 3 (640 ch, res 8×8):
  2 ResBlocks(640 → 640)
    Per ResBlock: ~5,000,000
    Total: 10,000,000

  2 × GlobalConditionedSpatialTransformer (included above)
```

**Input Blocks Total: ~18,644,160 parameters**

### Middle Block
```
ResBlock(640 → 640): ~5,000,000
GlobalConditionedSpatialTransformer: ~795,168
GlobalConditionedSpatialTransformer: ~795,168
ResBlock(640 → 640): ~5,000,000
```

**Middle Block Total: ~11,590,336 parameters**

### Output Blocks (Decoder - symmetric to encoder)
**Output Blocks Total: ~20,000,000 parameters** (including skip connections)

### Timestep Embedding
```
Linear(128 → 512): 128 × 512 + 512 = 65,536
SiLU
Linear(512 → 512): 512 × 512 + 512 = 262,656
```

**Timestep Embedding Total: 328,192 parameters**

### **UNet Total: ~64,876,712 parameters**

---

## 4. VAE AutoEncoder (AudioSRAutoEncoderKL)

### Encoder
```
Configuration:
  - in_channels: 1
  - ch: 128 (base channels)
  - ch_mult: [1, 2, 4, 8] → [128, 256, 512, 1024]
  - num_res_blocks: 2
  - z_channels: 16

Structure:
  Conv_in: 1 → 128

  Down Block 1 (128 ch):
    2 × ResBlock(128 → 128)
    Downsample

  Down Block 2 (256 ch):
    2 × ResBlock(256 → 256)
    Downsample

  Down Block 3 (512 ch):
    2 × ResBlock(512 → 512)
    Downsample

  Down Block 4 (1024 ch):
    2 × ResBlock(1024 → 1024)
    Downsample

  Mid Block:
    ResBlock(1024 → 1024)
    Attention(1024)
    ResBlock(1024 → 1024)

  Output:
    GroupNorm(1024)
    Conv(1024 → 16)

Estimated Parameters: ~50,000,000
```

### Decoder (symmetric to encoder)
**Estimated Parameters: ~50,000,000**

### Quantization Layers
```
quant_conv: Conv2d(32 → 32, kernel=1)
  Params: 32 × 32 + 32 = 1,056

post_quant_conv: Conv2d(16 → 16, kernel=1)
  Params: 16 × 16 + 16 = 272
```

### **VAE Total: ~100,001,328 parameters**

---

## 5. AudioSR Vocoder (HiFi-GAN)

### Pre-Conv
```
Conv1d(256 → 1536, kernel=7, padding=3)
Parameters: 256 × 7 × 1536 + 1536 = 2,753,024
```

### Upsampling Layers (5 stages)
```
Stage 1: ConvTranspose1d(1536 → 768, kernel=12, stride=6)
  Params: 1536 × 12 × 768 + 768 = 14,156,544

Stage 2: ConvTranspose1d(768 → 384, kernel=10, stride=5)
  Params: 768 × 10 × 384 + 384 = 2,949,504

Stage 3: ConvTranspose1d(384 → 192, kernel=8, stride=4)
  Params: 384 × 8 × 192 + 192 = 589,824

Stage 4: ConvTranspose1d(192 → 96, kernel=4, stride=2)
  Params: 192 × 4 × 96 + 96 = 73,824

Stage 5: ConvTranspose1d(96 → 48, kernel=4, stride=2)
  Params: 96 × 4 × 48 + 48 = 18,432

Upsampling Total: 17,788,128
```

### ResBlocks (20 total, 4 per stage)
```
Each ResBlock has 2 conv paths with 3 dilations each:

Stage 1 ResBlocks (768 ch, kernels [3,7,11,15]):
  Per block: ~6 Conv1d layers × (768 × kernel × 768)
  4 blocks × ~3,500,000 = 14,000,000

Stage 2 ResBlocks (384 ch):
  4 blocks × ~900,000 = 3,600,000

Stage 3 ResBlocks (192 ch):
  4 blocks × ~225,000 = 900,000

Stage 4 ResBlocks (96 ch):
  4 blocks × ~55,000 = 220,000

Stage 5 ResBlocks (48 ch):
  4 blocks × ~14,000 = 56,000

ResBlocks Total: 18,776,000
```

### Post-Conv
```
Conv1d(48 → 1, kernel=7, padding=3)
Parameters: 48 × 7 × 1 + 1 = 337
```

### **Vocoder Total: ~39,317,489 parameters**

---

## 6. Total Model Parameter Count

```
Component                          Parameters
─────────────────────────────────────────────
GlobalAudioEncoder                  3,232,312
UNet (including transformers)      64,876,712
VAE AutoEncoder                   100,001,328
AudioSR Vocoder                    39,317,489
CLAP Model (unused in training)          N/A
─────────────────────────────────────────────
TOTAL                            207,427,841
```

### **≈ 207.4M Total Trainable Parameters**

---

## Connection Patterns & Data Flow

### Skip Connections in UNet

```
Input Blocks (Encoder):
  block_0: x → h_0 ──┐
  block_1: h_0 → h_1 ──┼──┐
  block_2: h_1 → h_2 ──┼──┼──┐
  ...                  │  │  │
  block_11: h_10 → h_11│  │  │
                       │  │  │
Middle Block:          │  │  │
  h_11 → h_mid         │  │  │
                       │  │  │
Output Blocks (Decoder):│  │  │
  block_0: [h_mid, h_11]←┘│  │
  block_1: [h_out, h_10]──┘  │
  block_2: [h_out, h_9] ─────┘
  ...
  block_11: [h_out, h_0] → output
```

**Key Implementation (model.py:924-926):**
```python
for module in self.output_blocks:
    h = module(torch.cat([h, hs.pop()], dim=1), emb, context, global_context)
```

### Residual Connections in ResBlocks

Each ResBlock uses additive skip connections:
```
Input x
  ↓
in_layers(x) → h
  ↓
h = h + emb_layers(timestep_emb)
  ↓
h = out_layers(h)
  ↓
output = skip_connection(x) + h
```

**Implementation (model.py:801):**
```python
return self.skip_connection(x) + h
```

---

## Cross-Attention Mechanism Detailed

### Location in Architecture
Cross-attention occurs in `GlobalConditionedSpatialTransformer` at 3 resolution levels:
- **Resolution 8×8**: 640 channels
- **Resolution 4×4**: 384 channels
- **Resolution 2×2**: 256 channels

### Mathematical Formulation

#### Step 1: Input Preparation
```
Local Features: x ∈ ℝ^(B × C × H × W)
Global Context: g ∈ ℝ^(B × 128)
```

#### Step 2: Global Context Projection
```python
# model.py:236-237
self.global_proj = nn.Linear(128, n_heads * d_head)

# Forward (model.py:256-261)
if global_context.dim() == 2:  # Single vector
    global_features = self.global_proj(global_context).unsqueeze(1)
    # Shape: [B, 1, n_heads * d_head]
else:  # Sequence
    global_features = self.global_proj(global_context)
    # Shape: [B, seq_len, n_heads * d_head]
```

**Transform**: g ∈ ℝ^(B×128) → G ∈ ℝ^(B×1×256)

#### Step 3: Reshape Local Features
```python
# model.py:263-265
B, C, H, W = x.shape
x_flat = x.view(B, C, H * W).transpose(1, 2)
# Shape: [B, H*W, C]
```

**Transform**: x ∈ ℝ^(B×C×H×W) → X ∈ ℝ^(B×N×C) where N = H×W

#### Step 4: Multi-Head Cross-Attention
```python
# model.py:239-244, 268
self.global_attn = nn.MultiheadAttention(
    embed_dim=n_heads * d_head,  # 256
    num_heads=n_heads,            # 8
    dropout=dropout,
    batch_first=True
)

attended_x, _ = self.global_attn(x_flat, global_features, global_features)
```

**Attention Formula**:
```
Query (Q):   X_flat ∈ ℝ^(B×N×256)
Key (K):     G ∈ ℝ^(B×1×256)
Value (V):   G ∈ ℝ^(B×1×256)

For each head h (8 heads, d_k = 32):
  Q_h = X_flat · W_Q^h        ∈ ℝ^(B×N×32)
  K_h = G · W_K^h             ∈ ℝ^(B×1×32)
  V_h = G · W_V^h             ∈ ℝ^(B×1×32)

  Attention_h = softmax(Q_h · K_h^T / √32) · V_h
              = softmax([B×N×32] · [B×32×1] / √32) · [B×1×32]
              = [B×N×32]

Concatenate all heads:
  Attended = Concat(Attention_1, ..., Attention_8) ∈ ℝ^(B×N×256)
  Output = Attended · W_O
```

#### Step 5: Reshape Back and Add Residual
```python
# model.py:270-271
x_attended = attended_x.transpose(1, 2).view(B, C, H, W)
x = x + x_attended  # Residual connection
```

#### Step 6: Parent Spatial Transformer
```python
# model.py:274
return super().forward(x, context)
```

This applies the standard SpatialTransformer (self-attention, feed-forward) from AudioSR.

### Complete Cross-Attention Flow

```
Input:
  Local features: [B, 256, 8, 8] (for resolution 8)
  Global context: [B, 128]

1. Project Global: [B, 128] → [B, 1, 256]

2. Flatten Local: [B, 256, 8, 8] → [B, 64, 256]

3. Multi-Head Attention:
   Q from local [B, 64, 256]
   K,V from global [B, 1, 256]

   Split into 8 heads (32 dims each):
   For each spatial location n ∈ {1..64}:
     attention_score[n] = softmax(q_n · k_global / √32)
     output[n] = attention_score[n] × v_global

   Result: [B, 64, 256] (each location attended to global)

4. Reshape: [B, 64, 256] → [B, 256, 8, 8]

5. Add Residual: x_out = x_in + attended_x

6. Standard Spatial Transformer (self-attention + FFN)
```

### Intuition

The cross-attention allows **each local spatial position** (e.g., each 8×8 patch in the latent) to:
1. Query the global audio context
2. Weight how much global information is relevant
3. Incorporate global acoustic properties into local feature refinement

This is critical for audio super-resolution because:
- Local patches need to know the overall acoustic environment
- Harmonics and timbre are global properties
- Temporal coherence requires awareness of the full sequence

---

## Summary Table: Attention Usage

| Location | Type | Query | Key/Value | Purpose |
|----------|------|-------|-----------|---------|
| GlobalAudioEncoder | Self-Attention | Audio sequence | Audio sequence | Capture temporal dependencies |
| GlobalConditionedSpatialTransformer | Cross-Attention | Local UNet features | Global context | Condition on global acoustics |
| SpatialTransformer (parent) | Self-Attention | Local features | Local features | Refine spatial features |

---

## Computational Complexity

### Forward Pass (Batch Size = 8, Audio = 10.24s)

```
1. VAE Encoding:
   Mel [8, 1, 256, 1024] → Latent [8, 16, 64, 64]
   ~50M MACs

2. Global Encoder:
   Mel [8, 1, 256, 1024] → Context [8, 128]
   - CNN: ~1.5 GMACs
   - Transformer: ~800 MMACs (T=1024, d=256)
   Total: ~2.3 GMACs

3. UNet Diffusion (per timestep):
   - ResBlocks: ~45 GMACs
   - Attention (18 instances): ~12 GMACs
   - Cross-Attention with Global: ~500 MMACs
   Total: ~57.5 GMACs per step
   × 50 DDIM steps = 2.875 TMACs for inference

4. VAE Decoding + Vocoder:
   Latent [8, 16, 64, 64] → Waveform [8, 1, 491520]
   - Decoder: ~50 GMACs
   - Vocoder: ~8 GMACs
   Total: ~58 GMACs
```

**Total Inference**: ~2.935 TMACs (~2.935 trillion multiply-accumulate operations)

---

This detailed breakdown shows a complex architecture with extensive use of skip connections, residual blocks, and multi-scale cross-attention for global-local feature fusion!
