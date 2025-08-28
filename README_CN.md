# ProjectLily Z IV - 音频超分辨率系统

## 🎵 概述

ProjectLily Z IV 是一个基于 AudioSR 架构的先进音频超分辨率系统，旨在通过深度学习技术增强音频质量。该项目实现了使用潜在扩散模型进行音频增强的完整管道，支持训练和推理。

## ✨ 特性

- **音频超分辨率**：使用最先进的扩散模型增强音频质量
- **图形用户界面**：为训练和推理提供用户友好的界面
- **灵活的管道**：支持各种音频格式和配置
- **实时处理**：高效的分块处理，适用于大型音频文件
- **先进架构**：完整的 AudioSR 实现，包含 VAE 和扩散组件

## 🏗️ 系统架构

系统由几个关键组件组成：

- **潜在扩散模型**：在潜在空间中使用扩散进行核心音频增强
- **AudioSR 自动编码器**：基于 VAE 的编码器/解码器，用于梅尔频谱图处理
- **声码器**：从增强频谱图进行高质量音频合成
- **CLAP 集成**：音频-文本表示学习能力

## 📁 项目结构

```
ProjectLily_Z_IV/
├── config.yaml              # 主配置文件
├── model.py                 # 核心模型实现
├── inference.py             # 推理引擎和GUI
├── train.py                 # 训练脚本
├── train_gui.py             # 训练GUI应用程序
├── run_gui.py               # 主GUI启动器
├── dataset_loader.py        # 数据加载工具
├── preprocess_audio.py      # 音频预处理工具
├── inspector.py             # 模型检查工具
├── dealer.py                # 数据处理工具
├── package.py               # 模型打包工具
├── convert_trained_model.py # 模型转换工具
├── SOLUTION_SUMMARY.md      # 详细技术文档
└── codebase_prompt.md       # 开发上下文
```

## 🚀 快速开始

### 系统要求

- Python 3.8+
- 支持CUDA的GPU（推荐）
- 支持CUDA的PyTorch
- 必需依赖项（见安装部分）

### 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd ProjectLily_Z_IV
```

2. 安装依赖项：
```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch torchaudio
pip install numpy scipy matplotlib
pip install soundfile librosa rich plotext
pip install transformers pyyaml tqdm
```

### 运行应用程序

#### 方法1：推理GUI
```bash
python inference.py
```

#### 方法2：训练GUI
```bash
python train_gui.py
```

#### 方法3：主GUI启动器
```bash
python run_gui.py
```

#### 方法4：命令行训练
```bash
python train.py --config config.yaml
```

## ⚙️ 配置

系统通过 `config.yaml` 进行配置：

### 主要配置部分

- **data**：数据集路径、音频参数、预处理设置
- **model**：模型架构、损失函数、扩散参数
- **train**：训练超参数、优化设置
- **inference**：推理设置、块处理、输出参数

### 配置示例
```yaml
data:
  sample_rate: 48000        # 采样率
  n_mels: 256              # 梅尔频谱图频率箱数
  segment_seconds: 10.24   # 片段长度
  
model:
  params:
    base_learning_rate: 0.0001  # 基础学习率
    timesteps: 1000             # 扩散时间步
    
train:
  batch_size: 1            # 批大小
  epochs: 500              # 训练轮数
  lr: 0.0001              # 学习率
  
inference:
  chunk_seconds: 4.0       # 处理块大小
  sample_steps: 50         # 采样步数
  guidance_scale: 1.5      # 引导强度
```

## 📊 数据准备

### 目录结构
```
data/
├── train/
│   ├── high/          # 高质量音频文件
│   └── low/           # 低质量音频文件（可选）
└── valid/
    ├── high/          # 验证高质量文件
    └── low/           # 验证低质量文件（可选）
```

### 支持格式
- WAV 文件（推荐）
- MP3、FLAC、M4A（需要相应编解码器）
- 采样率：48kHz（可配置）

### 数据要求
- **格式**：WAV 推荐（16位或24位）
- **采样率**：48kHz（在config.yaml中可配置）
- **时长**：每个文件10秒以上
- **质量**：清晰、高质量的源材料
- **数量**：训练100+文件，验证10+文件

## 🎯 使用指南

### 训练模型

1. **准备数据**：将高质量音频文件放入 `data/train/high/`
2. **配置**：根据数据集调整 `config.yaml`
3. **开始训练**：使用GUI或命令行
4. **监控**：通过丰富的控制台输出或GUI跟踪进度

### 运行推理

1. **加载模型**：选择训练好的检查点文件
2. **输入音频**：选择要增强的音频文件
3. **配置**：设置推理参数（可选）
4. **处理**：运行增强并保存结果

### GUI功能

#### 推理GUI
- 输入音频和模型检查点的文件选择
- 实时参数调整
- 进度跟踪和可视化
- 输出音频预览和保存

#### 训练GUI
- 数据集管理和验证
- 训练进度可视化
- 损失跟踪和模型检查点
- 资源监控

## 🔧 高级功能

### 模型组件

#### AudioSR 自动编码器
- 具有16通道潜在空间的梅尔频谱图VAE
- 带残差块的可配置架构
- 用于扩散过程的高效编码/解码

#### 扩散模型
- DDIM采样以实现快速推理
- 可配置的时间步和噪声调度
- 带低通条件的条件生成

#### 声码器集成
- 高保真音频合成
- 多尺度判别器训练
- 实时处理能力

### 处理功能

- **基于块的处理**：高效处理长音频文件
- **动态归一化**：自动音频电平调整
- **质量控制**：静音音频检测和处理
- **格式灵活性**：多种输入/输出格式支持

## 🛠️ 开发说明

### 模型架构
系统实现了完整的AudioSR兼容架构：

- **151模块兼容性**：与AudioSR完美权重兼容
- **缺失模块解决**：所有识别的缺口都已解决
- **网络健壮性**：无需外部依赖的离线操作

### 关键改进
- 静音音频检测和归一化
- 完美兼容性的健壮权重加载
- 带实时反馈的增强GUI
- 全面的错误处理和日志记录

## 📈 性能表现

### 系统要求
- **最低**：GTX 1060 6GB，16GB RAM
- **推荐**：RTX 3080+ 10GB，32GB RAM
- **存储**：模型和数据10GB+

### 处理速度
- **GPU**：RTX 3080上约10倍实时速度
- **CPU**：约0.1倍实时速度（不推荐）
- **批处理**：根据可用VRAM可扩展

## 🔍 故障排除

### 常见问题

#### 输出静音
- **原因**：输入音频很安静
- **解决方案**：已实现自动归一化
- **手动检查**：检查输入音频电平

#### 内存错误
- **原因**：GPU内存不足
- **解决方案**：减少批大小或使用CPU
- **替代方案**：用更小的块处理

#### 模型加载问题
- **原因**：检查点格式不兼容
- **解决方案**：使用 `convert_trained_model.py`
- **验证**：检查模型架构兼容性

### 调试工具
- `inspector.py`：模型架构分析
- 所有GUI应用程序中的详细日志
- 内置错误报告和恢复

## 📚 技术文档

详细技术信息请参见：
- `SOLUTION_SUMMARY.md`：完整实现细节
- `OPERATION_GUIDE.md`：详细操作指南
- `codebase_prompt.md`：开发上下文和决策
- 所有模块中的内联代码文档

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature-name`
3. 提交更改：`git commit -am 'Add feature'`
4. 推送到分支：`git push origin feature-name`
5. 创建 Pull Request

## 📝 许可证

该项目根据MIT许可证获得许可 - 有关详细信息，请参见LICENSE文件。

## 🙏 致谢

- AudioSR团队的原始架构
- PyTorch社区的框架
- 扩散模型研究的贡献者

## 📧 支持

如有问题和疑问：
1. 检查故障排除部分
2. 查看技术文档
3. 在GitHub上创建issue
4. 联系维护者

---

**状态**：✅ 生产就绪  
**版本**：4.0  
**最后更新**：2024年