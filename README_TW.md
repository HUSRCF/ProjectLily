# ProjectLily Z IV - 音訊超解析度系統

## 🎵 概述

ProjectLily Z IV 是一個基於 AudioSR 架構的先進音訊超解析度系統，旨在透過深度學習技術增強音訊品質。該專案實現了使用潛在擴散模型進行音訊增強的完整管線，支援訓練和推論。

## ✨ 特性

- **音訊超解析度**：使用最先進的擴散模型增強音訊品質
- **圖形使用者介面**：為訓練和推論提供使用者友善的介面
- **靈活的管線**：支援各種音訊格式和配置
- **即時處理**：高效的分塊處理，適用於大型音訊檔案
- **先進架構**：完整的 AudioSR 實現，包含 VAE 和擴散元件

## 🏗️ 系統架構

系統由幾個關鍵元件組成：

- **潛在擴散模型**：在潛在空間中使用擴散進行核心音訊增強
- **AudioSR 自動編碼器**：基於 VAE 的編碼器/解碼器，用於梅爾頻譜圖處理
- **聲碼器**：從增強頻譜圖進行高品質音訊合成
- **CLAP 整合**：音訊-文字表示學習能力

## 📁 專案結構

```
ProjectLily_Z_IV/
├── config.yaml              # 主配置檔案
├── model.py                 # 核心模型實現
├── inference.py             # 推論引擎和GUI
├── train.py                 # 訓練腳本
├── train_gui.py             # 訓練GUI應用程式
├── run_gui.py               # 主GUI啟動器
├── dataset_loader.py        # 資料載入工具
├── preprocess_audio.py      # 音訊預處理工具
├── inspector.py             # 模型檢查工具
├── dealer.py                # 資料處理工具
├── package.py               # 模型封裝工具
├── convert_trained_model.py # 模型轉換工具
├── SOLUTION_SUMMARY.md      # 詳細技術文件
└── codebase_prompt.md       # 開發上下文
```

## 🚀 快速開始

### 系統需求

- Python 3.8+
- 支援CUDA的GPU（建議）
- 支援CUDA的PyTorch
- 必需相依套件（見安裝部分）

### 安裝

1. 複製儲存庫：
```bash
git clone <repository-url>
cd ProjectLily_Z_IV
```

2. 安裝相依套件：
```bash
pip install -r requirements.txt
```

或手動安裝：
```bash
pip install torch torchaudio
pip install numpy scipy matplotlib
pip install soundfile librosa rich plotext
pip install transformers pyyaml tqdm
```

### 執行應用程式

#### 方法1：推論GUI
```bash
python inference.py
```

#### 方法2：訓練GUI
```bash
python train_gui.py
```

#### 方法3：主GUI啟動器
```bash
python run_gui.py
```

#### 方法4：命令列訓練
```bash
python train.py --config config.yaml
```

## ⚙️ 配置

系統透過 `config.yaml` 進行配置：

### 主要配置部分

- **data**：資料集路徑、音訊參數、預處理設定
- **model**：模型架構、損失函數、擴散參數
- **train**：訓練超參數、最佳化設定
- **inference**：推論設定、塊處理、輸出參數

### 配置範例
```yaml
data:
  sample_rate: 48000        # 取樣率
  n_mels: 256              # 梅爾頻譜圖頻率箱數
  segment_seconds: 10.24   # 片段長度
  
model:
  params:
    base_learning_rate: 0.0001  # 基礎學習率
    timesteps: 1000             # 擴散時間步
    
train:
  batch_size: 1            # 批次大小
  epochs: 500              # 訓練輪數
  lr: 0.0001              # 學習率
  
inference:
  chunk_seconds: 4.0       # 處理塊大小
  sample_steps: 50         # 取樣步數
  guidance_scale: 1.5      # 引導強度
```

## 📊 資料準備

### 目錄結構
```
data/
├── train/
│   ├── high/          # 高品質音訊檔案
│   └── low/           # 低品質音訊檔案（可選）
└── valid/
    ├── high/          # 驗證高品質檔案
    └── low/           # 驗證低品質檔案（可選）
```

### 支援格式
- WAV 檔案（建議）
- MP3、FLAC、M4A（需要相應編解碼器）
- 取樣率：48kHz（可配置）

### 資料需求
- **格式**：WAV 建議（16位元或24位元）
- **取樣率**：48kHz（在config.yaml中可配置）
- **時長**：每個檔案10秒以上
- **品質**：清晰、高品質的來源材料
- **數量**：訓練100+檔案，驗證10+檔案

## 🎯 使用指南

### 訓練模型

1. **準備資料**：將高品質音訊檔案放入 `data/train/high/`
2. **配置**：根據資料集調整 `config.yaml`
3. **開始訓練**：使用GUI或命令列
4. **監控**：透過豐富的控制台輸出或GUI追蹤進度

### 執行推論

1. **載入模型**：選擇訓練好的檢查點檔案
2. **輸入音訊**：選擇要增強的音訊檔案
3. **配置**：設定推論參數（可選）
4. **處理**：執行增強並儲存結果

### GUI功能

#### 推論GUI
- 輸入音訊和模型檢查點的檔案選擇
- 即時參數調整
- 進度追蹤和視覺化
- 輸出音訊預覽和儲存

#### 訓練GUI
- 資料集管理和驗證
- 訓練進度視覺化
- 損失追蹤和模型檢查點
- 資源監控

## 🔧 進階功能

### 模型元件

#### AudioSR 自動編碼器
- 具有16通道潛在空間的梅爾頻譜圖VAE
- 帶殘差塊的可配置架構
- 用於擴散過程的高效編碼/解碼

#### 擴散模型
- DDIM取樣以實現快速推論
- 可配置的時間步和雜訊排程
- 帶低通條件的條件生成

#### 聲碼器整合
- 高保真音訊合成
- 多尺度判別器訓練
- 即時處理能力

### 處理功能

- **基於塊的處理**：高效處理長音訊檔案
- **動態正規化**：自動音訊電平調整
- **品質控制**：靜音音訊偵測和處理
- **格式靈活性**：多種輸入/輸出格式支援

## 🛠️ 開發說明

### 模型架構
系統實現了完整的AudioSR相容架構：

- **151模組相容性**：與AudioSR完美權重相容
- **缺失模組解決**：所有識別的缺口都已解決
- **網路健全性**：無需外部相依性的離線操作

### 關鍵改進
- 靜音音訊偵測和正規化
- 完美相容性的健全權重載入
- 帶即時回饋的增強GUI
- 全面的錯誤處理和記錄

## 📈 效能表現

### 系統需求
- **最低**：GTX 1060 6GB，16GB RAM
- **建議**：RTX 3080+ 10GB，32GB RAM
- **儲存**：模型和資料10GB+

### 處理速度
- **GPU**：RTX 3080上約10倍即時速度
- **CPU**：約0.1倍即時速度（不建議）
- **批次處理**：根據可用VRAM可擴展

## 🔍 故障排除

### 常見問題

#### 輸出靜音
- **原因**：輸入音訊很安靜
- **解決方案**：已實現自動正規化
- **手動檢查**：檢查輸入音訊電平

#### 記憶體錯誤
- **原因**：GPU記憶體不足
- **解決方案**：減少批次大小或使用CPU
- **替代方案**：用更小的塊處理

#### 模型載入問題
- **原因**：檢查點格式不相容
- **解決方案**：使用 `convert_trained_model.py`
- **驗證**：檢查模型架構相容性

### 除錯工具
- `inspector.py`：模型架構分析
- 所有GUI應用程式中的詳細記錄
- 內建錯誤回報和恢復

## 📚 技術文件

詳細技術資訊請參見：
- `SOLUTION_SUMMARY.md`：完整實現細節
- `OPERATION_GUIDE.md`：詳細操作指南
- `codebase_prompt.md`：開發上下文和決策
- 所有模組中的內嵌程式碼文件

## 🤝 貢獻

1. Fork 儲存庫
2. 建立功能分支：`git checkout -b feature-name`
3. 提交變更：`git commit -am 'Add feature'`
4. 推送到分支：`git push origin feature-name`
5. 建立 Pull Request

## 📝 授權

該專案根據MIT授權獲得授權 - 有關詳細資訊，請參見LICENSE檔案。

## 🙏 致謝

- AudioSR團隊的原始架構
- PyTorch社群的框架
- 擴散模型研究的貢獻者

## 📧 支援

如有問題和疑問：
1. 檢查故障排除部分
2. 檢視技術文件
3. 在GitHub上建立issue
4. 聯絡維護者

---

**狀態**：✅ 生產就緒  
**版本**：4.0  
**最後更新**：2024年