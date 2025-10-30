# 314113009_LAB2
EEG Classification with BCI competition dataset

# 📘 Lab2: EEG Classification with EEGNet & DeepConvNet

**完整的腦機介面（BCI）EEG 訊號分類專案**
**最後更新**: 2025-10-30  
**版本**: 1.0.0  
**作者**: 張涵崴 (wayne-714)
---

## 🎯 專案簡介

本專案實作了兩種用於腦電圖（EEG）訊號分類的深度學習模型：

- **EEGNet**: 一個輕量級的卷積神經網路，專為 BCI（腦機介面）設計，參數量僅 ~2,500
- **DeepConvNet**: 一個較深的卷積網路，參數量約 ~50,000，用於性能比較


### 🏆 最佳結果

- **EEGNet**: 83.80% 測試準確率（輕量、快速）
- **DeepConvNet**: 88.00% 測試準確率（高準確率）

---

## 💻 系統需求

### 硬體需求

| 項目 | 最低需求 | 建議配置 |
|------|---------|---------|
| **CPU** | Intel i5 或同等級 | Intel i7 或更高 |
| **RAM** | 8 GB | 16 GB 或更多 |
| **GPU** | 無（可用 CPU 訓練） | NVIDIA GPU（CUDA 支援） |
| **儲存空間** | 2 GB | 5 GB 或更多 |

### 軟體需求

- **作業系統**: Windows 10/11、macOS 10.15+、Linux（Ubuntu 20.04+）
- **Python**: 3.8 或以上版本
- **套件管理**: pip 或 conda

### GPU 加速（可選但強烈建議）

如果你有 NVIDIA GPU：
- **CUDA**: 11.8 或 12.1
- **cuDNN**: 對應版本
- 訓練速度可提升 **5-10 倍**

檢查 GPU 是否可用：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
---

## 📂 資料集準備

### 下載 BCI Competition Dataset

#### 資料格式

專案需要以下 4 個 `.npz` 檔案：

```
data/
├── S4b_train.npz      # 受試者 4 訓練資料
├── X11b_train.npz     # 受試者 11 訓練資料
├── S4b_test.npz       # 受試者 4 測試資料
└── X11b_test.npz      # 受試者 11 測試資料
```

1. 將下載的 `.npz` 檔案放入 `data/` 資料夾
2. 確認檔案路徑正確：
   ```
   lab2_EEG_classification/
   └── data/
       ├── S4b_train.npz
       ├── X11b_train.npz
       ├── S4b_test.npz
       └── X11b_test.npz
   ```

#### 驗證資料

執行資料檢查工具：
```bash
python inspect_data.py
```

你應該會看到：
```
================================================================================
Inspecting: ./data/S4b_train.npz
================================================================================

Available keys: ['signal', 'label']

--------------------------------------------------------------------------------
Key: 'signal'
--------------------------------------------------------------------------------
  Shape:    (540, 750, 2)
  Dtype:    float64
  Min:      ...
  Max:      ...
  ...
```

---

## 📁 專案結構

```
lab2_EEG_classification/
│
├── README.md                    # 📘 本說明文件
├── requirements.txt            # 📦 套件依賴清單
│
├── models/                     # 🧠 模型定義資料夾
│   └── EEGNet.py              # EEGNet 和 DeepConvNet 類別
│
├── data/                       # 💾 資料集資料夾
│   ├── S4b_train.npz          # 訓練資料 1
│   ├── X11b_train.npz         # 訓練資料 2
│   ├── S4b_test.npz           # 測試資料 1
│   └── X11b_test.npz          # 測試資料 2
│
├── dataloader.py               # 📥 資料載入與預處理
├── inspect_data.py             # 🔍 資料檢查工具
│
├── main.py                     # 🎯 主訓練腳本
├── run_experiments.py          # 🧪 批次實驗腳本
│
├── weights/                    # 💾 模型權重儲存（訓練後產生）
│   └── best.pt
│
└── results/                    # 📊 實驗結果（訓練後產生）
    ├── accuracy_curve.png
    ├── loss_curve.png
    └── confusion_matrix.png
```
---
下載專案

**方法一：使用 Git（推薦）**
```bash
git clone https://github.com/wayne-714/lab2_EEG_classification.git
cd lab2_EEG_classification
```

**方法二：直接下載 ZIP**
1. 點擊 GitHub 頁面的 "Code" → "Download ZIP"
2. 解壓縮到你的工作目錄
3. 進入資料夾：
   ```bash
   cd lab2_EEG_classification
   ```

---
安裝相依套件

**方法一：使用 requirements.txt（推薦）**

```bash
# 建立虛擬環境（可選但建議）
python -m venv venv

# 啟動虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安裝所有套件
pip install -r requirements.txt
```

**方法二：手動安裝個別套件**

```bash
# 深度學習框架
pip install torch>=2.0.0

# 數值計算
pip install numpy>=1.24.0

# 資料處理
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0

# 視覺化
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# 工具
pip install tqdm>=4.65.0
```

**安裝 PyTorch（根據你的系統）**：

- **有 CUDA GPU（推薦）**：
  ```bash
  # CUDA 11.8
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  # CUDA 12.1
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- **只用 CPU**：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- **macOS（Apple Silicon）**：
  ```bash
  pip install torch torchvision torchaudio
  ```

---

## 📖 程式碼說明

### 1. `models/EEGNet.py` - 模型定義

**用途**：定義 EEGNet 和 DeepConvNet 兩個深度學習模型。

**主要類別**：

#### `EEGNet` 類別
```python
class EEGNet(nn.Module):
    def __init__(self, activation='elu', dropout_rate=0.25, F1=16, D=2, F2=32, 
                 num_channels=2, num_classes=2, kernel_length=64)
```

**參數說明**：
- `activation`: 激活函數類型（'elu', 'relu', 'leakyrelu'）
- `dropout_rate`: Dropout 比率（0.0-1.0）
- `F1`: 時間濾波器數量（預設 16）
- `D`: 深度乘數（預設 2）
- `F2`: 逐點濾波器數量（預設 32）
- `num_channels`: EEG 通道數（預設 2）
- `num_classes`: 分類類別數（預設 2）
- `kernel_length`: 時間卷積核長度（預設 64）

**架構組成**：
1. **Block 1**: 時間卷積 - 提取時間特徵
2. **Block 2**: Depthwise 卷積 - 空間濾波
3. **Block 3**: Separable 卷積 - 參數減少的特徵提取
4. **Classifier**: 全連接分類層

#### `DeepConvNet` 類別
```python
class DeepConvNet(nn.Module):
    def __init__(self, activation='elu', dropout_rate=0.5, num_channels=2, num_classes=2)
```

**架構組成**：
- 4 個卷積區塊
- 每個區塊包含：卷積層 → BatchNorm → 激活 → MaxPooling → Dropout
- 最後接全連接層

**測試模型**：
```bash
python models/EEGNet.py
```

---

### 2. `dataloader.py` - 資料載入器

**用途**：載入和預處理 BCI Competition 資料集。

**主要函數**：

#### `read_bci_data()`
```python
def read_bci_data():
    """
    載入 BCI Competition 資料集
    
    Returns:
        train_data: (1080, 1, 2, 750) - 訓練資料
        train_label: (1080,) - 訓練標籤
        test_data: (1080, 1, 2, 750) - 測試資料
        test_label: (1080,) - 測試標籤
    """
```

**功能**：
1. 讀取 4 個 `.npz` 檔案
2. 合併訓練和測試資料（540+540=1080 樣本）
3. 標籤轉換：從 1-indexed (1, 2) 轉為 0-indexed (0, 1)
4. 維度重整：`(N, C, T)` → `(N, 1, C, T)` 符合模型輸入
5. 處理 NaN 值：用平均值填充

**使用範例**：
```python
from dataloader import read_bci_data

train_data, train_label, test_data, test_label = read_bci_data()
print(f"Training samples: {len(train_data)}")
# Output: Training samples: 1080
```

---

### 3. `inspect_data.py` - 資料檢查工具

**用途**：檢查和視覺化 EEG 資料集。

**主要功能**：

#### `inspect_npz_file(filepath)`
- 顯示 `.npz` 檔案的詳細資訊
- 統計數據：形狀、最小值、最大值、平均值、標準差
- NaN 值計數

#### `visualize_sample(filepath, sample_idx=0)`
- 視覺化單一 EEG 樣本
- 繪製各通道的時間序列波形
- 自動儲存為 `.png` 圖片

#### `visualize_multiple_samples(filepath, num_samples=3)`
- 同時視覺化多個樣本
- 方便比較不同樣本的波形模式

**執行方式**：
```bash
python inspect_data.py
```

**輸出**：
- 終端顯示統計資訊
- 生成 `sample_visualization_0.png`
- 生成 `multiple_samples_visualization.png`

---

### 4. `main.py` - 主訓練腳本

**用途**：訓練模型、評估性能、生成視覺化結果。

**主要類別/函數**：

#### `BCIDataset` 類別
```python
class BCIDataset(Dataset):
    """PyTorch Dataset 包裝器"""
```
- 將 NumPy 陣列轉為 PyTorch Dataset
- 支援 DataLoader 的批次載入

#### 訓練函數 `train()`
```python
def train(model, train_loader, test_loader, criterion, optimizer, args, device):
    """
    完整訓練循環
    
    功能：
    1. 訓練模型（多個 epoch）
    2. 每個 epoch 後在測試集評估
    3. 自動保存最佳模型
    4. 返回訓練歷史
    """
```

#### 視覺化函數
```python
plot_train_acc(...)      # 繪製準確率曲線
plot_train_loss(...)     # 繪製損失曲線
plot_confusion_matrix()  # 繪製混淆矩陣
```

**命令列參數**：

| 參數 | 預設值 | 說明 | 選項 |
|------|--------|------|------|
| `-num_epochs` | 300 | 訓練週期數 | 任意整數 |
| `-batch_size` | 64 | 批次大小 | 32, 64, 128... |
| `-lr` | 0.001 | 學習率 | 0.0001 ~ 0.01 |
| `-model` | eegnet | 模型類型 | eegnet, deepconvnet |
| `-activation` | elu | 激活函數 | elu, relu, leakyrelu |
| `-dropout` | 0.25 | Dropout 率 | 0.0 ~ 1.0 |

**使用範例**：
```bash
# 訓練 EEGNet（預設配置）
python main.py

# 訓練 DeepConvNet with ReLU
python main.py -model deepconvnet -activation relu -dropout 0.5

# 自訂超參數
python main.py -num_epochs 200 -batch_size 32 -lr 0.0005
```

---

### 5. `run_experiments.py` - 批次實驗腳本

**用途**：自動執行多組實驗，比較不同配置的性能。

**預設實驗配置**：

```python
experiments = [
    # EEGNet 實驗
    {"name": "Exp1: Baseline EEGNet", "model": "eegnet", "activation": "elu", "dropout": 0.25},
    {"name": "Exp2: EEGNet with ReLU", "model": "eegnet", "activation": "relu", "dropout": 0.25},
    {"name": "Exp3: EEGNet with LeakyReLU", "model": "eegnet", "activation": "leakyrelu", "dropout": 0.25},
    {"name": "Exp4: EEGNet dropout=0.1", "model": "eegnet", "activation": "elu", "dropout": 0.1},
    {"name": "Exp5: EEGNet dropout=0.5", "model": "eegnet", "activation": "elu", "dropout": 0.5},
    
    # DeepConvNet 實驗
    {"name": "Exp6: DeepConvNet baseline", "model": "deepconvnet", "activation": "elu", "dropout": 0.5},
    {"name": "Exp7: DeepConvNet dropout=0.3", "model": "deepconvnet", "activation": "elu", "dropout": 0.3},
    {"name": "Exp8: DeepConvNet with ReLU", "model": "deepconvnet", "activation": "relu", "dropout": 0.5},
]
```

**執行方式**：
```bash
python run_experiments.py
```

**預期輸出**：
```
================================================================================
Starting EEG Classification Batch Experiments
================================================================================

================================================================================
Running Experiment: Exp1: Baseline EEGNet
================================================================================
Model: eegnet
Activation: elu
Dropout: 0.25
...
✅ Exp1: Baseline EEGNet completed successfully in 20.5 minutes

[... 執行其他實驗 ...]

================================================================================
ALL EXPERIMENTS COMPLETED!
================================================================================
Total time: 3.5 hours

Results Summary:
  1. Exp1: Baseline EEGNet: ✅ Success
  2. Exp2: EEGNet with ReLU: ✅ Success
  ...
```

**自訂實驗**：

編輯 `run_experiments.py` 中的 `experiments` 列表，加入你想測試的配置。

---

## 🎮 使用教學

### 快速開始（5 分鐘體驗）

如果你只想快速測試模型，執行以下指令：

```bash
# 1. 確認環境
python check_installation.py

# 2. 檢查資料
python inspect_data.py

# 3. 快速訓練（10 個 epoch）
python main.py -num_epochs 10
```

訓練完成後，檢查 `results/` 資料夾中的圖片。

---

### 標準流程（完整實驗）

#### Step 1: 環境確認
```bash
python check_installation.py
```

#### Step 2: 資料檢查
```bash
python inspect_data.py
```
確認資料已正確載入且無錯誤。

#### Step 3: 訓練單一模型

**訓練 EEGNet（預設配置）**：
```bash
python main.py
```

**訓練 DeepConvNet**：
```bash
python main.py -model deepconvnet -activation elu -dropout 0.5 -num_epochs 300
```

**自訂配置**：
```bash
python main.py \
    -model eegnet \
    -activation relu \
    -dropout 0.3 \
    -lr 0.0005 \
    -batch_size 32 \
    -num_epochs 200
```

#### Step 4: 查看結果

訓練完成後，檢查以下檔案：

**模型權重**：
```
weights/best.pt
```

**視覺化結果**：
```
results/
├── accuracy_curve.png      # 訓練/測試準確率曲線
├── loss_curve.png          # 訓練損失曲線
└── confusion_matrix.png    # 混淆矩陣
```

**終端輸出**（範例）：
```
================================================================================
TRAINING COMPLETE
================================================================================
Best Test Accuracy: 83.80%

Classification Report:
              precision    recall  f1-score   support

     Class 0       0.84      0.85      0.84       540
     Class 1       0.85      0.84      0.84       540

    accuracy                           0.84      1080
```

---

### 進階用法：批次實驗

#### 執行所有預設實驗

```bash
python run_experiments.py
```

這會自動執行 8 組實驗，總耗時約 **3-4 小時**（視硬體而定）。

#### 修改實驗配置

編輯 `run_experiments.py`，自訂你想測試的配置：

```python
experiments = [
    {
        "name": "My Custom Experiment",
        "model": "eegnet",
        "activation": "elu",
        "dropout": 0.2,  # 自訂 dropout
        "lr": 0.001,
        "epochs": 150    # 較少的 epochs
    },
    # 加入更多實驗...
]
```
---

## 📊 實驗結果

#### EEGNet 實驗

| 配置 | 激活函數 | Dropout | 測試準確率 | F1-Score |
|------|---------|---------|-----------|----------|
| Exp1 | ELU | 0.25 | **83.80%** | 0.84 |
| Exp2 | ReLU | 0.25 | 82.10% | 0.82 |
| Exp3 | LeakyReLU | 0.25 | 81.90% | 0.82 |
| Exp4 | ELU | 0.1 | 80.50% | 0.81 |
| Exp5 | ELU | 0.5 | 82.40% | 0.82 |

**結論**：
- ✅ ELU 是最佳激活函數
- ✅ Dropout 0.25 是最佳正則化強度
- ⚠️ 過低 dropout (0.1) 導致過擬合

---

#### DeepConvNet 實驗

| 配置 | 激活函數 | Dropout | 測試準確率 | F1-Score |
|------|---------|---------|-----------|----------|
| Exp6 | ELU | 0.5 | 87.31% | 0.87-0.88 |
| Exp7 | ELU | 0.3 | 86.20% | 0.86 |
| Exp8 | ReLU | 0.5 | **88.00%** | 0.88-0.89 |

**結論**：
- 🏆 DeepConvNet + ReLU + Dropout 0.5 = 最佳配置
- ✅ DeepConvNet 優於 EEGNet 約 **4-5%**
- ✅ 高 Dropout (0.5) 對深層網路很重要

---

#### 模型比較

| 指標 | EEGNet | DeepConvNet | 差異 |
|------|--------|-------------|------|
| **測試準確率** | 83.80% | **88.00%** | +4.2% |
| **參數量** | ~2,500 | ~50,000 | 20× |
| **訓練速度** | 28 it/s | 15 it/s | 1.9× 慢 |
| **推論速度** | ~1 ms | ~2 ms | 2× 慢 |
| **記憶體** | ~10 KB | ~200 KB | 20× |
| **適用場景** | 即時 BCI | 離線研究 | - |
