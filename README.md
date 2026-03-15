# Underwater Acoustic Target Recognition

基于深度学习的水下声学目标识别（Underwater Acoustic Target Recognition）模型库。本项目主要针对 **ShipsEar** 等水下音频数据集，提供从数据预处理、特征提取（如Log-Mel谱图）、模型训练到结果分析的完整代码流水线。项目核心框架基于 **PyTorch** 和 **PyTorch Lightning** 构建。

## 📂 项目结构

```text
underwater-model/
├── Datasets/                           # 数据集处理模块
│   ├── ShipsEar_Data_Preprocessing.py  # ShipsEar 数据预处理（降噪、分帧等）
│   ├── ShipsEar_dataloader.py          # PyTorch DataLoader 定义
│   └── README.md                       # 数据集相关详细说明
├── src/
│   └── models/
│       └── custom_model.py             # 核心深度学习模型架构定义（如 HTAN 等）
├── Utils/                              # 工具函数与核心组件
│   ├── Feature_Extraction_Layer.py     # 特征提取层
│   ├── LitModel.py                     # PyTorch Lightning 训练逻辑封装 (Trainer)
│   ├── LogMelFilterBank.py             # Log-Mel 滤波器组计算实现
│   └── Network_functions.py            # 网络构建相关的辅助函数
├── shipsEar_AUDIOS/                    # 原始音频存放及处理目录
│   └── auto_label.py                   # 音频自动打标签脚本
├── Demo_Parameters.py                  # 全局超参数与配置文件
├── demo_light.py                       # 单次模型训练与测试的启动入口
├── run_experiments.sh                  # 批量执行实验/消融实验的自动化 Shell 脚本
├── feature_similarity_analysis.py      # 特征相似度与聚类分析工具
├── plot_curves.py                      # 训练损失、准确率曲线与结果可视化工具
├── shipsear_data_split.json            # 数据集划分配置 (Train/Val/Test)
├── split_audit_report.json             # 数据集划分审计报告
├── split_indices.txt                   # 数据集切分索引记录
├── htan_ablations_results.csv          # 消融实验结果自动记录文件
└── requirements.txt                    # Python 环境依赖包列表
```

## 🛠️ 环境依赖

请确保您的计算机上已安装 Python 3.8 或更高版本。建议使用虚拟环境（如 Conda 或 venv）。

使用以下命令安装所需的 Python 依赖包：

```bash
git clone [https://github.com/your-username/underwater-model.git](https://github.com/your-username/underwater-model.git)
cd underwater-model
pip install -r requirements.txt
```

*主要依赖包括：`torch`, `pytorch-lightning`, `librosa`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` 等。*

## 🚀 快速开始

### 1. 数据准备
1. 下载 **ShipsEar** 数据集，并将原始音频文件（`.wav`）放入 `shipsEar_AUDIOS/` 目录下。
2. 运行标签生成脚本，为音频文件生成对应的类别标签：
   ```bash
   cd shipsEar_AUDIOS
   python auto_label.py
   cd ..
   ```
3. 运行数据预处理脚本进行离线特征提取或数据清理（视具体需求而定）：
   ```bash
   python Datasets/ShipsEar_Data_Preprocessing.py
   ```

### 2. 参数配置
在开始训练之前，您可以通过修改 `Demo_Parameters.py` 文件来调整全局超参数：
* **数据参数**：采样率 (Sample Rate)、帧长 (Frame Length)、跳步 (Hop Length)
* **训练参数**：批次大小 (`batch_size`)、学习率 (`learning_rate`)、最大训练轮数 (`max_epochs`)
* **模型参数**：网络结构的具体维度和深度

### 3. 模型训练与评估
项目使用 PyTorch Lightning 封装了标准的训练、验证和测试流程。直接运行 `demo_light.py` 即可启动训练：

```bash
python demo_light.py
```
*说明：训练过程中会自动在终端输出进度条，并在每轮结束后验证准确率。最优模型权重会自动保存，训练结束后将在测试集上进行最终评估。*

### 4. 批量执行实验 (消融实验)
如果您需要测试不同的参数组合或运行消融实验（Ablation Study），可以使用提供的 Shell 脚本自动化运行：

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```
*实验的输出结果与评估指标将自动追加保存至 `htan_ablations_results.csv` 文件中，方便后续对比分析。*

## 📊 结果分析与可视化

训练或消融实验完成后，您可以使用内置的分析脚本对模型性能进行深度剖析：

* **绘制训练曲线**（Loss 和 Accuracy）：
  ```bash
  python plot_curves.py
  ```
* **特征相似度与特征空间分析**（如 t-SNE 降维可视化）：
  ```bash
  python feature_similarity_analysis.py
  ```

## 📌 数据集划分说明

为了保证实验的可重复性，数据集的划分被固定并记录在以下文件中：
* `shipsear_data_split.json`: 记录具体的划分配置和路径映射。
* `split_indices.txt`: 具体的样本索引。
* `split_audit_report.json`: 数据集划分的分布审计，确保训练/验证/测试集中各类别的均衡性。
