# 水声目标识别：基于物理谐波图与度量学习的 AST 架构

本项目是一个针对水下声学目标识别（如 ShipsEar 数据集）的高阶深度学习框架。针对真实海洋环境底噪复杂、有效样本稀缺、长尾分布严重等痛点，本框架在音频声谱 Transformer（AST）的基础上，创新性地引入了**参数高效微调（PEFT-LoRA）**、**物理谐波图卷积网络（Physical-Harmonic GCN）**以及**联合有监督对比度量学习（Supervised Contrastive Learning）**，实现了对极其微弱水声物理信号的精准捕获与强泛化分类。

---

## 1. 数据处理与划分机制 (Data Processing & Splitting)

由于水声数据集（如 ShipsEar）存在极高的段落间相关性，常规的随机打乱切分会导致严重的“数据泄露（Data Leakage）”。本项目采用了极其严苛的数据处理与划分管线：

### 1.1 录音级别划分 (Recording-Level Split)
* **核心防泄露策略：** 摒弃传统的“切片后随机划分”，采用**录音文件级别（Recording-Level）**的绝对物理隔离。同一个 `.wav` 原始长录音切分出的所有音频片段，只能唯一存在于训练集、验证集或测试集中的某一个集合中。
* **划分概况（以 ShipsEar 为例）：** 包含约 89 个独立录音文件夹，按照零泄露原则划分（例如 Train: 61, Val: 11, Test: 17）。

### 1.2 物理声谱提取与强数据增强 (SpecAugment)
* **特征提取：** 将原始 1D 音频信号通过 Log-Mel Filterbank 转化为 2D 对数梅尔声谱图（如 `128 × 157` 维度），捕捉丰富且细腻的频域能量。
* **掩码增强 (Masking)：** 在训练阶段，框架强制引入基于 `torchaudio.transforms` 的 **SpecAugment**。
  * **频域掩码 (Frequency Masking)：** 随机遮蔽最高 24 个频带，模拟真实海洋环境中的宽带噪声掩蔽效应。
  * **时域掩码 (Time Masking)：** 随机遮蔽最高 30 个时间帧，模拟远距离传播中螺旋桨节拍信号的随机衰落丢失。
  * **目的：** 逼迫模型脱离对特定频段的“死记硬背”，必须依赖未被遮蔽的物理谐波和时序信息进行反向推导。

---

## 🕸️ 2. 核心构图流程 (Graph Construction: Physical-Harmonic GCN)

传统 Vision Transformer 仅将声谱图视为普通 2D 图像进行全局注意力计算，忽略了水声信号极其特殊的物理规律。本项目在 `ast_graph.py` 中设计了**物理谐波图卷积网络**，其构图流程如下：

### 2.1 节点定义 (Nodes)
* AST 模型将输入的 2D 声谱图在时间和频率维度上切分成网格（Patch Token）。
* 构图时，拦截并剥离全局的 `[CLS]` 和 `[DIST]` token，将剩余的 **180 个时频 Patch**（12个频段 × 15个时间步）作为图的 **180 个节点**。每个节点包含 `768` 维的高维特征。

### 2.2 物理先验连边 (Physical Prior Edges)
建立一个纯物理驱动的常量邻接矩阵 $A_{physical}$，包含两种绝对物理法则连边：
1. **时序连续性边 (Temporal Continuity Edges)：**
   * **物理意义：** 捕捉螺旋桨空化噪声产生的平滑多普勒节拍（Rhythm）。
   * **连接方式：** 在同一个频带 $f$ 下，强制将时间步 $t$ 与相邻的未来帧 $t+1$ 建立双向强连接。
2. **跨频谐波共振边 (Harmonic Resonance Edges)：**
   * **物理意义：** 捕捉舰船柴油机和发电机产生的低频线谱（LOFAR）的基频与倍频现象（如 50Hz 与 100Hz 的共振）。
   * **连接方式：** 跨越不同的频率层，在基频节点 $f$ 与近似倍频节点 $2f+1$ 之间建立强物理连边。即使基频被海洋底噪完全掩盖，也能通过连边从高次谐波中“借用”能量进行重构修复。

### 2.3 动态语义与异构图融合 (Heterogeneous Graph Fusion)
* **动态图计算：** 利用自注意力机制（Self-Attention）根据实时输入数据计算一个动态相似度邻接矩阵 $A_{dynamic}$，捕捉非线性的突发海洋噪声关联。
* **门控融合：** 通过可学习参数 $\alpha$ 将物理先验图与数据驱动动态图进行异构融合：
 $$A_{combined} = \alpha A_{physical} + (1-\alpha) A_{dynamic}$$。
 随后通过 GCN 进行特征聚合。

---

## 🧠 3. 采用算法介绍 (Core Algorithms)

### 3.1 基于 LoRA 的参数高效微调 (PEFT-LoRA)
* 引入了预训练于 AudioSet 与 ImageNet 的强大 AST 基座模型。
* 冻结（Freeze）了 Transformer 骨干网络 98% 的参数，仅在注意力模块的 `Query` 和 `Value` 投影层旁路注入秩为 `32` 的 **LoRA (Low-Rank Adaptation)** 模块。
* 极大地降低了显存开销，并有效防止了大模型在小样本水声数据集上的灾难性过拟合。

### 3.2 特征强制路由与防惰性机制 (Anti-Lazy Model Effect)
* 预训练大模型的原生全局 `[CLS]` 特征往往带有极强的“空气声学偏见”，极易导致特征坍塌。
* 本算法**彻底切断了传统的全局特征捷径**，强制要求所有分类决策必须基于经过 `PhysicalHarmonicGCN` 聚合推导后的节点均值特征，逼迫网络彻底重建水下声学物理空间。

### 3.3 联合有监督对比度量学习 (Joint Supervised Contrastive Learning)
为彻底解决水声目标类别极端不平衡以及小样本导致的泛化崩溃问题，废弃了单纯的交叉熵分类范式，引入了**度量元学习思想**：
* **特征降维与投影：** 模型剥离了传统的一体化分类器，转而输出高维（768维）的物理声学指纹。
* **Supervised Contrastive Loss (SupCon)：** 在高维度量空间中施加严苛惩罚——强制同一类别（同型船只）在当前 Batch 里的所有特征向量相互靠拢聚类，并将不同类别的向量狠狠推远。
* **联合优化：** 采用 `Loss = CrossEntropy + 0.5 * SupConLoss` 进行联合多任务优化。
* **黄金评价指标：** 全面弃用易受长尾分布欺骗的 Accuracy，将模型保存和性能评估的核心锚点全部替换为 **AUPRC (Area Under the Precision-Recall Curve)**，此外**废除极小验证集导致的早停陷阱**，强制获取完整聚类后的最终模型权重，确保对罕见目标的真正识别能力。

---

## 4. 环境依赖与快速启动 (Installation & Usage)

### 4.1 环境依赖
请确保安装以下核心库（推荐 Python 3.9+）：
```bash
pip install torch torchvision torchaudio
pip install lightning torchmetrics
pip install timm==0.4.5
```

### 4.2 一键启动训练
本项目基于 PyTorch Lightning 构建，支持单机单卡/多卡自动调度。运行以下命令以启动具有物理图和度量学习的完全体模型：

```bash
python demo_light.py \
    --data_selection 1 \
    --train_mode lora \
    --train_batch_size 32 \
    --lora_rank 32
```

### 4.3 日志与数据分析
* **强制完全体输出：** 训练过程默认无视验证集早停波动，强制执行 50 个 Epoch 的深度对比聚类，并在测试集上调用最后一轮（Last Checkpoint）的完美权重。
* **科研绘图支持：** 系统同时配置了 `TensorBoardLogger` 和 `CSVLogger`。运行结束后，可直接前往 `tb_logs/.../csv_logs/` 目录提取 `metrics.csv`，极大地简化了导入 Origin 或 MATLAB 绘制高清晰度 Loss/AUPRC 收敛曲线的流程。
