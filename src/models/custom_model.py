import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------
# 1. 严谨的多尺度 CNN (Multi-Scale Feature Extractor)
# ------------------------------------------------------------
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 分支1：局部纹理 (Local Branch) - 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        # 分支2：时序延展 (Temporal Branch) - 1x7 
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        # 分支3：频率延展 (Spectral Branch) - 7x1 
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        # 分支4：全局上下文 (Context Branch)
        self.branch4_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        
        self.fuse_pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        x4 = self.branch4_pool(x)
        x4 = x4.expand(-1, -1, x.shape[2], x.shape[3]) 
        
        out = torch.cat([x1, x2, x3, x4], dim=1) 
        return self.fuse_pool(out)

# ------------------------------------------------------------
# 2. 水声语义知识图谱网络 (Acoustic Knowledge-Guided GCN)
# ------------------------------------------------------------
class AcousticKnowledgeGCN(nn.Module):
    def __init__(self, in_channels, num_freq_bins, sr=16000, fmin=0, fmax=8000):
        super().__init__()
        self.num_freq_bins = num_freq_bins  
        self.in_channels = in_channels
        self.use_prior_mask = True # 默认开启，由外部 HTAN 动态控制
        
        # 1. 实体特征投影 (Q, K, V)
        self.query = nn.Linear(in_channels, in_channels // 2)
        self.key = nn.Linear(in_channels, in_channels // 2)
        self.value = nn.Linear(in_channels, in_channels) # 新增 Value 投影
        
        self.gcn_weight = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.3)

        # ==================== 核心创新点 1：水声知识实体嵌入 ====================
        # 为每一个物理频带分配一个可学习的全局语义向量，赋予图节点先验记忆
        self.entity_embedding = nn.Parameter(torch.randn(1, num_freq_bins, in_channels))
        
        # ==================== 核心创新点 2：可学习的专家知识图谱 ====================
        # 用物理公式算出来的矩阵作为参数初始化，让网络在微调时可以自动适应真实海洋信道
        expert_prior = self._build_expert_knowledge_prior(sr, fmin, fmax)
        self.knowledge_adj = nn.Parameter(expert_prior.clone().float())
        
        # ==================== 核心创新点 3：知识与数据融合门控 ====================
        # alpha 决定了在构图时，模型有多“相信”专家图谱
        self.alpha = nn.Parameter(torch.tensor([0.5])) 

    def _build_expert_knowledge_prior(self, sr, fmin, fmax, tol=0.2):
        """物理专家知识：计算梅尔频带中心频率的谐波倍频矩阵"""
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mels = np.linspace(mel_min, mel_max, self.num_freq_bins)
        center_freqs = 700 * (10**(mels / 2595) - 1) 
        
        A = torch.zeros(self.num_freq_bins, self.num_freq_bins)
        for i in range(self.num_freq_bins):
            for j in range(self.num_freq_bins):
                if i == j:
                    A[i, j] = 1.0
                    continue
                
                # 局部频带连续性偏置
                if abs(i - j) == 1:
                    A[i, j] = max(A[i, j].item(), 0.3)
                    A[j, i] = max(A[j, i].item(), 0.3)
                
                # 谐波共振偏置
                ratio = center_freqs[j] / (center_freqs[i] + 1e-8)
                for k in [2.0, 3.0, 4.0]:
                    if abs(ratio - k) < tol:
                        A[i, j] = max(A[i, j].item(), np.exp(- ((ratio - k)**2) / (0.1**2)))
                        A[j, i] = A[i, j]
        
        row_sum = A.sum(dim=1, keepdim=True)
        A = A / (row_sum + 1e-8)
        return A

    def forward(self, x):
        B_T, n_freqs, C = x.shape
        
        # --- 步骤 A：融入实体记忆 ---
        # x_embedded 既有当前帧的数据特征，又有该频带的全局统计特性
        x_embedded = x + self.entity_embedding
        
        Q = self.query(x_embedded)  
        K = self.key(x_embedded)
        V = self.value(x) # Value 保留原始数据高保真度
        
        # --- 步骤 B：生成数据驱动关系网 ---
        scale = K.shape[-1] ** 0.5
        A_data = torch.bmm(Q, K.transpose(1, 2)) / scale
        
        # --- 步骤 C：物理知识图谱掩码融合开关 ---
        if getattr(self, 'use_prior_mask', True):
            # 保证专家知识图谱权重非负，并广播到 Batch
            A_kg = F.relu(self.knowledge_adj)
            A_kg = A_kg.unsqueeze(0).expand(B_T, -1, -1)
            
            # 软性门控融合 (彻底替换原来的 -1e9 掩码阻断)
            A_fused = A_data + self.alpha * A_kg
        else:
            # 如果消融实验关掉掩码，就退化为纯自注意力网络
            A_fused = A_data
        
        A_dynamic = F.softmax(A_fused, dim=-1)
        A_dynamic = self.dropout(A_dynamic)

        # --- 步骤 D：图网络特征聚合 ---
        z = torch.bmm(A_dynamic, V)  
        z = self.gcn_weight(z)
        out = self.norm(x + F.relu(z))
        
        return out

# ------------------------------------------------------------
# 3. 关键帧时序注意力 (Temporal Attention Pooling)
# ------------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)  
        attn_weights = F.softmax(attn_weights, dim=1) 
        global_feat = torch.sum(x * attn_weights, dim=1) 
        return global_feat, attn_weights

# ------------------------------------------------------------
# 4. 最终主模型：HTAN (包含完善的消融开关逻辑)
# ------------------------------------------------------------
class HTAN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, base_channels=32, 
                 input_fdim=128, input_tdim=1024,
                 use_graph=True, use_prior_mask=True,
                 use_temporal_encoder=True, use_temporal_attention=True):
        super().__init__()
        
        # 记录消融开关状态
        self.use_graph = use_graph
        self.use_prior_mask = use_prior_mask
        self.use_temporal_encoder = use_temporal_encoder
        self.use_temporal_attention = use_temporal_attention
        
        self.frontend = nn.Sequential(
            MultiScaleConvBlock(in_channels, base_channels),      
            MultiScaleConvBlock(base_channels, base_channels*2), 
            MultiScaleConvBlock(base_channels*2, base_channels*4) 
        )
        
        # 动态推导输出尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(2, in_channels, input_fdim, input_tdim) 
            
            # 为了严谨，在过 dummy 前临时开启 eval 模式，之后再恢复
            self.frontend.eval()
            dummy_out = self.frontend(dummy_input)
            self.frontend.train()
            
            _, cnn_out_c, f_out, t_out = dummy_out.shape
            
        # ★ 替换为新的知识图谱 GCN，不影响外围传参逻辑
        self.harmonic_gcn = AcousticKnowledgeGCN(
            in_channels=cnn_out_c, 
            num_freq_bins=f_out,
            sr=16000, fmin=0, fmax=8000 
        )
        self.harmonic_gcn.use_prior_mask = self.use_prior_mask
        
        # 如果使用图网络，通道由于 mean+max 拼接会翻倍
        gru_input_size = cnn_out_c * 2 if self.use_graph else cnn_out_c
        
        self.temporal_encoder = nn.GRU(
            input_size=gru_input_size, 
            hidden_size=gru_input_size // 2, 
            num_layers=1,  
            batch_first=True, 
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttention(in_dim=gru_input_size)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_input_size),
            nn.Dropout(0.4),
            nn.Linear(gru_input_size, num_classes)
        )

    def forward(self, x, extract_feature=False):  
        x = self.frontend(x)  
        B, C, F_out, T_out = x.shape
        
        # --- 模块 1：图网络开关 ---
        if self.use_graph:
            x = x.permute(0, 3, 2, 1).contiguous().view(B * T_out, F_out, C)
            x = self.harmonic_gcn(x)
            
            x = x.view(B, T_out, F_out, C)
            x_mean = x.mean(dim=2)          
            x_max = x.max(dim=2).values     
            x = torch.cat([x_mean, x_max], dim=-1) # [B, T, 2*C]
        else:
            # 仅做全局频率平均 [B, T, C]
            x = x.mean(dim=2).transpose(1, 2) 
        
        # --- 模块 2：时序编码开关 ---
        if self.use_temporal_encoder:
            x, _ = self.temporal_encoder(x) 
            
        # --- 模块 3：帧级注意力开关 ---
        if self.use_temporal_attention:
            x, attn_weights = self.temporal_attention(x) 
        else:
            # 全局时间平均池化
            x = x.mean(dim=1)
        
        # 核心拦截点：如果是自监督预训练模式，提取出高维特征就直接跑路
        if extract_feature:
            return x
            
        # 如果是正常监督微调/测试模式，则正常走分类器
        logits = self.classifier(x)
        return logits