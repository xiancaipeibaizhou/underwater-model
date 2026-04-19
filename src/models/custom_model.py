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
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
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
# 🌟 2. 新增：时序熵驱动并行门控 (Sequence-Driven Gating)
# ------------------------------------------------------------
class SNRAwareGating(nn.Module):
    """
    致敬 AEGIS 的熵驱动融合思想：
    全局感知当前样本的信噪比/混乱度，动态分配物理图谱(A_kg)的话语权 alpha
    """
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid() # 输出 0~1 的动态权重
        )

    def forward(self, x):
        # 输入 x: [B, C, F, T] (CNN 提取的多尺度时空特征)
        B = x.shape[0]
        global_context = self.pool(x).view(B, -1) # [B, C]
        alpha = self.mlp(global_context) # [B, 1]
        
        # 约束 alpha 的范围在 [0.05, 0.95]，防止图拓扑彻底断开或变得死板
        alpha = alpha * 0.9 + 0.05 
        return alpha.unsqueeze(-1) # 返回 [B, 1, 1] 供图谱广播相乘

# ------------------------------------------------------------
# 3. 动态水声语义图网络 (Dynamic Acoustic Knowledge GCN)
# ------------------------------------------------------------
class AcousticKnowledgeGCN(nn.Module):
    def __init__(self, in_channels, num_freq_bins, sr=16000, fmin=0, fmax=8000):
        super().__init__()
        self.num_freq_bins = num_freq_bins  
        self.in_channels = in_channels
        self.use_prior_mask = True 
        
        self.query = nn.Linear(in_channels, in_channels // 2)
        self.key = nn.Linear(in_channels, in_channels // 2)
        self.value = nn.Linear(in_channels, in_channels) 
        
        self.gcn_weight = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.3)

        self.entity_embedding = nn.Parameter(torch.randn(1, num_freq_bins, in_channels))
        
        expert_prior = self._build_expert_knowledge_prior(sr, fmin, fmax)
        self.knowledge_adj = nn.Parameter(expert_prior.clone().float())
        
        # 静态 alpha 退化为 fallback（当消融实验关掉动态门控时使用）
        self.fallback_alpha = nn.Parameter(torch.tensor([0.5])) 

    def _build_expert_knowledge_prior(self, sr, fmin, fmax, tol=0.2):
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
                if abs(i - j) == 1:
                    A[i, j] = max(A[i, j].item(), 0.3)
                    A[j, i] = max(A[j, i].item(), 0.3)
                
                ratio = center_freqs[j] / (center_freqs[i] + 1e-8)
                for k in [2.0, 3.0, 4.0]:
                    if abs(ratio - k) < tol:
                        A[i, j] = max(A[i, j].item(), np.exp(- ((ratio - k)**2) / (0.1**2)))
                        A[j, i] = A[i, j]
        
        row_sum = A.sum(dim=1, keepdim=True)
        A = A / (row_sum + 1e-8)
        return A

    def forward(self, x, dynamic_alpha=None, B=None, T=None):
        B_T, n_freqs, C = x.shape
        
        x_embedded = x + self.entity_embedding
        
        Q = self.query(x_embedded)  
        K = self.key(x_embedded)
        V = self.value(x) 
        
        scale = K.shape[-1] ** 0.5
        A_data = torch.bmm(Q, K.transpose(1, 2)) / scale
        
        if getattr(self, 'use_prior_mask', True):
            A_kg = F.relu(self.knowledge_adj)
            A_kg = A_kg.unsqueeze(0).expand(B_T, -1, -1)
            
            # 🌟 核心融合：使用动态传入的 Alpha 进行样本级图谱约束
            if dynamic_alpha is not None and B is not None and T is not None:
                # dynamic_alpha 原本是 [B, 1, 1], 需要沿着时间轴 T 复制展开成 [B*T, 1, 1]
                alpha_expanded = dynamic_alpha.repeat_interleave(T, dim=0)
                A_fused = A_data + alpha_expanded * A_kg
            else:
                A_fused = A_data + self.fallback_alpha * A_kg
        else:
            A_fused = A_data
        
        A_dynamic = F.softmax(A_fused, dim=-1)
        A_dynamic = self.dropout(A_dynamic)

        z = torch.bmm(A_dynamic, V)  
        z = self.gcn_weight(z)
        out = self.norm(x + F.relu(z))
        
        return out

# ------------------------------------------------------------
# 4. 关键帧时序注意力 (Temporal Attention Pooling)
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
# 5. 最终主模型：Dynamic-HTAN (集成并行门控)
# ------------------------------------------------------------
class HTAN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, base_channels=32, 
                 input_fdim=128, input_tdim=1024,
                 use_graph=True, use_prior_mask=True,
                 use_temporal_encoder=True, use_temporal_attention=True):
        super().__init__()
        
        self.use_graph = use_graph
        self.use_prior_mask = use_prior_mask
        self.use_temporal_encoder = use_temporal_encoder
        self.use_temporal_attention = use_temporal_attention
        
        self.frontend = nn.Sequential(
            MultiScaleConvBlock(in_channels, base_channels),      
            MultiScaleConvBlock(base_channels, base_channels*2), 
            MultiScaleConvBlock(base_channels*2, base_channels*4) 
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(2, in_channels, input_fdim, input_tdim) 
            self.frontend.eval()
            dummy_out = self.frontend(dummy_input)
            self.frontend.train()
            _, cnn_out_c, f_out, t_out = dummy_out.shape
            
        # 🌟 并行模块：信噪比感知门控网络
        self.snr_gating = SNRAwareGating(in_channels=cnn_out_c)
            
        self.harmonic_gcn = AcousticKnowledgeGCN(
            in_channels=cnn_out_c, 
            num_freq_bins=f_out,
            sr=16000, fmin=0, fmax=8000 
        )
        self.harmonic_gcn.use_prior_mask = self.use_prior_mask
        
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
        
        # --- 模块 1：图网络开关与动态门控 ---
        if self.use_graph:
            # 🌟 计算当前这段音频的专属动态 Alpha
            dynamic_alpha = self.snr_gating(x) 
            
            x = x.permute(0, 3, 2, 1).contiguous().view(B * T_out, F_out, C)
            # 🌟 将 dynamic_alpha 注入 GCN 进行样本级控制
            x = self.harmonic_gcn(x, dynamic_alpha=dynamic_alpha, B=B, T=T_out)
            
            x = x.view(B, T_out, F_out, C)
            x_mean = x.mean(dim=2)          
            x_max = x.max(dim=2).values     
            x = torch.cat([x_mean, x_max], dim=-1)
        else:
            x = x.mean(dim=2).transpose(1, 2) 
        
        # --- 模块 2：时序编码开关 ---
        if self.use_temporal_encoder:
            x, _ = self.temporal_encoder(x) 
            
        # --- 模块 3：帧级注意力开关 ---
        if self.use_temporal_attention:
            x, attn_weights = self.temporal_attention(x) 
        else:
            x = x.mean(dim=1)
        
        if extract_feature:
            return x
            
        logits = self.classifier(x)
        return logits