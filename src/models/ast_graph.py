import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalHarmonicGCN(nn.Module):
    def __init__(self, in_channels=768, num_freq_bins=12, num_time_steps=15, dropout=0.3):
        super().__init__()
        self.F_bins = num_freq_bins
        self.T_steps = num_time_steps
        self.num_patches = num_freq_bins * num_time_steps 
        
        # 降低映射维度，防止模型过度拟合动态噪声
        reduced_dim = in_channels // 4  
        self.query = nn.Linear(in_channels, reduced_dim)
        self.key = nn.Linear(in_channels, reduced_dim)

        self.gcn_weight = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout) # 新增：随机丢弃神经元，防过拟合
        
        # 初始时，让物理先验法则占据绝对主导 (0.8)，只给动态图 (0.2) 的空间
        self.alpha = nn.Parameter(torch.tensor(0.8))

        self.register_buffer('A_physical', self._build_physical_adjacency())

    def _build_physical_adjacency(self):
        A = torch.zeros(self.num_patches, self.num_patches)
        for f in range(self.F_bins):
            for t in range(self.T_steps):
                node_idx = f * self.T_steps + t
                # 时序连续性
                if t < self.T_steps - 1:
                    right_idx = f * self.T_steps + (t + 1)
                    A[node_idx, right_idx] = 1.0
                    A[right_idx, node_idx] = 1.0
                # 跨频谐波共振
                harmonic_f = f * 2 + 1 
                if harmonic_f < self.F_bins:
                    harmonic_idx = harmonic_f * self.T_steps + t
                    A[node_idx, harmonic_idx] = 1.0
                    A[harmonic_idx, node_idx] = 1.0

        row_sum = A.sum(dim=1, keepdim=True)
        A = A / (row_sum + 1e-8)
        return A

    def forward(self, x):
        B, N, C = x.shape

        Q = self.query(x)  
        K = self.key(x)    
        A_dynamic = torch.bmm(Q, K.transpose(1, 2)) / ((C // 4) ** 0.5)
        A_dynamic = F.softmax(A_dynamic, dim=-1)  
        A_dynamic = self.dropout(A_dynamic) # 新增：随机丢弃某些注意力边

        # 限制 alpha 在 0 到 1 之间
        alpha_clamped = torch.sigmoid(self.alpha)
        A_combined = alpha_clamped * self.A_physical.unsqueeze(0) + (1 - alpha_clamped) * A_dynamic

        z = torch.bmm(A_combined, x)  
        z = self.gcn_weight(z)
        z = self.dropout(z) # 新增：图卷积输出后丢弃特征

        out = self.norm(x + F.relu(z))
        return out