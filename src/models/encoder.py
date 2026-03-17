import torch.nn as nn

class UnderwaterEncoder(nn.Module):
    def __init__(self, original_backbone):
        super().__init__()
        # 把你现有的 CNN/AST 剥离掉最后一层全连接层
        self.feature_extractor = original_backbone 
        
    def forward(self, x):
        # x: [B, C, F, T] (频谱图)
        return self.feature_extractor(x)