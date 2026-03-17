import torch.nn as nn

class SSLProjector(nn.Module):
    """用于自监督对比学习和重建的投影头"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    """用于下游微调的分类头"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)