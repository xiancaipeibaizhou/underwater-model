# src/lightning/ssl_litmodel.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class SSL_LitModel(pl.LightningModule):
    def __init__(self, encoder, projector, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # 假设 dataloader 返回两个增强视图 x1, x2 (例如经过不同的时频掩码)
        (x1, x2), _ = batch 
        
        # 1. 提取特征
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # 2. 计算对比损失 (InfoNCE) 或者 MSE 距离
        # 初版可以先用简单的 Cosine Similarity Loss 或 SimSiam 的负余弦相似度
        loss = self.contrastive_loss(z1, z2) 
        
        self.log('train_ssl_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)