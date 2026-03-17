import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class Finetune_LitModel(pl.LightningModule):
    def __init__(self, encoder, classifier, num_classes, feature_dim, lr=1e-4, alpha=0.1):
        """
        encoder: Stage 1 预训练好的编码器
        classifier: 重新初始化的分类头
        num_classes: 类别数量
        feature_dim: encoder 输出的特征维度
        lr: 学习率
        alpha: Prototype Loss 的权重
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.lr = lr
        self.alpha = alpha
        
        # 【关键】使用 register_buffer 注册类原型中心
        # 这样它会随着模型一起移动到 GPU，且会保存在 checkpoint 中，但它不是网络的可训练参数(不用梯度更新)
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        
        # 记录每个类别更新了多少次，用于滑动平均更新原型
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

    def compute_prototype_loss(self, features, labels):
        """
        计算 Prototype Alignment Loss，并更新 Prototype
        """
        # 1. 计算当前 batch 特征到对应类原型的距离 (MSE 距离)
        # 获取当前 batch 中每个样本对应的类原型
        batch_prototypes = self.prototypes[labels] 
        # 计算特征与原型的均方误差 (拉近同类距离)
        proto_loss = F.mse_loss(features, batch_prototypes)
        
        # 2. 更新类原型 (使用滑动平均 - Exponential Moving Average)
        momentum = 0.9
        with torch.no_grad(): # 更新原型不需要算梯度
            for i in range(len(features)):
                c = labels[i]
                f = features[i]
                if self.prototype_counts[c] == 0:
                    self.prototypes[c] = f
                else:
                    self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * f
                self.prototype_counts[c] += 1
                
        return proto_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # 提取特征和分类结果
        features = self.encoder(x)
        logits = self.classifier(features)
        
        # 1. 基础分类损失 (Cross Entropy)
        ce_loss = F.cross_entropy(logits, y)
        
        # 2. 创新点2：Prototype Alignment Loss
        proto_loss = self.compute_prototype_loss(features, y)
        
        # 3. 总损失
        total_loss = ce_loss + self.alpha * proto_loss
        
        # 记录日志
        self.log('train_ce_loss', ce_loss, prog_bar=True)
        self.log('train_proto_loss', proto_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 微调阶段，通常分类头学习率大一点，encoder 学习率小一点（或者干脆冻结 encoder）
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.lr * 0.1}, # encoder 微调学习率设为十分之一
            {'params': self.classifier.parameters(), 'lr': self.lr}
        ])
        return optimizer