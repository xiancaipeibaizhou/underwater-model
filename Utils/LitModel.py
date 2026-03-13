import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, AveragePrecision
from src.models.custom_model import HTAN

class LitModel(L.LightningModule):
    def __init__(self, Params, model_name, num_classes, numBins=None, RR=None):
        """
        初始化 PyTorch Lightning 模型包装器
        numBins 和 RR 保留是为了兼容原版 demo_light.py 的传参，在纯 HTAN 模型中不再强制需要。
        """
        super().__init__()
        self.save_hyperparameters()
        self.Params = Params
        self.num_classes = num_classes
        self.model_name = model_name

        # ---------------------------------------------------------
        # 1. 初始化核心架构 (HTAN)
        # ---------------------------------------------------------
        if self.model_name == 'HTAN':
            print("🚀 Initializing HTAN (Harmonic-Temporal Attention Network) from scratch...")
            
            # 动态计算时间步 T_dim (根据 5秒切片, 16000Hz采样率, 512 hop_length 推算)
            expected_t_dim = int((Params['segment_length'] * Params['sample_rate']) / Params['hop_length']) + 1
            
            self.model = HTAN(
                num_classes=self.num_classes,
                in_channels=1, 
                base_channels=32,
                input_fdim=Params['number_mels'],
                input_tdim=expected_t_dim
            )
        else:
            raise ValueError(f"❌ Unsupported model: {model_name}. Please use 'HTAN'.")

        # ---------------------------------------------------------
        # 2. 定义损失函数与评估指标
        # ---------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用 torchmetrics 进行准确率和 AUPRC 的计算
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        # 为了兼容你 demo_light.py 中的 monitor='val_auprc'
        self.val_auprc = AveragePrecision(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        
        # AUPRC 需要 softmax 后的概率
        probs = torch.softmax(logits, dim=1)
        self.val_auprc(probs, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auprc', self.val_auprc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 使用 Adam 优化器，学习率由启动参数传入
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.Params['lr'], 
            weight_decay=1e-4  # 加入轻微的正则化防止过拟合
        )
        
        # 加入学习率衰减策略 (ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auprc", # 根据验证集的 AUPRC 来衰减学习率
                "frequency": 1
            },
        }