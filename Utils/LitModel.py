import torch
import torch.nn as nn
import lightning as L
# 1. 新增 F1Score 导入
from torchmetrics import Accuracy, AveragePrecision, F1Score
from src.models.custom_model import HTAN

class LitModel(L.LightningModule):
    def __init__(self, Params, model_name, num_classes, numBins=None, RR=None):
        super().__init__()
        self.save_hyperparameters()
        self.Params = Params
        self.num_classes = num_classes
        self.model_name = model_name

        # ---------------------------------------------------------
        # 1. 初始化核心架构 (HTAN) 及其消融开关
        # ---------------------------------------------------------
        if self.model_name == 'HTAN':
            print("🚀 Initializing HTAN (Harmonic-Temporal Attention Network)...")
            expected_t_dim = int((Params['segment_length'] * Params['sample_rate']) / Params['hop_length']) + 1
            
            # 2. 将消融开关从 Params 传入，如果 Params 没传，默认设为 True (即 Full Model)
            self.model = HTAN(
                num_classes=self.num_classes,
                in_channels=1, 
                base_channels=32,
                input_fdim=Params['number_mels'],
                input_tdim=expected_t_dim,
                use_graph=Params.get('use_graph', True),
                use_prior_mask=Params.get('use_prior_mask', True),
                use_temporal_encoder=Params.get('use_temporal_encoder', True),
                use_temporal_attention=Params.get('use_temporal_attention', True)
            )
        else:
            raise ValueError(f"❌ Unsupported model: {model_name}. Please use 'HTAN'.")

        # ---------------------------------------------------------
        # 2. 定义损失函数与评估指标
        # ---------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        self.val_auprc = AveragePrecision(task="multiclass", num_classes=self.num_classes)
        
        # 3. 核心修改：新增 Macro-F1 指标
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_macro_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

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
        self.val_macro_f1(preds, y) # 计算 Macro-F1
        
        probs = torch.softmax(logits, dim=1)
        self.val_auprc(probs, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True) # 记录 Macro-F1
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, y)
        self.test_macro_f1(preds, y) # 计算 Test Macro-F1
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_macro_f1', self.test_macro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.Params['lr'], 
            weight_decay=1e-4 
        )
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
                "monitor": "val_macro_f1", # 4. 核心修改：衰减策略基于 Macro-F1
                "frequency": 1
            },
        }