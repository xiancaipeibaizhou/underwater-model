import torch
import torch.nn as nn
import lightning as L
from torchmetrics import F1Score
from src.models.custom_model import HTAN
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer

# ==========================================
# 引入 sklearn 与 seaborn，严格仿写 MILAN 可视化
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def plot_and_save_confusion_matrix(cm, target_names, save_path):
    """MILAN 同款高级混淆矩阵画图函数 (论文级)"""
    clean_target_names = [str(name).replace('\x96', '-').replace('\u2013', '-') for name in target_names]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) 
    
    num_classes = len(clean_target_names)
    fig_width = max(8, num_classes * 1.2)
    fig_height = max(6, num_classes * 1.0)
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=1.1) 
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 过滤掉 0，保持画面极度整洁
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else "0"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap='Blues', cbar=True,
                xticklabels=clean_target_names, yticklabels=clean_target_names, vmin=0.0, vmax=1.0)
    
    plt.title('Normalized Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


class LitModel(L.LightningModule):
    def __init__(self, Params, model_name, num_classes, numBins=None, RR=None):
        super().__init__()
        self.save_hyperparameters()
        self.Params = Params
        self.num_classes = num_classes
        self.model_name = model_name

        self.feature_extractor = Feature_Extraction_Layer(
            input_feature=Params.get('audio_feature', 'LogMelFBank'),
            sample_rate=Params.get('sample_rate', 16000),
            window_length=Params.get('window_length', 2048),
            hop_length=Params.get('hop_length', 512),
            number_mels=Params.get('number_mels', 128),
            segment_length=Params.get('segment_length', 5)
        )

        if self.model_name == 'HTAN':
            expected_t_dim = int((Params.get('segment_length', 5) * Params.get('sample_rate', 16000)) / Params.get('hop_length', 512)) + 1
            
            def safe_get(key, default=True):
                if hasattr(Params, key): return getattr(Params, key)
                if isinstance(Params, dict): return Params.get(key, default)
                return default

            self.model = HTAN(
                num_classes=self.num_classes,
                in_channels=1, 
                base_channels=32,
                input_fdim=Params.get('number_mels', 128),
                input_tdim=expected_t_dim,
                use_graph=safe_get('use_graph', True),
                use_prior_mask=safe_get('use_prior_mask', True),
                use_temporal_encoder=safe_get('use_temporal_encoder', True),
                use_temporal_attention=safe_get('use_temporal_attention', True)
            )
        else:
            raise ValueError(f"❌ Unsupported model: {model_name}. Please use 'HTAN'.")

        self.criterion = nn.CrossEntropyLoss()
        
        # 仅保留用于 EarlyStopping 监控的验证集 F1
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        
        # 拦截器：收集预测和标签以交给 sklearn 处理
        self.test_preds = []
        self.test_targets = []
        self.class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False) # 关掉 logger
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_macro_f1(preds, y) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        # 拦截并存起来，不在 Lightning 内部算乱七八糟的指标
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        
        return self.criterion(logits, y)

    def on_test_epoch_end(self):
        """完全使用 sklearn 接管所有严谨指标的计算与绘图"""
        if len(self.test_preds) > 0:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            
            # 1. 算尽天下指标
            acc = accuracy_score(targets, preds)
            apr = precision_score(targets, preds, average='weighted', zero_division=0)
            re = recall_score(targets, preds, average='weighted', zero_division=0)
            f1_mac = f1_score(targets, preds, average='macro', zero_division=0)
            f1_wei = f1_score(targets, preds, average='weighted', zero_division=0)
            
            # 将指标抛出给外部的 demo_light.py 拿去写 CSV
            self.custom_metrics = {
                'ACC': acc,
                'APR_Weighted': apr,
                'RE_Weighted': re,
                'F1_Macro': f1_mac,
                'F1_Weighted': f1_wei
            }
            
            # 2. 定位我们在主函数指定的专属极简文件夹
            save_dir = getattr(self, "test_save_dir", ".")
            
            # 3. 绘制 SOTA 级混淆矩阵
            cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
            plot_and_save_confusion_matrix(cm, self.class_names, os.path.join(save_dir, "confusion_matrix.png"))
            
            # 4. 打印极其华丽的分类报告
            report = classification_report(targets, preds, target_names=self.class_names, digits=4, zero_division=0)
            with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
                f.write(report)
            
            print("\n" + "="*60)
            print("🚀 [TEST SET] DETAILED CLASSIFICATION REPORT")
            print("="*60)
            print(report)
            print("="*60)
            print(f"✅ Metrics & Confusion Matrix Image accurately saved to:\n   {save_dir}\n")
            
            # 清空内存防 OOM
            self.test_preds.clear()
            self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.Params['lr'], 
            weight_decay=self.Params.get('weight_decay', 1e-5)  # 👈 从超参数字典获取
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
                "monitor": "val_macro_f1", 
                "frequency": 1
            },
        }