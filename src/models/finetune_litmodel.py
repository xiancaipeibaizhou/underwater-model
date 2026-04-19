import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def plot_and_save_confusion_matrix(cm, target_names, save_path):
    """MILAN 同款高级混淆矩阵画图函数"""
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

class Finetune_LitModel(L.LightningModule):
    def __init__(self, encoder, feature_extractor, num_classes, feature_dim, lr=1e-4, alpha=0.1):
        super().__init__()
        self.encoder = encoder              # 预训练好的 HTAN 模型
        self.feature_extractor = feature_extractor # 物理特征提取器
        self.num_classes = num_classes
        self.lr = lr
        self.alpha = alpha                  # 创新点2：Prototype Loss 的权重 (通常设为 0.1)
        
        # 🌟 核心创新：注册类别原型 (Prototypes) 矩阵
        # register_buffer 可以让它存在 GPU 上并保存在权重里，但不会作为网络参数被反向传播更新
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        
        # 验证集指标监控
        self.val_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.criterion = nn.CrossEntropyLoss()

    def compute_prototype_loss(self, features, labels):
        """计算特征与类别原型的距离，并使用滑动平均动态更新原型"""
        # 🌟 核心修复：必须先对特征进行 L2 归一化，防止绝对数值爆炸！
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 1. 提取当前 batch 样本对应的本类原型
        batch_prototypes = self.prototypes[labels] 
        # 🌟 同样对取出的原型进行归一化
        batch_prototypes_norm = F.normalize(batch_prototypes, p=2, dim=1)
        
        # 2. 拉近样本与本类中心的距离 (MSE)
        proto_loss = F.mse_loss(features_norm, batch_prototypes_norm)
        
        # 3. 滑动平均 (EMA) 更新系统中的类别原型 (使用归一化前的原始特征更新即可)
        momentum = 0.9
        with torch.no_grad(): 
            for i in range(len(features)):
                c = labels[i]
                f = features[i] # 用原始数值更新
                if self.prototype_counts[c] == 0:
                    self.prototypes[c] = f
                else:
                    self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * f
                self.prototype_counts[c] += 1
                
        return proto_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # 1. 过特征提取器得到 2D 频谱 (微调阶段默认不加残酷掩码)
        x_spec = self.feature_extractor(x)
        
        # 2. 🌟 拦截高维特征：供 Prototype 计算使用
        features = self.encoder(x_spec, extract_feature=True)
        # 3. 正常通过分类器输出预测
        logits = self.encoder.classifier(features)
        
        # 4. 计算多任务 Loss
        ce_loss = self.criterion(logits, y)
        proto_loss = self.compute_prototype_loss(features, y)
        
        total_loss = ce_loss + self.alpha * proto_loss
        
        self.log('train_ce_loss', ce_loss, prog_bar=False)
        self.log('train_proto_loss', proto_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_spec = self.feature_extractor(x)
        logits = self.encoder(x_spec) # eval 模式下会自动跳过 extract_feature 直接输出预测
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_macro_f1(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_macro_f1', self.val_macro_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_spec = self.feature_extractor(x)
        logits = self.encoder(x_spec) 
        preds = torch.argmax(logits, dim=1)
        
        # 拦截并存起来，不在 Lightning 内部算乱七八糟的指标
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_targets = []
            
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        
        return self.criterion(logits, y)

    def on_test_epoch_end(self):
        """完全使用 sklearn 接管所有严谨指标的计算与绘图"""
        if hasattr(self, 'test_preds') and len(self.test_preds) > 0:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            
            # 1. 算尽天下指标
            acc = accuracy_score(targets, preds)
            apr = precision_score(targets, preds, average='weighted', zero_division=0)
            re = recall_score(targets, preds, average='weighted', zero_division=0)
            f1_mac = f1_score(targets, preds, average='macro', zero_division=0)
            f1_wei = f1_score(targets, preds, average='weighted', zero_division=0)
            
            # 抛出给外部拿去写 CSV
            self.custom_metrics = {
                'ACC': acc,
                'APR_Weighted': apr,
                'RE_Weighted': re,
                'F1_Macro': f1_mac,
                'F1_Weighted': f1_wei
            }
            
            # 2. 定位我们在主函数指定的专属保存文件夹
            save_dir = getattr(self, "test_save_dir", ".")
            class_names = getattr(self, "class_names", [f"Class {i}" for i in range(self.num_classes)])
            
            # 3. 绘制 SOTA 级混淆矩阵
            cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
            plot_and_save_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix.png"))
            
            # 4. 打印并保存华丽的分类报告
            report = classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0)
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
        # 🌟 差异化学习率：给预训练好的身体较小的学习率，给新头较大的学习率
        encoder_params = [p for n, p in self.encoder.named_parameters() if 'classifier' not in n]
        classifier_params = self.encoder.classifier.parameters()
        
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': self.lr * 0.1}, 
            {'params': classifier_params, 'lr': self.lr}
        ])
        return optimizer