# LitModel.py

import torch
from torch import nn
from Utils.Network_functions import initialize_model
import torch.nn.functional as F
import lightning as L
import torchmetrics

# =========================================================================
# >>> 核心创新：有监督对比度量损失 (Supervised Contrastive Loss) <<<
# 它的作用不是分类，而是在高维空间中像万有引力一样，把同类的物理指纹拉近，异类推远
# =========================================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [Batch, 768] 的高维物理声学指纹
        device = features.device
        # 对特征进行 L2 归一化，将其投影到超球面上计算余弦距离
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        # 计算所有样本两两之间的相似度矩阵 (Cosine Similarity)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # 增加数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 掩码矩阵：排除自己和自己的对比
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 找到 Batch 中标签相同的所有正样本对 (同类船只)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_pos = mask_pos * logits_mask 

        # 计算度量距离损失
        pos_sum = mask_pos.sum(1)
        pos_sum[pos_sum == 0] = 1.0 # 防止除零
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / pos_sum

        loss = - mean_log_prob_pos
        return loss.mean()
# =========================================================================

class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, numBins, RR):
        super().__init__()

        self.learning_rate = Params['lr']
        self._skip_feature = bool(
            Params.get('skip_feature', (Params.get('data_selection', -1) in (3, 4))))
        
        self.model_ft, self.feature_extraction_layer = initialize_model(
            model_name, num_classes, numBins, RR, Params['sample_rate'],
            segment_length=Params['segment_length'], window_length=Params['window_length'],
            hop_length=Params['hop_length'], number_mels=Params['number_mels'],
            t_mode=Params['train_mode'], h_shared=Params['histograms_shared'],
            a_shared=Params['adapters_shared'], parallel=Params['parallel'],
            input_feature=Params['feature'], adapter_location=Params['adapter_location'],
            adapter_mode=Params['adapter_mode'], histogram_location=Params['histogram_location'],
            histogram_mode=Params['histogram_mode'], lora_target=Params['lora_target'],
            lora_rank=Params['lora_rank'], r_shared=Params['lora_shared'],
            b_mode=Params['bias_mode'], ssf_shared=Params['ssf_shared'],
            ssf_mode=Params['ssf_mode'], skip_feature=self._skip_feature,
            image_input_hw=Params.get('image_input_hw', (96, 96)),
        )

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        
        # AUPRC 评价指标
        self.val_auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_classes, average='macro')
        
        self.save_hyperparameters()

        self.encoder = self.feature_extraction_layer if self.feature_extraction_layer is not None else nn.Identity()

        # =========================================================================
        # >>> 结构拆解：剥离分类头，露出高维特征 <<<
        # =========================================================================
        if hasattr(self.model_ft, 'mlp_head'):
            self.classifier = self.model_ft.mlp_head
            # 将主干网络原本的分类头替换为空壳，让它直接吐出 [Batch, 768] 的特征矩阵
            self.model_ft.mlp_head = nn.Identity() 
        else:
            self.classifier = nn.Linear(768, num_classes)
            
        # 实例化度量损失计算器
        self.supcon_loss_fn = SupervisedContrastiveLoss(temperature=0.1)
        # =========================================================================

        if isinstance(self.encoder, torch.nn.Identity):
            print("[LitModel] Feature extractor: SKIPPED")
        else:
            print("[LitModel] Feature extractor: ENABLED (Audio feature path)")

    def forward(self, x):
        # 推理时：特征提取 + 分类，保持流水线不变
        features = self.model_ft(self.encoder(x))
        return self.classifier(features)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        # 1. 提取高维物理指纹 (嵌入特征) -> [Batch, 768]
        features = self.model_ft(self.encoder(x))
        
        # 2. 映射到类别概率 -> [Batch, num_classes]
        logits = self.classifier(features)
        
        # =========================================================================
        # >>> 联合损失计算 (Joint Meta-Learning Loss) <<<
        # =========================================================================
        # 损失 A：普通的分类交叉熵损失
        loss_ce = F.cross_entropy(logits, y)
        
        # 损失 B：度量聚类损失，强迫同类高维特征在空间中互相靠近
        loss_metric = self.supcon_loss_fn(features, y)
        
        # 最终损失：按照一定比例融合 (0.5 是顶刊常用经验值)
        loss = loss_ce + 0.5 * loss_metric
        # =========================================================================

        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)
        # 把两个独立的 loss 也打印出来方便观察聚类情况
        self.log('loss_ce', loss_ce, on_step=False, on_epoch=True)
        self.log('loss_metric', loss_metric, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        features = self.model_ft(self.encoder(x))
        logits = self.classifier(features)
        val_loss = F.cross_entropy(logits, y)

        self.val_acc(logits, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        self.val_auprc(logits, y)
        self.log('val_auprc', self.val_auprc, on_step=False, on_epoch=True)

        return val_loss
 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        features = self.model_ft(self.encoder(x))        
        logits = self.classifier(features)
        test_loss = F.cross_entropy(logits, y)
        
        self.test_acc(logits, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        self.test_auprc(logits, y)
        self.log('test_auprc', self.test_auprc, on_step=False, on_epoch=True)

        return test_loss

    def configure_optimizers(self):
        base_lr = self.learning_rate  
        optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr)
        return optimizer