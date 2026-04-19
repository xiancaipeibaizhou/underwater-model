import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# 导入你刚刚修改过的各个模块
from Datasets.ShipsEar_dataloader import ShipsEarDataModule
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from src.models.custom_model import HTAN
from ssl_litmodel import SSL_LitModel  # 假设你把 ssl_litmodel.py 也放在了根目录

# 如果你有超参数配置文件，也可以导入 Params，这里我们显式定义关键参数
def main():
    print("🚀 开始执行 Stage 1: 水声自监督预训练 (Self-Supervised Pretraining)...")
    L.seed_everything(42)

    # =========================================================
    # 1. 初始化数据模块 (开启 is_ssl=True)
    # =========================================================
    # 注意：这里的 parent_folder 请确保指向你真实的切片数据集路径
    datamodule = ShipsEarDataModule(
        parent_folder='./shipsEar_AUDIOS',  
        batch_size={'train': 32, 'val': 32, 'test': 32},
        num_workers=8,
        is_ssl=True  
    )
    datamodule.setup()

    # =========================================================
    # 2. 初始化物理特征提取器 (提取频谱并施加时频掩码)
    # =========================================================
    feature_extractor = Feature_Extraction_Layer(
        input_feature='LogMelFBank',
        sample_rate=16000,
        window_length=2048,
        hop_length=512,
        number_mels=128,
        segment_length=5
    )
    
    # 动态获取输出的频段(F)和时间帧(T)维度，喂给 HTAN
    ft_dims = feature_extractor.output_dims
    inpf, inpt = ft_dims[1], ft_dims[2]
    print(f"📊 提取的频谱维度: Freq={inpf}, Time={inpt}")

    # =========================================================
    # 3. 初始化 HTAN 作为特征编码器 (Encoder)
    # =========================================================
    # 在预训练阶段，分类数量 num_classes 失去意义，只是占位符
    base_channels = 32
    use_graph = True
    
    encoder = HTAN(
        num_classes=5, 
        in_channels=1,
        base_channels=base_channels,
        input_fdim=inpf,
        input_tdim=inpt,
        use_graph=use_graph,
        use_prior_mask=True,
        use_temporal_encoder=True,
        use_temporal_attention=True
    )
    
    # 计算 HTAN 最终输出的特征维度 (用于初始化对比学习的投影头)
    # 根据你的代码: gru_input_size = cnn_out_c * 2 if use_graph else cnn_out_c
    # cnn_out_c = base_channels * 4
    htan_out_dim = (base_channels * 4 * 2) if use_graph else (base_channels * 4)

    # =========================================================
    # 4. 组装自监督 Lightning Module
    # =========================================================
    model = SSL_LitModel(
        encoder=encoder,
        feature_extractor=feature_extractor,
        feature_dim=htan_out_dim,
        lr=1e-3
    )

    # =========================================================
    # 5. 设置回调与 Trainer
    # =========================================================
    # 保存 loss 最低的预训练权重
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/ssl_pretrain/',
        filename='best_ssl_encoder-{epoch:02d}-{ssl_loss:.4f}',
        monitor='ssl_loss',
        mode='min',
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("tb_logs", name="ssl_pretrain")

    trainer = L.Trainer(
        max_epochs=100,            # 自监督通常需要较长的 epoch 收敛
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10
    )

    # =========================================================
    # 6. 正式开始训练
    # =========================================================
    # 预训练阶段，我们实际上不需要验证集算分类准确率，只看 train_loss (InfoNCE) 下降即可
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
    
    print("\n🎉 Stage 1 预训练结束！最优编码器权重已保存在 ./checkpoints/ssl_pretrain/ 目录下。")

if __name__ == '__main__':
    main()