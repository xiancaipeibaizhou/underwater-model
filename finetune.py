import os
import glob
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from Datasets.ShipsEar_dataloader import ShipsEarDataModule
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from src.models.custom_model import HTAN
from ssl_litmodel import SSL_LitModel
from src.models.finetune_litmodel import Finetune_LitModel

def main():
    print("🚀 开始执行 Stage 2: 跨域稳健微调 (Finetuning with Prototype Alignment)...")
    L.seed_everything(42)

    # =========================================================
    # 1. 自动寻找 Stage 1 保存的最佳预训练权重
    # =========================================================
    ckpt_dir = './checkpoints/ssl_pretrain/'
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError("🚨 未找到预训练权重！请确保你已经成功运行了 pretrain.py 并生成了 .ckpt 文件。")
    
    # 默认拿最近生成的一个（或者你可以手动写死 best_ckpt 的路径）
    best_ckpt = sorted(ckpt_files)[-1]
    print(f"📦 成功锁定预训练权重: {os.path.basename(best_ckpt)}")

    # =========================================================
    # 2. 初始化数据模块 (回到常规模式，is_ssl=False)
    # =========================================================
    datamodule = ShipsEarDataModule(
        parent_folder='./shipsEar_AUDIOS', # 这里用你服务器上的实际音频切片路径
        batch_size={'train': 32, 'val': 32, 'test': 32},
        num_workers=8,
        is_ssl=False  # 🌟 关闭双视图，开启正常监督流
    )
    datamodule.setup()

    # =========================================================
    # 3. 初始化相同维度的骨干网络
    # =========================================================
    feature_extractor = Feature_Extraction_Layer(
        input_feature='LogMelFBank', sample_rate=16000, window_length=2048,
        hop_length=512, number_mels=128, segment_length=5
    )
    ft_dims = feature_extractor.output_dims
    
    encoder = HTAN(
        num_classes=5, in_channels=1, base_channels=32,
        input_fdim=ft_dims[1], input_tdim=ft_dims[2],
        use_graph=True, use_prior_mask=True,
        use_temporal_encoder=True, use_temporal_attention=True
    )

    # =========================================================
    # 4. 核心注入：从 SSL 模型中剥离出预训练好的 Encoder
    # =========================================================
    print("🔄 正在将预训练知识注入模型...")
    # 利用 Pytorch Lightning 机制安全加载
    ssl_model = SSL_LitModel.load_from_checkpoint(
        best_ckpt, 
        encoder=encoder, 
        feature_extractor=feature_extractor,
        feature_dim=32 * 4 * 2 # HTAN 输出特征维数
    )
    pretrained_encoder = ssl_model.encoder

    # =========================================================
    # 5. 组装创新点 2：带原型约束的微调模型
    # =========================================================
    model = Finetune_LitModel(
        encoder=pretrained_encoder,
        feature_extractor=feature_extractor,
        num_classes=5,
        feature_dim=32 * 4 * 2,
        lr=1e-2,    
        alpha=0.01   # Prototype Loss 的约束强度
    )

    # =========================================================
    # 6. 设置回调与开始训练
    # =========================================================
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/finetune/',
        filename='best_finetune-{epoch:02d}-{val_macro_f1:.4f}',
        monitor='val_macro_f1',
        mode='max',
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=100, # 微调通常收敛很快，50 个 epoch 足够
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    print("\n🧪 训练完成，正在自动加载最优权重进行 Test 集评测...")
    
    # 1. 创建结果保存目录
    res_dir = './results/finetune_results'
    os.makedirs(res_dir, exist_ok=True)
    
    # 2. 把保存路径和类别名称传给模型
    model.test_save_dir = res_dir
    model.class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'] 
    
    # 3. 触发测试阶段 (ckpt_path='best' 会自动加载刚刚验证集 F1 最高的权重)
    trainer.test(model, dataloaders=datamodule.test_dataloader(), ckpt_path='best')
    
    # 4. 仿照 demo_light 将最终指标持久化为 CSV
    import pandas as pd
    if hasattr(model, 'custom_metrics'):
        df = pd.DataFrame([model.custom_metrics])
        csv_path = os.path.join(res_dir, 'finetune_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"📊 核心指标已成功汇总并保存至 CSV: {csv_path}")
    print("\n🎉 Stage 2 微调结束！最优分类模型已保存在 ./checkpoints/finetune/ 目录下。")

if __name__ == '__main__':
    main()