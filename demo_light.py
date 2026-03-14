import numpy as np
import argparse
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import os

from Utils.LitModel import LitModel

from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(Params):
    model_name = Params['Model_name']
    batch_size = Params['batch_size']
    t_batch_size = batch_size['train']
    num_workers = Params['num_workers']
    
    if Params['data_selection'] == 1:
        dataset_dir = 'shipsEar_AUDIOS/' 
        Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)
        data_module = ShipsEarDataModule(parent_folder=dataset_dir, batch_size=batch_size, num_workers=num_workers)
        num_classes = 5 
    else:
        raise ValueError('当前仅支持 ShipsEar 数据集')
    
    DataName = "ShipsEar"
    torch.set_float32_matmul_precision('medium') 
    
    # ==========================================
    # SOTA 级功能 1：定义组名 (Group)，严格区分消融变体
    # ==========================================
    group_str = f"G{int(Params['use_graph'])}_P{int(Params['use_prior_mask'])}_TE{int(Params['use_temporal_encoder'])}_TA{int(Params['use_temporal_attention'])}"
    
    print(f'\n🚀 Starting Experiments for {DataName} using {model_name} | Group: {group_str} 🚀')
    
    # ==========================================
    # SOTA 级功能 2：准备全局的汇总 CSV 表格
    # ==========================================
    csv_file = "htan_ablations_results.csv" 
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Model,Group,Run_Index,Seed,ACC,APR_Weighted,RE_Weighted,F1_Macro,F1_Weighted,Val_Macro_F1\n")

    numRuns = 3
    for run_number in range(0, numRuns):
        current_seed = run_number + 42
        seed_everything(current_seed, workers=True) 
        
        # ==========================================
        # SOTA 级功能 3：建立像 MILAN 一样的极简本地文件夹
        # 路径规范：results/ShipsEar/G1_P1_TE1_TA1/Run_0/
        # ==========================================
        save_dir = os.path.join("results", DataName, group_str, f"Run_{run_number}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f'\n>>> Starting [ {group_str} ] | Run {run_number+1}/{numRuns} (Seed: {current_seed}) <<<\n')
        print(f'📁 Outputs will be saved strictly to: {save_dir}\n')
    
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,          # 强制把最佳权重保存在纯净目录下，别乱跑
            filename='best_model',
            monitor='val_macro_f1', 
            save_top_k=1,
            mode='max',
            verbose=False,
            save_weights_only=True
        )
    
        early_stopping_callback = EarlyStopping(
            monitor='val_macro_f1', 
            patience=Params['patience'],
            verbose=True,
            mode='max'
        )

        model_wrapper = LitModel(Params, model_name, num_classes)

        if run_number == 0:
            num_params = count_trainable_params(model_wrapper)
            print(f'\n💡 Total Trainable Parameters: {num_params / 1e6:.4f} M\n')

        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=False, # 🌟【核心手术】：完全关闭 Lightning 所有自带 Logger，彻底告别 tb_logs！
            log_every_n_steps=10,
            enable_progress_bar=True,
            accelerator='gpu',       
            devices="auto"
        )
        
        # 1. 开始训练
        trainer.fit(model=model_wrapper, datamodule=data_module) 
        best_val_f1 = checkpoint_callback.best_model_score.item()
    
        # 加载刚才存入 save_dir 的最佳权重
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
        )
    
        # 2. 验证模型 (注入纯净目录，LitModel 会在里面画图)
        best_model.test_save_dir = save_dir
        trainer.test(model=best_model, datamodule=data_module)
        
        # 获取在 LitModel 内部算好的所有严谨指标
        metrics = best_model.custom_metrics
    
        # 3. 自动追加写入全局 CSV 表格
        with open(csv_file, "a") as f:
            f.write(f"{DataName},{model_name},{group_str},{run_number},{current_seed},"
                    f"{metrics['ACC']:.4f},{metrics['APR_Weighted']:.4f},{metrics['RE_Weighted']:.4f},"
                    f"{metrics['F1_Macro']:.4f},{metrics['F1_Weighted']:.4f},{best_val_f1:.4f}\n")
    
        # 在文件夹里留一份基础日志证明
        with open(os.path.join(save_dir, "metrics.txt"), "w") as file:
            file.write(f"=== {model_name} ({group_str}) Run {run_number} ===\n")
            file.write(f"Best Validation Macro-F1: {best_val_f1:.4f}\n")
            for k, v in metrics.items():
                file.write(f"Test {k}: {v:.4f}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Advanced UATR HTAN Experiments')
    parser.add_argument('--model', type=str, default='HTAN', help='Select baseline model architecture')
    parser.add_argument('--data_selection', type=int, default=1, help='Dataset selection: 1=ShipsEar')
    
    # 0/1 控制消融，非常严谨
    parser.add_argument('--use_graph', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_prior_mask', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_temporal_encoder', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    parser.add_argument('--use_temporal_attention', type=int, choices=[0, 1], default=1, help='1=True, 0=False')
    
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='input batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train each model for')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for Dataloader')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank', help='Audio feature for extraction')
    parser.add_argument('--window_length', type=int, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, default=512, help='hop length')
    parser.add_argument('--number_mels', type=int, default=128, help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Dataset Sample Rate')
    parser.add_argument('--segment_length', type=int, default=5, help='Dataset Segment Length')
                 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 彻底抛弃旧的 Demo_Parameters.py，直接把命令行参数转为字典
    params = vars(args).copy()
    
    # 补充内部逻辑需要的字段映射
    params['Model_name'] = args.model
    params['batch_size'] = {
        'train': args.train_batch_size,
        'val': args.val_batch_size,
        'test': args.test_batch_size
    }
    
    # 确保消融开关是严谨的布尔值
    for key in ['use_graph', 'use_prior_mask', 'use_temporal_encoder', 'use_temporal_attention']:
        params[key] = bool(params[key])
        
    main(params)