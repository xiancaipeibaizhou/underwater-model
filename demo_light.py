import numpy as np
import argparse
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger  # 🌟 新增：论文级轻量化日志记录器
import os

from Utils.LitModel import LitModel

from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(Params):
    model_name = Params['Model_name']
    batch_size = Params['batch_size']
    num_workers = Params['num_workers']
    
    if Params['data_selection'] == 1:
        dataset_dir = 'shipsEar_AUDIOS/' 
        Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)
        data_module = ShipsEarDataModule(
            parent_folder=dataset_dir, 
            batch_size=batch_size, 
            num_workers=num_workers,
            test_snr=Params.get('test_snr')
        )
        num_classes = 5 
    else:
        raise ValueError('当前仅支持 ShipsEar 数据集')
    
    DataName = "ShipsEar"
    torch.set_float32_matmul_precision('medium') 
    
    snr_suffix = f"_SNR{Params['test_snr']}" if Params.get('test_snr') is not None else "_Clean"
    group_str = f"G{int(Params['use_graph'])}_P{int(Params['use_prior_mask'])}_TE{int(Params['use_temporal_encoder'])}_TA{int(Params['use_temporal_attention'])}{snr_suffix}"
    
    print(f'\n🚀 Starting Experiments for {DataName} using {model_name} | Group: {group_str} 🚀')
    
    csv_file = "htan_ablations_results.csv" 
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Model,ExpTime,Group,Run_Index,Seed,ACC,APR_Weighted,RE_Weighted,F1_Macro,F1_Weighted,Val_Macro_F1\n")

    numRuns = 1 if Params.get('test_only') else 3

    for run_number in range(0, numRuns):
        current_seed = run_number + 42
        seed_everything(current_seed, workers=True) 
        
        # 🌟 核心修复 2：按时间戳生成独立根目录，彻底杜绝覆盖！
        exp_folder_name = f"{DataName}_{Params['exp_time']}" if Params.get('exp_time') else DataName
        save_dir = os.path.join("results", exp_folder_name, group_str, f"Run_{run_number}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f'\n>>> Starting [ {group_str} ] | Run {run_number+1}/{numRuns} (Seed: {current_seed}) <<<\n')
        print(f'📁 Outputs will be saved strictly to: {save_dir}\n')
    
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
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

        # 🌟 核心修复 1：引入 CSVLogger 记录每个 Epoch 的 Loss 和 F1 曲线
        csv_logger = CSVLogger(save_dir=save_dir, name="training_curves")

        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=csv_logger, # 👈 重新启用专门为论文准备的 CSV 记录器
            log_every_n_steps=10,
            enable_progress_bar=True,
            accelerator='gpu',       
            devices="auto"
        )
        
        if not Params.get('test_only'):
            trainer.fit(model=model_wrapper, datamodule=data_module) 
            best_val_f1 = checkpoint_callback.best_model_score.item()
            best_model_path = checkpoint_callback.best_model_path
        else:
            print(f"\n⚠️ [鲁棒性测试模式] 跳过训练，直接加载 Clean 权重: {Params['ckpt_path']}")
            if Params.get('ckpt_path') is None or not os.path.exists(Params['ckpt_path']):
                raise FileNotFoundError(f"🚨 找不到指定的权重文件: {Params.get('ckpt_path')}！请检查路径。")
            best_model_path = Params['ckpt_path']
            best_val_f1 = 0.0 

        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
        )
    
        best_model.test_save_dir = save_dir
        trainer.test(model=best_model, datamodule=data_module)
        
        metrics = best_model.custom_metrics
    
        with open(csv_file, "a") as f:
            f.write(f"{DataName},{model_name},{Params.get('exp_time', 'None')},{group_str},{run_number},{current_seed},"
                    f"{metrics['ACC']:.4f},{metrics['APR_Weighted']:.4f},{metrics['RE_Weighted']:.4f},"
                    f"{metrics['F1_Macro']:.4f},{metrics['F1_Weighted']:.4f},{best_val_f1:.4f}\n")
    
        with open(os.path.join(save_dir, "metrics.txt"), "w") as file:
            file.write(f"=== {model_name} ({group_str}) Run {run_number} ===\n")
            file.write(f"Best Validation Macro-F1: {best_val_f1:.4f}\n")
            for k, v in metrics.items():
                file.write(f"Test {k}: {v:.4f}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Advanced UATR HTAN Experiments')
    parser.add_argument('--model', type=str, default='HTAN', help='Select baseline model architecture')
    parser.add_argument('--data_selection', type=int, default=1, help='Dataset selection: 1=ShipsEar')
    
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

    parser.add_argument('--test_snr', type=float, default=None, help='测试集注入的 SNR (dB)')
    parser.add_argument('--test_only', action='store_true', help='跳过训练，仅加载权重进行测试')
    parser.add_argument('--ckpt_path', type=str, default=None, help='仅测试模式下加载的权重路径')             
    parser.add_argument('--exp_time', type=str, default='', help='按时间戳生成独立文件夹防止覆盖')             
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = vars(args).copy()
    
    params['Model_name'] = args.model
    params['batch_size'] = {
        'train': args.train_batch_size,
        'val': args.val_batch_size,
        'test': args.test_batch_size
    }
    
    for key in ['use_graph', 'use_prior_mask', 'use_temporal_encoder', 'use_temporal_attention']:
        params[key] = bool(params[key])
        
    main(params)