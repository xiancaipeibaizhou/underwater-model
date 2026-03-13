import numpy as np
import argparse
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import os

from Demo_Parameters import Parameters
from Utils.LitModel import LitModel

# 只保留 ShipsEar 相关的核心数据模块
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
        raise ValueError('代码已进行极简优化，当前仅支持 ShipsEar 数据集 (请确保参数 --data_selection 1)')
    
    DataName = "ShipsEar"
    print(f'\n🚀 Starting Experiments for {DataName} using {model_name} 🚀')
    
    numRuns = 3
    progress_bar = True 
    torch.set_float32_matmul_precision('medium')
    
    # 记录 Macro-F1 的列表
    all_val_f1s = []
    all_test_f1s = []
    
    # 构建极其清晰的日志保存路径名
    exp_name = (
        f"{model_name}_G{int(Params.get('use_graph', True))}"
        f"_P{int(Params.get('use_prior_mask', True))}"
        f"_TE{int(Params.get('use_temporal_encoder', True))}"
        f"_TA{int(Params.get('use_temporal_attention', True))}"
    )

    for run_number in range(0, numRuns):
        # 不同的 Run 使用不同的 Seed，以验证模型初始化稳定性 (数据划分不变，因为 dataloader 会读取 txt)
        seed_everything(run_number + 42, workers=True) 
                  
        print(f'\n>>> Starting Run {run_number+1}/{numRuns} (Seed: {run_number+42}) <<<\n')
    
        checkpoint_callback = ModelCheckpoint(
            monitor='val_macro_f1', # 强制监控 Macro-F1
            filename='best-{epoch:02d}-{val_macro_f1:.4f}',
            save_top_k=1,
            mode='max',
            verbose=True,
            save_weights_only=True
        )
    
        early_stopping_callback = EarlyStopping(
            monitor='val_macro_f1', # 强制监控 Macro-F1
            patience=Params['patience'],
            verbose=True,
            mode='max'
        )

        # 实例化模型
        model_wrapper = LitModel(Params, model_name, num_classes)

        if run_number == 0:
            num_params = count_trainable_params(model_wrapper)
            print(f'\n💡 Total Trainable Parameters: {num_params / 1e6:.4f} M\n')

        logger = TensorBoardLogger(
            save_dir=f"tb_logs/{DataName}_{exp_name}/Run_{run_number}",
            name="metrics"
        )
        csv_logger = CSVLogger(save_dir=logger.save_dir, name="csv_logs")

        # 修复了重复实例化 Trainer 的 Bug
        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=[logger, csv_logger], 
            log_every_n_steps=10,
            enable_progress_bar=progress_bar,
            accelerator='gpu',       
            devices="auto"
        )
        
        # 启动训练
        trainer.fit(model=model_wrapper, datamodule=data_module) 
        
        # 提取最佳验证集的 Macro-F1
        best_val_f1 = checkpoint_callback.best_model_score.item()
        all_val_f1s.append(best_val_f1)
    
        # 加载最佳权重进行测试
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
        )
    
        test_results = trainer.test(model=best_model, datamodule=data_module)
        # 获取测试集的 Macro-F1
        best_test_f1 = test_results[0]['test_macro_f1']
        all_test_f1s.append(best_test_f1)
    
        # 记录单次运行结果
        results_filename = f"tb_logs/{DataName}_{exp_name}/Run_{run_number}/metrics.txt"
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number} (Seed {run_number+42}):\n\n")
            file.write(f"Best Validation Macro-F1: {best_val_f1:.4f}\n")
            file.write(f"Best Test Macro-F1: {best_test_f1:.4f}\n\n")
    
    # 计算均值和标准差
    overall_avg_val_f1 = np.mean(all_val_f1s)
    overall_std_val_f1 = np.std(all_val_f1s)
    overall_avg_test_f1 = np.mean(all_test_f1s)
    overall_std_test_f1 = np.std(all_test_f1s)
    
    # 记录汇总结果
    summary_filename = f"tb_logs/{DataName}_{exp_name}/summary_metrics.txt"
    with open(summary_filename, "a") as file:
        file.write("=== Overall Results Across All 3 Runs ===\n\n")
        file.write(f"Overall Average Best Validation Macro-F1: {overall_avg_val_f1:.4f} ± {overall_std_val_f1:.4f}\n")
        file.write(f"Overall Average Best Test Macro-F1: {overall_avg_test_f1:.4f} ± {overall_std_test_f1:.4f}\n\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Advanced UATR HTAN Experiments')
    parser.add_argument('--model', type=str, default='HTAN', help='Select baseline model architecture')
    parser.add_argument('--data_selection', type=int, default=1, help='Dataset selection: 1=ShipsEar')
    
    # -------------------------------------------------------------
    # 💡 核心消融实验开关 (Ablation Switches)
    # 使用 --no-[开关名] 来关闭该模块。例如: --no-use_prior_mask
    # -------------------------------------------------------------
    parser.add_argument('--use_graph', default=True, action=argparse.BooleanOptionalAction, help='是否使用图网络模块')
    parser.add_argument('--use_prior_mask', default=True, action=argparse.BooleanOptionalAction, help='是否使用谐波物理掩码')
    parser.add_argument('--use_temporal_encoder', default=True, action=argparse.BooleanOptionalAction, help='是否使用 BiGRU 时序编码器')
    parser.add_argument('--use_temporal_attention', default=True, action=argparse.BooleanOptionalAction, help='是否使用帧级注意力池化')
    
    # 训练超参数
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='input batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train each model for')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for Dataloader')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # 特征参数
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
    
    # 将 args 转化为字典并注入给 Parameters
    # 假设你的 Demo_Parameters.py 中 Parameters 类支持直接读取 params_dict
    # 或者我们在 main 函数中直接手动更新 Params
    params = Parameters(args)
    
    # 强制将消融开关同步进 Params 字典供 LitModel 读取
    params_dict = vars(args)
    for key in ['use_graph', 'use_prior_mask', 'use_temporal_encoder', 'use_temporal_attention']:
        params[key] = params_dict[key]
        
    main(params)