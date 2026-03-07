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

# 【精简】：只保留 ShipsEar 水声数据集相关的核心数据模块
from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(Params):
    model_name = Params['Model_name']
    numBins = Params['numBins']
    RR = Params['RR']
    h_shared = Params['histograms_shared']
    a_shared = Params['adapters_shared']
    
    batch_size = Params['batch_size']
    t_batch_size = batch_size['train']
    num_workers = Params['num_workers']
    
    run_number = 0
    seed_everything(run_number+1, workers=True)
    
    # 【核心瘦身】：移除了其余无用数据集的分支，默认且强制运行 ShipsEar
    if Params['data_selection'] == 1:
        # 指向我们刚刚完成物理分类贴标签并切片完成的目录
        dataset_dir = 'shipsEar_AUDIOS/' 
        
        # 这一步会执行安全检查，如果已经切片完成，它会自动跳过，不会重复消耗算力
        Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)
        
        # 加载防泄露的 Train/Val/Test 数据流
        data_module = ShipsEarDataModule(parent_folder=dataset_dir, batch_size=batch_size, num_workers=num_workers)
        num_classes = 5 # A, B, C, D, E 五大类
    else:
        raise ValueError('代码已进行极简优化，当前仅支持 ShipsEar 数据集 (请确保参数 --data_selection 1)')
    
    DataName = "ShipsEar"
    print('\nStarting Experiments for ' + DataName)
    
    numRuns = 3
    progress_bar = True # 建议开启进度条，方便观察每个 Epoch 的训练速度
    
    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    all_test_accs = []
    
    for run_number in range(0, numRuns):
        if run_number != 0:
            seed_everything(run_number+1, workers=True)
                  
        print(f'\nStarting Run {run_number}\n')
    
        checkpoint_callback = ModelCheckpoint(
            monitor='val_auprc',
            filename='best-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max',
            verbose=True,
            save_weights_only=True
        )
    
        early_stopping_callback = EarlyStopping(
            monitor='val_auprc',
            patience=Params['patience'],
            verbose=True,
            mode='max'
        )

        # 实例化大模型
        model_AST = LitModel(Params, model_name, num_classes, numBins, RR)

        num_params = count_trainable_params(model_AST)
        print(f'Total Trainable Parameters: {num_params}')

        logger = TensorBoardLogger(
            save_dir = (
                f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
                f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
                f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
                f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/Run_{run_number}"
            ),
            name="metrics"
        )

        csv_logger = CSVLogger(save_dir=logger.save_dir, name="csv_logs")

        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=[logger, csv_logger],  # <--- 修改这里：同时传入两个 logger
            log_every_n_steps=10,
            enable_progress_bar=progress_bar,
            accelerator='gpu',       
            devices="auto"
        )

        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=logger,
            log_every_n_steps=10,
            enable_progress_bar=progress_bar,
            accelerator='gpu',       
            devices="auto"
        )
        
        # 启动训练
        trainer.fit(model=model_AST, datamodule=data_module) 
        
        best_val_acc = checkpoint_callback.best_model_score.item()
        all_val_accs.append(best_val_acc)
    
        # 训练完成后加载最佳权重进行测试
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
        )
    
        test_results = trainer.test(model=best_model, datamodule=data_module)
        best_test_acc = test_results[0]['test_acc']
        all_test_accs.append(best_test_acc)
    
        results_filename = (
            f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
            f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
            f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
            f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/Run_{run_number}/metrics.txt"
        )

        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}:\n\n")
            file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            file.write(f"Best Test Accuracy: {best_test_acc:.4f}\n\n")
    
    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
    overall_avg_test_acc = np.mean(all_test_accs)
    overall_std_test_acc = np.std(all_test_accs)
    
    summary_filename = (
            f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
            f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
            f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
            f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/summary_metrics.txt"
        )

    with open(summary_filename, "a") as file:
        file.write("Overall Results Across All Runs\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Run Advanced UATR AST Experiments')
    parser.add_argument('--model', type=str, default='AST', help='Select baseline model architecture')
    parser.add_argument('--histograms_shared', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--adapters_shared', default=True, action=argparse.BooleanOptionalAction)
    
    # 强制保留数据选择参数，但默认只支持 ShipsEar(1)
    parser.add_argument('--data_selection', type=int, default=1, help='Dataset selection: 1=ShipsEar')
    
    parser.add_argument('-numBins', type=int, default=16, help='Number of bins for histogram layer.')
    parser.add_argument('-RR', type=int, default=128, help='Adapter Reduction Rate (default: 128)')
    parser.add_argument('--train_mode', type=str, default='lora', help='full_fine_tune or linear_probing or adapters or histogram or lora or ssf')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='input batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train each model for')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for Dataloader')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank', help='Audio feature for extraction')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--window_length', type=int, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, default=512, help='hop length')
    parser.add_argument('--number_mels', type=int, default=128, help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Dataset Sample Rate')
    parser.add_argument('--segment_length', type=int, default=5, help='Dataset Segment Length')
    parser.add_argument('--adapter_location', type=str, default='None')
    parser.add_argument('--adapter_mode', type=str, default='None')
    parser.add_argument('--histogram_location', type=str, default='None')
    parser.add_argument('--histogram_mode', type=str, default='None')
    parser.add_argument('--lora_target', type=str, default='q')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_shared', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bias_mode', type=str, default='full')   
    parser.add_argument('--ssf_shared', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--ssf_mode', type=str, default='full')                  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
# python demo_light.py --data_selection 1 --train_mode lora --train_batch_size 32
