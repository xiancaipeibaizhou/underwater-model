import os
import json
import collections
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import lightning as L
from scipy.io import wavfile

class ShipsEarDataset(Dataset):
    def __init__(self, segment_list):
        self.segment_list = segment_list

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        file_path, label = self.segment_list[idx]
        
        # Load the audio signal
        sample_rate, signal = wavfile.read(file_path)
        signal = torch.tensor(signal, dtype=torch.float)
        
        label = torch.tensor(label, dtype=torch.long)

        return signal, label

class ShipsEarDataModule(L.LightningDataModule):
    # 【修改点 1】：将 val_test_split 的默认值从 1/3 改为 0.5，实现 70/15/15 划分
    def __init__(self, parent_folder='./Datasets/ShipsEar', batch_size=None, num_workers=8,
                 train_split=0.7, val_test_split=0.5, random_seed=42, shuffle=False,
                 split_file='shipsear_data_split.txt'):
        super().__init__()
        
        self.batch_size = batch_size or {'train': 64, 'val': 64, 'test': 64}

        self.parent_folder = parent_folder
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.shuffle = shuffle
        
        # File to save/load splits
        self.split_file = split_file
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def save_splits(self, folder_lists):
        """Save train/val/test splits to a text file."""
        with open(self.split_file, 'w') as f:
            for split in ['train', 'val', 'test']:
                f.write(f"{split}:\n")
                for folder_path, label in folder_lists[split]:
                    f.write(f"{folder_path},{label}\n")
                    
    def load_splits(self):
        """Load train/val/test splits from a text file."""
        folder_lists = {'train': [], 'val': [], 'test': []}
        
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file {self.split_file} does not exist!")
        
        with open(self.split_file, 'r') as f:
            current_split = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_split = line[:-1]
                else:
                    folder_path, label = line.split(',')
                    folder_lists[current_split].append((folder_path, int(label)))
                    
        return folder_lists

    # 【修改点 2】：新增专属的审计报告生成器
    def generate_split_audit(self, folder_lists, segment_lists):
        """
        生成并保存严格的数据划分审计报告 (JSON)，一旦发现泄漏立即抛出异常。
        """
        audit_report = {
            "protocol": "Stratified Recording-Level Split (Strict Non-overlapping)",
            "seed_used": self.random_seed,
            "train": {"recordings_count": len(folder_lists['train']), "segments_per_class": collections.defaultdict(int)},
            "val": {"recordings_count": len(folder_lists['val']), "segments_per_class": collections.defaultdict(int)},
            "test": {"recordings_count": len(folder_lists['test']), "segments_per_class": collections.defaultdict(int)},
            "leakage_check": "PASSED"
        }

        # 统计各集合中的类别切片分布
        for split in ['train', 'val', 'test']:
            for file_path, label in segment_lists[split]:
                audit_report[split]["segments_per_class"][f"Class_{label}"] += 1

        # 致命的数据泄漏校验 (Recording-level 物理隔离检查)
        train_folders = set([f[0] for f in folder_lists['train']])
        val_folders = set([f[0] for f in folder_lists['val']])
        test_folders = set([f[0] for f in folder_lists['test']])

        if train_folders.intersection(val_folders) or train_folders.intersection(test_folders) or val_folders.intersection(test_folders):
            audit_report["leakage_check"] = "FAILED: Overlapping recordings detected across splits!"
            print("🚨 致命错误：检测到数据泄漏！同一个录音文件出现在了不同的集合中！")
            
            # 必须导出泄漏报告留存证据，然后强行中断
            with open('split_audit_report.json', 'w') as f:
                json.dump(audit_report, f, indent=4)
            raise ValueError("Data Leakage Detected at Recording Level! Please check your split generation.")
        else:
            print("✅ 数据划分审计通过：Train/Val/Test 实现了严格的录音级别物理隔离。")

        # 正常导出合规的审计报告
        with open('split_audit_report.json', 'w') as f:
            json.dump(audit_report, f, indent=4)

    def setup(self, stage=None):
        # Check if split file exists and load it if available
        if os.path.exists(self.split_file):
            print(f"\nLoading splits from {self.split_file}")
            folder_lists = self.load_splits()
        
        else:
            print(f"Creating new splits and saving them to {self.split_file}")
            
            # Get classes in directory (A, B, C, D, E)
            ships_classes = [f.name for f in os.scandir(self.parent_folder) if f.is_dir()]
            class_mapping = {ship: idx for idx, ship in enumerate(ships_classes)}
            folder_lists = {'train': [], 'test': [], 'val': []}

            # Loop over each class and split data into train/val/test sets at recording level (subfolder level)
            for label in ships_classes:
                label_path = os.path.join(self.parent_folder, label)
                subfolders = os.listdir(label_path)

                # Split subfolders into training and test/validation sets
                subfolders_train, subfolders_test_val = train_test_split(
                    subfolders,
                    train_size=self.train_split,
                    shuffle=self.shuffle,
                    random_state=self.random_seed,
                )

                # Split test/validation set further into test and validation sets
                subfolders_test, subfolders_val = train_test_split(
                    subfolders_test_val,
                    test_size=self.val_test_split,
                    shuffle=self.shuffle,
                    random_state=self.random_seed,
                )

                # Add subfolders to appropriate folder list with their class labels
                for subfolder in subfolders_train:
                    folder_lists['train'].append((os.path.join(label_path, subfolder), class_mapping[label]))

                for subfolder in subfolders_test:
                    folder_lists['test'].append((os.path.join(label_path, subfolder), class_mapping[label]))

                for subfolder in subfolders_val:
                    folder_lists['val'].append((os.path.join(label_path, subfolder), class_mapping[label]))
            
            # Save splits to a text file for future use.
            self.save_splits(folder_lists)

        segment_lists = {'train': [], 'test': [], 'val': []}
        recording_counts = {split: len(folder_lists[split]) for split in ['train', 'val', 'test']}
        total_recordings = sum(recording_counts.values())
        
        # Loop over each partition and gather all segments (files) within each folder
        for split in ['train', 'test', 'val']:
            for folder_path, label in folder_lists[split]:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            segment_lists[split].append((file_path, label))
        
        # 【修改点 3】：在这里调用审计生成器 (紧接在 segment_lists 填充完毕后，实例化 Dataset 之前)
        self.generate_split_audit(folder_lists, segment_lists)

        # Initialize datasets
        self.train_dataset = ShipsEarDataset(segment_lists['train'])
        self.val_dataset = ShipsEarDataset(segment_lists['val'])
        self.test_dataset = ShipsEarDataset(segment_lists['test'])

        total_samples = (len(self.train_dataset) +len(self.val_dataset) +len(self.test_dataset))
                         
        print(f"\nNumber of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        print(f"Number of test samples: {len(self.test_dataset)}\n")
        
        print(f"\nRecording folders  – train: {recording_counts['train']}, "
              f"val: {recording_counts['val']}, "
              f"test: {recording_counts['test']}, "
              f"total: {total_recordings}")

        print(f"Total number of samples across all splits: {total_samples}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size['val'],
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size['test'],
            num_workers=self.num_workers
        )