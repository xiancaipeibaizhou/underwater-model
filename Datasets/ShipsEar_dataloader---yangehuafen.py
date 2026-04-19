# 严格划分

import os
import json
import collections
from datetime import datetime
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import hashlib  # 🌟 新增：用于生成固定的路径哈希种子
import random  # 🌟 新增：用于样本级随机种子
# 🌟 新增：严谨的样本级确定性 AWGN 噪声注入函数

def add_awgn(signal, snr_db, seed_string=None):
    """根据目标信噪比 (SNR) 注入绝对固定的高斯白噪声"""
    if snr_db is None:
        return signal
        
    sig_power = np.mean(signal ** 2)
    if sig_power == 0:
        return signal
        
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    # 强制固定随机种子，确保同一个文件无论测多少次，加的噪声波形完全一致
    if seed_string is not None:
        seed = int(hashlib.md5(seed_string.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # 生成 float32 格式的噪声，对齐原信号
    noise = rng.normal(0, np.sqrt(noise_power), len(signal)).astype(np.float32)
    return signal + noise


class ShipsEarDataset(Dataset):
    # 🌟 修改 1：接收 snr_db 参数
    def __init__(self, segment_list, target_sr=16000, normalize_waveform=False, snr_db=None, is_ssl=False):
        self.segment_list = segment_list
        self.target_sr = target_sr
        self.normalize_waveform = normalize_waveform
        self.snr_db = snr_db
        self.is_ssl = is_ssl

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        file_path, label = self.segment_list[idx]
        
        try:
            sample_rate, signal = wavfile.read(file_path)
        except Exception as e:
            raise RuntimeError(f"🚨 读取音频文件失败: {file_path}. 详情: {e}")

        if sample_rate != self.target_sr:
            raise ValueError(f"🚨 采样率异常！期望 {self.target_sr}Hz, 但文件是 {sample_rate}Hz。")
        
        if len(signal) == 0:
            raise ValueError(f"🚨 发现空音频文件: {file_path}")

        signal = signal.astype(np.float32)
        
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        # 🌟 关键修改：如果是预训练模式，返回双视图！
        if getattr(self, 'is_ssl', False):
            # 视图 1：随机波形增益 + 随机动态高斯白噪声
            gain1 = random.uniform(0.7, 1.3)
            sig_v1 = signal.copy() * gain1
            sig_v1 = add_awgn(sig_v1, snr_db=random.uniform(15, 30), seed_string=None) 
            
            # 视图 2：另一种随机增益 + 另一种强度的动态噪声
            gain2 = random.uniform(0.7, 1.3)
            sig_v2 = signal.copy() * gain2
            sig_v2 = add_awgn(sig_v2, snr_db=random.uniform(10, 25), seed_string=None)
            
            if self.normalize_waveform:
                max_v1 = np.max(np.abs(sig_v1))
                if max_v1 > 0: sig_v1 = sig_v1 / max_v1
                max_v2 = np.max(np.abs(sig_v2))
                if max_v2 > 0: sig_v2 = sig_v2 / max_v2
                
            # 【核心对齐】必须返回严格的嵌套元组：((视图1, 视图2), 占位标签)
            return (torch.tensor(sig_v1, dtype=torch.float), torch.tensor(sig_v2, dtype=torch.float)), torch.tensor(-1, dtype=torch.long)

        # 常规监督模式 (微调 / 测试)
        else:
            signal = add_awgn(signal, self.snr_db, seed_string=file_path)

            if self.normalize_waveform:
                max_val = np.max(np.abs(signal))
                if max_val > 0:
                    signal = signal / max_val

            return torch.tensor(signal, dtype=torch.float), torch.tensor(label, dtype=torch.long)

class ShipsEarDataModule(L.LightningDataModule):
    # 🌟 修改 3：接收外部传来的 test_snr
    def __init__(self, parent_folder='./Datasets/ShipsEar', batch_size=None, num_workers=8,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42, 
                 normalize_waveform=False, split_file='shipsear_data_split.json', audit_file='split_audit_report.json',
                 test_snr=None, is_ssl=False):
        super().__init__()
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "🚨 比例之和必须等于 1.0"
        
        self.batch_size = batch_size or {'train': 64, 'val': 64, 'test': 64}
        self.parent_folder = parent_folder
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.normalize_waveform = normalize_waveform
        self.split_file = split_file
        self.audit_file = audit_file
        self.test_snr = test_snr # 保存测试集 SNR 配置
        self.is_ssl = is_ssl

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _verify_and_load_split(self, current_class_mapping):
        if not os.path.exists(self.split_file):
            return None
            
        try:
            with open(self.split_file, 'r') as f:
                data = json.load(f)
                
            meta = data.get('metadata', {})
            
            if meta.get('random_seed') != self.random_seed: return None
            if meta.get('train_ratio') != self.train_ratio: return None
            if meta.get('val_ratio') != self.val_ratio: return None
            if meta.get('test_ratio') != self.test_ratio: return None
            if meta.get('parent_folder') != self.parent_folder: return None
            if meta.get('class_mapping') != current_class_mapping: return None
            
            print(f"✅ 校验通过：成功复用历史切分文件 ({meta.get('timestamp')})")
            return data.get('folder_lists')
            
        except Exception as e:
            print(f"⚠️ 解析切分文件失败 ({e})，将重新生成...")
            return None

    def save_splits(self, folder_lists, class_mapping):
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "protocol": "Class-wise Recording-level Split",
                "random_seed": self.random_seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "parent_folder": self.parent_folder,
                "class_mapping": class_mapping,
                "shuffle": True 
            },
            "folder_lists": folder_lists
        }
        with open(self.split_file, 'w') as f:
            json.dump(data, f, indent=4)

    def check_data_leakage(self, folder_lists, segment_lists):
        splits = ['train', 'val', 'test']
        recordings = {split: set() for split in splits}
        segments = {split: set() for split in splits}

        for split in splits:
            for folder_path, _ in folder_lists[split]:
                recordings[split].add(os.path.abspath(folder_path))
            for file_path, _ in segment_lists[split]:
                segments[split].add(os.path.abspath(file_path))

        if recordings['train'].intersection(recordings['val']) or \
           recordings['train'].intersection(recordings['test']) or \
           recordings['val'].intersection(recordings['test']):
            raise ValueError("🚨 数据泄漏！Train/Val/Test 之间存在 Recording 级别的重叠！")

        if segments['train'].intersection(segments['val']) or \
           segments['train'].intersection(segments['test']) or \
           segments['val'].intersection(segments['test']):
            raise ValueError("🚨 数据泄漏！Train/Val/Test 之间存在 Segment 绝对路径的重叠！")

    def _print_and_verify_distributions(self, folder_lists, segment_lists, inverse_class_mapping, class_mapping):
        splits = ['train', 'val', 'test']
        num_classes = len(inverse_class_mapping)
        
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "class_mapping": class_mapping,
            "splits": {}
        }
        
        print("\n" + "="*75)
        print("📊 数据集全局划分与类分布审计报告 (Recording & Segment Level)")
        print("="*75)
        print(f"🗺️  类别映射: {class_mapping}")
        print("-" * 75)

        for split in splits:
            print(f"[{split.upper():^5} SET]")
            rec_counts = collections.Counter([label for _, label in folder_lists[split]])
            seg_counts = collections.Counter([label for _, label in segment_lists[split]])
            
            total_recs = sum(rec_counts.values())
            total_segs = sum(seg_counts.values())
            
            audit_data["splits"][split] = {
                "total_recordings": total_recs,
                "total_segments": total_segs,
                "class_distribution": {}
            }
            
            print(f"总计 -> Recordings: {total_recs:<4} | Segments: {total_segs}")
            
            for class_idx in range(num_classes):
                c_name = inverse_class_mapping[class_idx]
                c_rec = rec_counts.get(class_idx, 0)
                c_seg = seg_counts.get(class_idx, 0)
                
                if c_rec == 0 or c_seg == 0:
                    raise AssertionError(f"🚨 数据失衡: {split.upper()} 集中, 类别 [{c_name}] 数量为 0！")

                rec_pct = (c_rec / total_recs) * 100 if total_recs > 0 else 0
                seg_pct = (c_seg / total_segs) * 100 if total_segs > 0 else 0
                
                audit_data["splits"][split]["class_distribution"][c_name] = {
                    "recordings": c_rec,
                    "segments": c_seg
                }
                
                print(f" Class {c_name:<2} (ID:{class_idx}) | "
                      f"Recs: {c_rec:>3} ({rec_pct:>5.1f}%) | "
                      f"Segs: {c_seg:>4} ({seg_pct:>5.1f}%)")
            print("-" * 75)
            
        with open(self.audit_file, 'w') as f:
            json.dump(audit_data, f, indent=4)
        print(f"💾 审计报告已保存至: {self.audit_file}")
        print("="*75 + "\n")

    def setup(self, stage=None):
        ships_classes = sorted([f.name for f in os.scandir(self.parent_folder) if f.is_dir()])
        class_mapping = {ship: idx for idx, ship in enumerate(ships_classes)}
        inverse_class_mapping = {idx: ship for ship, idx in class_mapping.items()}

        folder_lists = self._verify_and_load_split(class_mapping)

        if folder_lists is None:
            print(f"🚀 正在基于比例 {self.train_ratio}:{self.val_ratio}:{self.test_ratio} 生成 Class-wise Recording-level Split...")
            folder_lists = {'train': [], 'test': [], 'val': []}

            for label in ships_classes:
                label_path = os.path.join(self.parent_folder, label)
                subfolders = sorted([f.name for f in os.scandir(label_path) if f.is_dir()])
                
                n_total = len(subfolders)
                
                n_train = int(n_total * self.train_ratio)
                n_val_test = n_total - n_train
                relative_val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
                n_val = int(n_val_test * relative_val_ratio)
                n_test = n_val_test - n_val
                
                if n_train == 0 or n_val == 0 or n_test == 0:
                    raise ValueError(
                        f"🚨 致命冲突: 类别 [{label}] 的物理录音数 ({n_total}) 无法满足 "
                        f"{self.train_ratio}:{self.val_ratio}:{self.test_ratio} 的划分！"
                    )

                subfolders_train, subfolders_val_test = train_test_split(
                    subfolders, train_size=self.train_ratio, shuffle=True, random_state=self.random_seed
                )

                subfolders_val, subfolders_test = train_test_split(
                    subfolders_val_test, train_size=relative_val_ratio, shuffle=True, random_state=self.random_seed
                )

                for subfolder in subfolders_train:
                    folder_lists['train'].append([os.path.join(label_path, subfolder), class_mapping[label]])
                for subfolder in subfolders_val:
                    folder_lists['val'].append([os.path.join(label_path, subfolder), class_mapping[label]])
                for subfolder in subfolders_test:
                    folder_lists['test'].append([os.path.join(label_path, subfolder), class_mapping[label]])
            
            self.save_splits(folder_lists, class_mapping)

        segment_lists = {'train': [], 'test': [], 'val': []}
        for split in ['train', 'test', 'val']:
            for folder_path, label in folder_lists[split]:
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith('.wav'):
                        file_path = os.path.join(folder_path, file)
                        if os.path.isfile(file_path):
                            segment_lists[split].append((file_path, label))
        
        self.check_data_leakage(folder_lists, segment_lists)

        # 🌟 修改：只给 train_dataset 开启 is_ssl 模式，验证集和测试集永远是常规模式
        self.train_dataset = ShipsEarDataset(segment_lists['train'], normalize_waveform=self.normalize_waveform, snr_db=None, is_ssl=self.is_ssl)
        self.val_dataset = ShipsEarDataset(segment_lists['val'], normalize_waveform=self.normalize_waveform, snr_db=None, is_ssl=False)
        self.test_dataset = ShipsEarDataset(segment_lists['test'], normalize_waveform=self.normalize_waveform, snr_db=self.test_snr, is_ssl=False)

        self._print_and_verify_distributions(folder_lists, segment_lists, inverse_class_mapping, class_mapping)
        
        # 🌟 修改 5：高能预警打印
        if self.test_snr is not None:
            print(f"🌪️  [高能预警] 当前测试集已注入固定随机种子的 AWGN 白噪声，信噪比 SNR = {self.test_snr} dB")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], num_workers=self.num_workers, pin_memory=True)

        