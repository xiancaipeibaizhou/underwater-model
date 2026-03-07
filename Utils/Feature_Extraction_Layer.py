# Feature_Extraction_Layer.py

import torch.nn as nn
import torch
import torchaudio.transforms as T
from .LogMelFilterBank import MelSpectrogramExtractor  

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=4096, 
                 hop_length=512, number_mels=64, segment_length=5):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.sample_frequency = sample_rate 
        self.num_channels = 1
        self.input_feature = input_feature
        
        # Initialize logmelfbank
        win_length = window_length
        n_fft = window_length
        hop_length = hop_length 
        n_mels = number_mels
        fmin = 1
        fmax = 8000
        
        self.LogMelFBank = MelSpectrogramExtractor(
            sample_rate=sample_rate, 
            n_fft=n_fft,
            win_length=win_length, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmin=fmin, 
            fmax=fmax
        )

        self.features = {'LogMelFBank': self.LogMelFBank}
        
        # =====================================================================
        # >>> 核心创新：物理声谱掩码数据增强 (SpecAugment) <<<
        # 强制增加训练难度，逼迫 PhysicalHarmonicGCN 发挥谐波推导与时序重建的作用
        # =====================================================================
        # 频率掩码：随机遮蔽最高 24 个梅尔频带 (模拟 LOFAR 线谱被海洋宽带噪声淹没)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=24)
        # 时间掩码：随机遮蔽最高 30 个时间帧 (模拟螺旋桨节拍信号在远距离传播中的衰落缺失)
        self.time_masking = T.TimeMasking(time_mask_param=30)
        # =====================================================================
                
        self.output_dims = None
        self.calculate_output_dims()

    def calculate_output_dims(self):
        try:
            length_in_seconds = self.segment_length  
            samples = int(self.sample_rate * length_in_seconds)
            dummy_input = torch.randn(1, samples)  
            with torch.no_grad():
                output = self.features[self.input_feature](dummy_input)
                self.output_dims = output.shape
        except Exception as e:
            print(f"Failed to calculate output dimensions: {e}\n")
            self.output_dims = None
            
    def forward(self, x):
        # 提取基础对数梅尔声谱图: [Batch, Freq, Time]
        x = self.features[self.input_feature](x) 

        # 仅在模型处于训练模式 (Training) 时，施加残酷的物理掩码
        # 验证集和测试集 (Eval) 必须看清晰的完整图像
        if self.training:
            x = self.freq_masking(x)
            x = self.time_masking(x)

        x = x.unsqueeze(1) # 增加通道维度 -> [Batch, 1, Freq, Time]

        return x