# Network_functions.py

from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from src.models.ast_base import ASTBase
from src.models.ast_linear_probe import ASTLinearProbe
from src.models.ast_adapter import ASTAdapter
from src.models.ast_histogram import ASTHistogram
from src.models.ast_bias import ASTBias
from src.models.ast_lora import ASTLoRA
from src.models.ast_ssf import ASTSSF

# 🌟 1. 导入你的 HTAN 模型
from src.models.custom_model import HTAN

def initialize_model(model_name, num_classes, numBins, RR, sample_rate=16000,segment_length=5,
                     t_mode='full_fine_tune', h_shared=True, a_shared=True,
                     parallel=True, input_feature='STFT',
                     window_length=512, hop_length=256, number_mels=64,
                     adapter_location='ffn', adapter_mode='parallel', 
                     histogram_location='ffn', histogram_mode='parallel',
                     lora_target='q', lora_rank=4, r_shared=False, b_mode='full',
                     ssf_shared=False, ssf_mode='full', skip_feature=False, image_input_hw=(96, 96),
                     **kwargs): # 🌟 2. 添加 **kwargs 来接收外部可能传来的消融实验参数
    
    if not skip_feature:
        feature_layer = Feature_Extraction_Layer(
            input_feature=input_feature, sample_rate=sample_rate, segment_length=segment_length,
            window_length=window_length, hop_length=hop_length, number_mels=number_mels
        )
        ft_dims = feature_layer.output_dims
        inpf, inpt = ft_dims[1], ft_dims[2]
        print(f'feature shape f by t: {inpf} by {inpt}')
    else:
        feature_layer = None
        inpf, inpt = image_input_hw
        print(f'skipping feature extraction; using input HxW = {inpf}x{inpt}')
        
    # Initialize the appropriate model
    
    # 🌟 3. 核心修改：在此处拦截模型名称，如果是 HTAN 则实例化它
    if model_name == 'HTAN':
        print("\n🚀 Initializing HTAN (HarmonicFrequencyGCN) Model...")
        
        # 安全提取消融实验开关，如果没有传入则默认设为 True
        use_graph = kwargs.get('use_graph', True)
        use_prior_mask = kwargs.get('use_prior_mask', True)
        use_temporal_encoder = kwargs.get('use_temporal_encoder', True)
        use_temporal_attention = kwargs.get('use_temporal_attention', True)
        
        model_ft = HTAN(
            num_classes=num_classes,
            in_channels=1,
            input_fdim=inpf,
            input_tdim=inpt,
            use_graph=use_graph,
            use_prior_mask=use_prior_mask,
            use_temporal_encoder=use_temporal_encoder,
            use_temporal_attention=use_temporal_attention
        )
        
    # ============ 下面是原本的 AST 系列模型逻辑 ============
    elif t_mode == 'full_fine_tune':
        model_ft = ASTBase(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt)
    elif t_mode == 'linear_probing':
        model_ft = ASTLinearProbe(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt)
    elif t_mode == 'adapters':
        model_ft = ASTAdapter(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                              RR=RR, adapter_location=adapter_location,
                              adapter_shared=a_shared, adapter_mode=adapter_mode)
    elif t_mode == 'histogram':
        model_ft = ASTHistogram(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                                NumBins=numBins, histogram_location=histogram_location,
                                hist_shared=h_shared, histogram_mode=histogram_mode)
    elif t_mode == 'bias':                            
            model_ft = ASTBias(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt, bias_mode=b_mode) #choose 'full', 'query', 'mlp'
        
    elif t_mode == 'lora':    
        model_ft = ASTLoRA(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                           lora_shared=r_shared, lora_rank=lora_rank, lora_update_mode=lora_target)

    elif t_mode == 'ssf':                            
            model_ft = ASTSSF(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt, ssf_shared=ssf_shared, ssf_mode=ssf_mode) #choose 'full', 'mhsa_only'
     
    else:
        raise ValueError(f"Unknown training mode: {t_mode} or model: {model_name}")

    print("\n\nSettings:")
    print(f"Model selected: {model_name}")
    print(f"Training mode: {t_mode}")
    
    if model_name != 'HTAN':
        print(f"Adapter mode: {adapter_mode}")
        print(f"Adapter location: {adapter_location}")
        print(f"Adapter shared: {a_shared}")
        print(f"Histogram mode: {histogram_mode}")
        print(f"Histogram location: {histogram_location}")
        print(f"Histogram shared: {h_shared}")
        print(f"lora: {lora_target}, rank: {lora_rank}, shared: {r_shared}")
        print(f"ssf_shared: {ssf_shared}, ssf_mode: {ssf_mode}\n")
    else:
        print(f"use_graph: {kwargs.get('use_graph', True)}")
        print(f"use_prior_mask: {kwargs.get('use_prior_mask', True)}")
        print(f"use_temporal_encoder: {kwargs.get('use_temporal_encoder', True)}")
        print(f"use_temporal_attention: {kwargs.get('use_temporal_attention', True)}\n")
    
    # Set requires_grad for parameters (保证 HTAN 也能被求导训练)
    if t_mode == 'full_fine_tune' or model_name == 'HTAN':
        for param in model_ft.parameters():
            param.requires_grad = True
            
    return model_ft, feature_layer