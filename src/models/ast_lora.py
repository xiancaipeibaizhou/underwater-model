import torch
import torch.nn as nn
import os
import math  
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

# 导入我们刚刚创建的物理图卷积网络
from src.models.ast_graph import PhysicalHarmonicGCN

# ------------------------------------------------------------
# LoRA_qkv Module with Optional "q" or "qv" Update Modes
# ------------------------------------------------------------
class LoRA_qkv(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 4, alpha: float = 1.0, update_mode: str = "qv"):
        super(LoRA_qkv, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        assert self.out_features % 3 == 0, "Expected out_features to be divisible by 3"
        self.dim = self.out_features // 3  
        self.r = r
        self.alpha = alpha
        self.update_mode = update_mode.lower()
        if self.update_mode not in ["q", "qv"]:
            raise ValueError("update_mode must be either 'q' or 'qv'")
        self.scaling = alpha / r if r > 0 else 0

        self.register_buffer("weight", linear_layer.weight.detach().clone())
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.detach().clone())
        else:
            self.bias = None

        if r > 0:
            self.lora_A_q = nn.Parameter(torch.zeros(self.dim, r))
            self.lora_B_q = nn.Parameter(torch.zeros(r, self.in_features))
            nn.init.kaiming_uniform_(self.lora_B_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A_q)
            if self.update_mode == "qv":
                self.lora_A_v = nn.Parameter(torch.zeros(self.dim, r))
                self.lora_B_v = nn.Parameter(torch.zeros(r, self.in_features))
                nn.init.kaiming_uniform_(self.lora_B_v, a=math.sqrt(5))
                nn.init.zeros_(self.lora_A_v)
            else:
                self.lora_A_v = None
                self.lora_B_v = None
        else:
            self.lora_A_q = None
            self.lora_B_q = None
            self.lora_A_v = None
            self.lora_B_v = None

    def forward(self, x):
        original_out = torch.nn.functional.linear(x, self.weight, self.bias) 
        if self.r > 0:
            delta_q = (x @ self.lora_B_q.T) @ self.lora_A_q.T  
            delta_q = delta_q * self.scaling
            out = original_out.clone()
            out[..., :self.dim] += delta_q  
            
            if self.update_mode == "qv":
                delta_v = (x @ self.lora_B_v.T) @ self.lora_A_v.T  
                delta_v = delta_v * self.scaling
                out[..., 2*self.dim:3*self.dim] += delta_v  
            return out
        else:
            return original_out

# ------------------------------------------------------------
# PatchEmbed: override timm's input shape restriction.
# ------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class ASTLoRA(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True, 
                 lora_shared=True, lora_rank=4, lora_update_mode="qv"):
        super(ASTLoRA, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5'

        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
        
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach() \
                    .reshape(1, self.original_num_patches, self.original_embedding_dim) \
                    .transpose(1, 2) \
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

            self.physical_gcn = PhysicalHarmonicGCN(in_channels=self.original_embedding_dim, num_freq_bins=f_dim, num_time_steps=t_dim)
            self.fusion_weight = nn.Parameter(torch.tensor(0.1))

        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('Currently a model pretrained only on AudioSet is not supported, please set imagenet_pretrain=True.')
            if model_size != 'base384':
                raise ValueError('Currently only the base384 AudioSet pretrained model is supported.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model_path = 'pretrained_models/audioset_0.4593.pth'
            if not os.path.exists(model_path):
                raise Exception(f"预训练模型文件不存在！请手动下载模型并放到 {model_path} 路径下。")
            
            sd = torch.load(model_path, map_location=device, weights_only=True)
            audio_model = ASTLoRA(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                                     imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")
            
            if lora_shared:
                shared_lora_qkv = None
                for blk in self.v.blocks:
                    if shared_lora_qkv is None:
                        shared_lora_qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank, alpha=1.0, update_mode=lora_update_mode)
                    blk.attn.qkv = shared_lora_qkv
            else:
                for blk in self.v.blocks:
                    blk.attn.qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank, alpha=1.0, update_mode=lora_update_mode)
   
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))
    
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach() \
                .reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            # >>> 实例化图网络模块和融合权重 (基于动态计算出的 F_dim 和 T_dim) <<<
            self.physical_gcn = PhysicalHarmonicGCN(in_channels=self.original_embedding_dim, num_freq_bins=f_dim, num_time_steps=t_dim)
            self.fusion_weight = nn.Parameter(torch.tensor(0.1)) # 刚才漏掉了这句

            self.freeze_base_model()
        
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def freeze_base_model(self):
        for param in self.v.parameters():
            param.requires_grad = False
        
        for blk in self.v.blocks:
            for name, param in blk.attn.qkv.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        
        for param in self.mlp_head.parameters():
            param.requires_grad = True

        # >>> 必须解冻图网络层，让谐波和时序注意力机制能够被训练！ <<<
        for param in self.physical_gcn.parameters():
            param.requires_grad = True
            
        # >>> 解冻门控融合权重 (刚才漏掉了这句) <<<
        self.fusion_weight.requires_grad = True
        
        print("Base model frozen. Only LoRA, Classifier, Physical GCN, and Fusion Gate are trainable.")
    
    def forward(self, x):
        B = x.shape[0]

        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            residual = x
            x = blk.norm1(x)
            x = blk.attn(x)   
            x = residual + x
            x = blk.drop_path(x)

            residual = x
            x = blk.norm2(x)
            x = blk.mlp(x)
            x = residual + x
            x = blk.drop_path(x)

        # -----------------------------------------------------------------
        # >>> 核心前向逻辑注入：使用图网络增强特征，并与全局特征融合 <<<
        # -----------------------------------------------------------------
        cls_dist_tokens = x[:, :2, :] 
        patch_tokens = x[:, 2:, :]    

        patch_tokens = self.physical_gcn(patch_tokens)

        # global_feat = (cls_dist_tokens[:, 0] + cls_dist_tokens[:, 1]) / 2
        # graph_feat = patch_tokens.mean(dim=1)  
        
        # # 使用门控机制！融合权重被 sigmoid 限制在 0-1 之间
        # beta = torch.sigmoid(self.fusion_weight)
        # x = (1.0 - beta) * global_feat + beta * graph_feat
        # # -----------------------------------------------------------------

        # x = self.mlp_head(x)
        graph_feat = patch_tokens.mean(dim=1)  
        
        x = self.mlp_head(graph_feat)
        return x