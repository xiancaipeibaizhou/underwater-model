#!/bin/bash

# ==============================================================================
# HTAN 最终绝杀实验：固定划分 + 动态时间戳防覆盖 + 加载推理
# ==============================================================================

# 🌟 生成统一的实验时间戳 (例如: 20260314_103000)
EXP_TIME=$(date +"%Y%m%d_%H%M%S")

echo "----------------------------------------------------------------------"
echo "🌟 本次实验全局时间戳: ${EXP_TIME}"
echo "🌟 所有结果将保存在: results/ShipsEar_${EXP_TIME}/"
echo "----------------------------------------------------------------------"

echo "【Phase 1: 训练 Clean 基线与补全消融】 (复用已有 shipsear_data_split.json)"
echo "----------------------------------------------------------------------"

echo "▶️ [1/4] 训练 最简基线 (G0_P0_TE0_TA0_Clean)"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 0 --use_prior_mask 0 --use_temporal_encoder 0 --use_temporal_attention 0 

echo "▶️ [2/4] 训练 完整架构 (G1_P1_TE1_TA1_Clean)"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 1 --use_prior_mask 1 --use_temporal_encoder 1 --use_temporal_attention 1 

echo "▶️ [3/4] 训练 验证 TA 独立作用 (G0_P0_TE1_TA1_Clean)"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 0 --use_prior_mask 0 --use_temporal_encoder 1 --use_temporal_attention 1 

echo "▶️ [4/4] 训练 验证 TA 在完整架构作用 (G1_P1_TE1_TA0_Clean)"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 1 --use_prior_mask 1 --use_temporal_encoder 1 --use_temporal_attention 0 


echo "----------------------------------------------------------------------"
echo "【Phase 2: 恶劣环境鲁棒性生死局】 (仅推理！严格加载 Phase 1 权重)"
echo "----------------------------------------------------------------------"

# 🌟 动态拼接刚才训练好的带时间戳的权重路径
CKPT_SIMPLEST="results/ShipsEar_${EXP_TIME}/G0_P0_TE0_TA0_Clean/Run_0/best_model.ckpt"
CKPT_FULL="results/ShipsEar_${EXP_TIME}/G1_P1_TE1_TA1_Clean/Run_0/best_model.ckpt"

echo "▶️ [5/8] 10dB 噪声攻击 -> 测试 最简基线"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 0 --use_prior_mask 0 --use_temporal_encoder 0 --use_temporal_attention 0 \
    --test_only --ckpt_path $CKPT_SIMPLEST --test_snr 10.0

echo "▶️ [6/8] 10dB 噪声攻击 -> 测试 完整架构"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 1 --use_prior_mask 1 --use_temporal_encoder 1 --use_temporal_attention 1 \
    --test_only --ckpt_path $CKPT_FULL --test_snr 10.0

echo "▶️ [7/8] 0dB 极限攻击 -> 测试 最简基线"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 0 --use_prior_mask 0 --use_temporal_encoder 0 --use_temporal_attention 0 \
    --test_only --ckpt_path $CKPT_SIMPLEST --test_snr 0.0

echo "▶️ [8/8] 0dB 极限攻击 -> 测试 完整架构"
python demo_light.py --exp_time ${EXP_TIME} --model HTAN --use_graph 1 --use_prior_mask 1 --use_temporal_encoder 1 --use_temporal_attention 1 \
    --test_only --ckpt_path $CKPT_FULL --test_snr 0.0

echo "✅ 实验闭环完成！请查阅 htan_ablations_results.csv 提取定稿数据！"