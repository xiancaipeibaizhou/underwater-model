#!/bin/bash

# ==============================================================================
# HTAN 核心消融与基线实验自动运行脚本 (For RTX 4090)
# 脚本将依次运行 4 个模型，所有结果会自动汇总到 htan_ablations_results.csv
# ==============================================================================

# 设置统一的超参数
BATCH_SIZE=64
LR=1e-3

echo "🚀 [启动] 开始执行 HTAN 论文级核心实验..."
echo "📊 最终结果将自动汇总至: htan_ablations_results.csv"
echo "----------------------------------------------------------------------"

# 1. 跑你们的杀手锏：HTAN Full (完整物理启发模型)
echo "▶️ [1/4] 正在训练: HTAN (Full Model - 物理图网络 + 时序编码 + 帧级注意力)"
python demo_light.py --model HTAN \
    --train_batch_size $BATCH_SIZE --lr $LR \
    --use_graph 1 \
    --use_prior_mask 1 \
    --use_temporal_encoder 1 \
    --use_temporal_attention 1
echo "✅ HTAN Full 训练完成！"
echo "----------------------------------------------------------------------"

# 2. 跑核心对照：HTAN w/o Prior (无物理先验的纯数据驱动图)
echo "▶️ [2/4] 正在训练: HTAN w/o Prior (仅剥离物理掩码矩阵)"
python demo_light.py --model HTAN \
    --train_batch_size $BATCH_SIZE --lr $LR \
    --use_graph 1 \
    --use_prior_mask 0 \
    --use_temporal_encoder 1 \
    --use_temporal_attention 1
echo "✅ HTAN w/o Prior 训练完成！"
echo "----------------------------------------------------------------------"

# 3. 跑强基线：CRNN (经典水声基线)
# 关闭图网络和注意力，保留 1层 BiGRU
echo "▶️ [3/4] 正在训练: CRNN (经典时序基线 - 无图网络，无帧注意力)"
python demo_light.py --model HTAN \
    --train_batch_size $BATCH_SIZE --lr $LR \
    --use_graph 0 \
    --use_prior_mask 0 \
    --use_temporal_encoder 1 \
    --use_temporal_attention 0
echo "✅ CRNN 训练完成！"
echo "----------------------------------------------------------------------"

# 4. 跑最弱基线：Plain CNN (纯卷积模型)
# 关闭所有后端模块，仅保留前端多尺度卷积和全局池化
echo "▶️ [4/4] 正在训练: Plain CNN (最弱纯卷积基线 - 仅多尺度前端)"
python demo_light.py --model HTAN \
    --train_batch_size $BATCH_SIZE --lr $LR \
    --use_graph 0 \
    --use_prior_mask 0 \
    --use_temporal_encoder 0 \
    --use_temporal_attention 0
echo "✅ Plain CNN 训练完成！"
echo "----------------------------------------------------------------------"

echo "🎉 所有 4 组核心实验已全部完成！"
echo "👉 请查看项目根目录下的 htan_ablations_results.csv 获取最终对比表格。"
echo "👉 请前往 tb_logs/ 目录下查看每一组的 Confusion Matrix 截图。"