# SensSmartTech Baseline 与扩展接口

## 当前 baseline

```text
PPG [B, 4, 2000]
  -> BaselinePPGEncoder
       latent [B, 128, 125]
       skips  [(B, 32, 1000), (B, 64, 500), (B, 128, 250)]
  -> BaselineECGDecoder
ECG [B, 4, 2000]
```

当前 baseline 的假设是：输入和目标已经由 SensSmartTech 预处理；A/B
运动状态只保留在 metadata 中，不作为模型条件；训练目标只有 ECG 的逐点
MSE。Encoder 中的 Dropout1d 为轻量正则，Decoder 输出附近不使用 Dropout。

运行：

```powershell
python -m src.train --config configs/exp_baseline.yaml
python scripts/evaluate.py --run runs/senssmarttech_baseline --baselines
```

评估结果会同时包含总体结果和 `model_by_activity.A/B` 分层结果。

## 模块接口约定

注册表位于 `src/models/registry.py`。Encoder 可以返回两种形式：

1. 旧模块直接返回 `latent` tensor；
2. 新模块返回字典，至少包含 `latent`，也可以附带 `skips`、`mu`、`logvar`、
   `z_content`、`z_style` 等字段。

Decoder 接收上述 tensor 或字典。因此后续增加 VAE 或因子化表示时，不必
   破坏现有训练入口；只需要在训练器中为新增字段和损失注册对应模块。

## 后续想法的推荐插入位置

| 想法 | 第一插入位置 | 需要新增的证据 |
|---|---|---|
| 1D-CNN | 替换 `BaselinePPGEncoder` 的残差块 | 参数量、收敛速度、总体/A/B 结果 |
| TCN | Encoder 的下采样块或 latent refinement | 感受野、长时依赖、跨被试性能 |
| Dropout | Encoder 中间层，先试 0/0.1/0.2 | 过拟合曲线和波形形态指标 |
| Mask | 训练数据增强或 Encoder 输入 mask | mask 比例、缺失通道/片段鲁棒性 |
| VAE | Encoder 返回 `mu/logvar`，训练器增加 KL | 重建质量、KL 曲线、采样稳定性 |
| 通道博弈/通道重要性 | PPG 输入门控或通道遮挡实验 | 每个通道的边际贡献和跨被试一致性 |
| 特征博弈论 | latent/中间层做遮挡或 Shapley 近似 | 解释结果是否和性能变化一致 |
| 因果/IRM | 先把 subject 或 activity 当 environment | 环境划分、OOD gap、风险方差 |
| GRL | latent 后接 subject classifier | 被试分类准确率下降且 ECG 性能不降 |
| InfoNCE | PPG/ECG 双编码器的 projection head | 正负样本定义和跨模态检索指标 |
| Flow/Diffusion | latent 之后、ECG decoder 之前 | 多样性、分布质量和生理一致性 |

## 推荐消融顺序

```text
E0  当前四路 PPG + Res-Encoder/Decoder + MSE
E1  E0 + Dropout
E2  E0 + 频域或多尺度波形损失（一次只加一种）
E3  E0 + TCN 或更深 1D-CNN
E4  E0 + mask/通道遮挡增强
E5  E0 + PPG/ECG latent alignment
E6  E5 + z_content/z_style 因子化
E7  E6 + subject-GRL
E8  E7 + VAE 或 latent Flow（二选一先做）
E9  E8 + IRM/V-REx/因果分析
```

每个实验只改变一个主要因素，固定 `subject-wise 22/5/5`、随机种子、
预处理和评估协议。A/B 结果用于回答“运动状态下是否退化”，而不是作为
模型已经学到因果关系的证据。

## 需要避免的混淆

- VAE 的随机采样不是 Dropout；VAE 必须有 `mu`、`logvar`、重参数化和 KL。
- GRL 只能抑制指定标签信息，不能自动得到因果表示。
- 全局特征去相关不等于因果识别，也可能破坏真实的生理相关性。
- 峰值检测、PTT 和 HR 指标当前用于评估；若要作为损失，需要先改成可微近似。
- SensSmartTech 的 ECG 是 4 导联、窗口长度 2000；不要把旧的 12 导联/1250
  样本注释直接当作当前数据事实。
