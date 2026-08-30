# V2.0 技术方案与实验结果报告

## 1. 任务目标

本项目需要完成图像级二分类：判断输入图像是 **真实图像（Real）** 还是 **AI 生成图像（AIGC）**。除干净图像性能外，模型还需要在 JPEG 压缩、模糊、缩放、噪声、颜色调整和中心裁剪后保持稳定，并满足模型参数量严格小于 2B 的限制。

V2.0 的目标不是堆叠复杂分支，而是先验证一个清晰的假设：**通用视觉表征 + 与赛题一致的复合退化训练 + clean/degraded 特征一致性，能否形成一个可靠的鲁棒基线。**

## 2. 数据与划分

### 2.1 训练数据

V2.0 仅使用 CIFAKE：

- 训练集：100,000 张，Real/Fake 各 50,000 张；
- 内部验证集：20,000 张，Real/Fake 各 10,000 张；
- 标签约定：`0 = Real`，`1 = AI`。

CIFAKE 的训练集和测试集图像不同，因此不存在直接把测试图片用于梯度更新的问题。不过两者来自相同数据集、相同内容域和相同生成流程，所以内部测试衡量的是 **同分布泛化**，不能代表对新生成器和新真实图像来源的泛化能力。

### 2.2 外部演示验证集

按赛题要求，以下数据只用于外部演示验证，不参与训练：

- COCO val2017：4,998 张真实图片；
- DALL·E 3 Advanced：8,843 张 AI 图片。

我们通过原始 CSV 清单精确选择图片，而不是递归读取整个压缩包，从而排除 COCO 其他 split 和未标注文件。

## 3. 模型结构

V2.0 使用单一共享编码器：

```text
输入图像
   ↓
DINOv2-B/14
   ↓
[CLS token ; patch-token mean]
   ↓
LayerNorm → Linear → GELU → Dropout → Linear
   ↓
单个二分类 logit
```

具体设置：

- Backbone：`facebook/dinov2-base`；
- 输入分辨率：448×448；
- 池化：拼接 CLS token 与所有 patch token 的均值；
- 分类头：1536 → 512 → 1；
- 总参数量：87,371,009，远低于 2B；
- V2.0 只使用 global branch，尚未加入 local patches 或频谱分支。

选择 DINOv2 的原因是它提供较强的通用视觉表征，能够减少从零训练对数据规模的要求。选择 Base 而非 Large，是为了以较低成本快速验证训练策略；当前实验表明，瓶颈主要在训练数据域，而不在模型容量。

## 4. 鲁棒训练方法

### 4.1 复合退化

对每张训练图片 `x`，在线生成退化版本 `T(x)`。退化类型与赛题一致：

- JPEG：quality 90 / 70 / 50 / 30；
- Gaussian blur：sigma 0.5 / 1.0 / 2.0；
- Resize：0.5× / 0.25× 后恢复原尺寸；
- Gaussian noise：sigma 0.02 / 0.05 / 0.10；
- Color jitter：亮度、对比度、饱和度 ±20%；
- Center crop：保留中心 80% 后恢复原尺寸。

训练不是每次只应用一种变换，而是按照课程随机组合：

- 前期：0–1 个轻度变换；
- 中期：1–2 个轻/中度变换；
- 后期：1–3 个变换，并包含严重压缩、模糊和缩放。

Real 与 AI 图片使用完全相同的退化流水线，避免模型学习“压缩图就是 Real”等数据捷径。

### 4.2 配对分类与特征一致性

同一张图片的 clean 和 degraded 版本共享 DINOv2 编码器：

```text
x    → encoder → z_clean → logit_clean
T(x) → encoder → z_aug   → logit_aug
```

总损失：

```text
L = BCE_clean + BCE_aug + 0.2 × (1 - cosine(z_clean, z_aug))
```

两项 BCE 让 clean 和 degraded 图像都具有正确分类结果；一致性损失约束同一内容在退化前后的内部表征保持接近。

### 4.3 训练环境

- GPU：NVIDIA A100 80GB PCIe；
- PyTorch：2.8.0 + CUDA 12.8 runtime；
- 精度：BF16；
- Batch size：16；
- Epoch：10；
- Backbone 第 1 个 epoch 冻结，之后解冻；
- Backbone learning rate：1e-5；
- Head learning rate：1e-4；
- 优化器：AdamW；
- Scheduler：warmup + cosine decay。

Checkpoint 保存为：

```text
outputs/dinov2_v2/best.pt
outputs/dinov2_v2/last.pt
outputs/dinov2_v2/history.json
```

## 5. 实验结果

### 5.1 CIFAKE 内部验证

训练日志中观察到的 CIFAKE 内部验证 accuracy 均约为 95% 或更高；一次明确记录为：

- Accuracy：0.9547；
- F1：0.9558；
- AUROC：0.9934。

这说明模型能够很好地区分 CIFAKE 内部的 Real/Fake，并且优化过程已经正常收敛。

### 5.2 外部验证：100 张均衡 smoke test

外部 smoke test 使用 50 张 COCO val2017 和 50 张 DALL·E 3 Advanced：

| 条件 | Accuracy | AUROC | F1 |
|---|---:|---:|---:|
| Clean | 0.5800 | 0.5878 | 0.5435 |
| JPEG 30 | 0.5800 | 0.5860 | 0.5333 |
| Blur 2.0 | 0.6400 | 0.7628 | 0.5714 |
| Resize 0.25× | 0.6200 | 0.6810 | 0.5778 |
| Color jitter -20% | 0.5400 | 0.5634 | 0.5000 |
| Color jitter +20% | 0.5400 | 0.5406 | 0.4889 |

该次实验的 mean transformed accuracy 为 0.5820，最差条件为 color jitter -20%，accuracy 为 0.5400。

### 5.3 外部验证：1,000 张均衡 smoke test

扩大到 500 张 COCO val2017 与 500 张 DALL·E 3 Advanced 后：

- Clean accuracy：0.5800；
- Mean transformed accuracy：0.5875；
- Worst condition：center crop 80%；
- Worst-condition accuracy：0.5510。

样本扩大十倍后 clean accuracy 仍为 58%，说明外部性能问题不是 100 张小样本造成的偶然波动。

## 6. 结果解释

V2.0 得到了两个不同层面的结论。

第一，**复合退化与特征一致性确实让模型对后处理保持稳定**。外部数据上 mean transformed accuracy 没有明显低于 clean，说明 JPEG、blur、resize、noise 等变换没有造成额外的大幅崩溃。

第二，**稳定不等于正确**。外部 clean accuracy 只有 58%，clean AUROC 只有 0.5878。因为 AUROC 本身较低，所以单纯调整分类阈值无法解决问题。模型在 CIFAKE 上约 95%、在新数据域上约 58%，表明它主要学习了 CIFAKE 特有的分辨率、内容或生成器痕迹，而不是通用 AIGC 特征。

强模糊和强降采样反而提高外部 AUROC，也是重要证据。CIFAKE 图像分辨率很低，而 V2.0 将其放大到 448×448；当高分辨率外部图像被模糊或降采样后，它们的有效频率分布更接近 CIFAKE，模型反而更容易判断。这是典型的数据域捷径，而不是理想的取证能力。

## 7. V2.0 的优点与局限

### 优点

- 模型规模小于 100M，远低于 2B 限制；
- clean/degraded 配对训练实现正确；
- 退化参数与赛题一致；
- NVIDIA BF16 训练和 Mac/CPU checkpoint 加载接口兼容；
- 保存了 clean、mean transformed、worst condition 等鲁棒性结果；
- 外部验证严格与训练隔离。

### 局限

- 训练数据只有 CIFAKE，来源和生成器过于单一；
- CIFAKE 分辨率低，与实际社交媒体和 DALL·E 3 图像存在巨大域差异；
- 内部验证不是 generator-disjoint；
- 输入直接拉伸到正方形，会破坏原始长宽比；
- `best.pt` 按 accuracy 选择，尚未使用 AUROC 或 worst-group 指标；
- 尚未加入 local native-resolution patches。

## 8. V2.1 方向

V2.1 保持 DINOv2-B、global branch 和 paired consistency，不立即扩大模型。主要变化为：

1. 用 SID_Set、多个非保留 WildFake 生成器和少量 CIFAKE 构建平衡训练集；
2. 严格排除 COCO val2017 和 DALL·E Advanced，并进行精确哈希与感知哈希去重；
3. 按生成器划分 train/validation，而不是普通随机切分；
4. 按 `label + generator` 做训练采样平衡；
5. 使用保持长宽比的 resize + center crop；
6. 增加 balanced accuracy、Real recall、Fake recall，并按 AUROC 选择 best checkpoint；
7. V2.1 外部 AUROC 明显改善后，再进入 V3 global + local patch。

## 9. 当前结论

V2.0 是一个有效的训练与鲁棒性流水线基线，但不是一个已经具备强跨生成器泛化能力的最终检测器。实验支持以下判断：

> 鲁棒训练能够减少后处理导致的性能下降，但无法替代训练数据多样性。对于 AIGC 检测，跨来源、跨生成器的数据设计比单纯增大 backbone 更重要。

