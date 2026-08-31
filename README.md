# 鲁棒 AI 图像检测器（V0–V3）

这是一个面向 hackathon 的 DINOv2 二分类原型，当前刻意只实现：

- V0：DINOv2 全局图像编码器 + 二分类 MLP；
- V1：与题目参数一致的 0–3 项复合退化和课程式增强；
- V2：clean/degraded 配对分类损失与特征一致性损失；
- V3：同一 DINOv2 共享编码 global image 与 2 个高纹理、2 个低纹理 local patches。

暂未加入频谱分支、worst-group loss 和自动 degradation gating。V2.1/V3 的 generator-disjoint 划分由数据 manifest 明确控制。

V2.0 的完整技术说明与外部实验结论见 [`V2_REPORT.md`](V2_REPORT.md)。针对外部泛化问题的数据扩展版见 [`V2_1_GUIDE.md`](V2_1_GUIDE.md)、`prepare_v21_data.py` 和 `config.v2_1.yaml`。共享编码器的 global + 4 texture patch 版本见 [`V3_GUIDE.md`](V3_GUIDE.md) 与 `config.v3.yaml`。SID_Set 约 40k + CIFAKE 约 40k + WildFake 约 60k 的来源平衡方案见 [`V3_EXPANDED_DATA_GUIDE.md`](V3_EXPANDED_DATA_GUIDE.md) 与 `config.v3_expanded.yaml`。

新电脑获取代码、WildFake 演示验证集并运行 checkpoint 的完整流程见
[`VALIDATION_SETUP.md`](VALIDATION_SETUP.md)。

## 关键约定

- 标签固定为 `0 = Real`、`1 = AI`。
- 默认 backbone 是 `facebook/dinov2-base`，方便先跑通；把 `config.yaml` 中的 `backbone` 改为 `facebook/dinov2-large` 即可切换 DINOv2-L/14。
- 默认输入为 448×448，DINOv2 的位置编码在前向时插值。
- 模型构建时检查总参数量，必须严格小于 2B。
- `runtime.device` 支持 `auto/cuda/cuda:0/mps/cpu`；checkpoint 先映射到 CPU，再迁移到目标设备，因此 Mac 训练结果可直接在 NVIDIA CUDA 上加载。
- CUDA 精度支持 `auto/fp32/fp16/bf16`；`auto` 优先使用受支持的 BF16，否则使用带 GradScaler 的 FP16。TF32、cuDNN benchmark 和确定性模式均可配置。
- Real 和 AI 样本使用完全相同的退化抽样流程，避免“压缩图 = Real”之类的捷径。
- SID_Set 的标签 2 表示 tampered，不完全等同于全图 AI 生成；默认 `label_map` 不包含 2，因此会排除。只有在实验定义明确时再将 2 映射为 1。

## 安装

建议使用带 CUDA 的 Linux 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会从 Hugging Face 下载 DINOv2 权重。448 分辨率下，DINOv2-L 显存压力明显高于 B；若显存不足，先减小 `batch_size`，增加梯度累积，或暂时把 `image_size` 改为 224。

迁移到 NVIDIA 机器后，将配置改为：

```yaml
runtime:
  device: cuda:0
  precision: auto
  allow_tf32: true
  cudnn_benchmark: true
  deterministic: false
```

先按目标机器的 CUDA/驱动组合安装相匹配的 PyTorch CUDA 构建，再运行预检：

```bash
python cuda_preflight.py --config config.yaml --run-forward
```

预检会报告 PyTorch/CUDA 版本、GPU 型号、GPU 数量、BF16 支持情况，并执行一次真实的 GPU 前向。当前代码明确支持单卡 CUDA；checkpoint 与设备解耦。多卡 DDP/Accelerate 尚未实现，后续可以在不改动模型接口和数据格式的前提下接入。

## 数据配置

### 1. CIFAKE / ImageFolder

默认配置适配如下结构，目录名大小写不敏感：

```text
data/cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

### 2. CSV manifest

推荐多数据集实验使用 manifest：

```csv
path,label,split,generator,source,id
images/real_001.jpg,0,train,camera,coco_train2017,r1
images/ai_001.png,1,train,stable-diffusion,custom,a1
images/ai_002.png,1,val,midjourney,custom,a2
```

配置示例：

```yaml
data:
  train:
    kind: manifest
    path: data/manifest.csv
    root: data
    split: train
    forbid:
      source: [coco_val2017]
      generator: [dalle_advanced]
  val:
    kind: manifest
    path: data/manifest.csv
    root: data
    split: val
```

`include` / `exclude` 可对任意 CSV 字段做精确过滤；`forbid` 一旦匹配就直接终止训练，适合做防泄漏保护。**题目指定的 WildFake 演示子集（COCO val2017 4,998 张与 DALL·E Advanced 8,843 张）不得出现在训练清单中。** 最稳妥的做法是把它单独保存为 `data.robustness_test`，只传给 `evaluate.py --source robustness_test`。

### 3. Hugging Face SID_Set

```yaml
data:
  train:
    kind: huggingface
    name: saberzl/SID_Set
    split: train
  val:
    kind: huggingface
    name: saberzl/SID_Set
    split: val
  label_map: {0: 0, 1: 1}  # 排除 tampered 类别 2
```

SID_Set 数据量较大；首次验证流水线时建议先使用 CIFAKE 或创建小型 manifest。

## 训练与评估

```bash
python train.py --config config.yaml
python evaluate.py \
  --config config.yaml \
  --checkpoint outputs/dinov2_v2/best.pt \
  --source val
```

首次端到端冒烟测试使用均衡小子集：

```bash
python train.py --config config.smoke.yaml
```

`config.smoke.yaml` 从 CIFAKE 的每个类别选取 500 张训练图和 100 张测试图，使用 224×224、1 epoch，并保存到 `outputs/smoke/`。限样本由数据源的 `max_samples_per_class` 控制，抽样顺序固定，便于复现。

训练保存 `last.pt`、按 clean validation accuracy 选出的 `best.pt` 和 `history.json`。评估输出：

- 每个 JPEG、blur、resize、noise 严重度的 Accuracy / AUROC / F1；
- color jitter 的 ±20% 两个端点；
- center crop 80%；
- clean accuracy、mean transformed accuracy、worst-condition accuracy；
- `robustness.csv` 与 `robustness.json`。

## 复合退化课程

- 前 25% epoch：0–1 个轻度退化；
- 25%–60% epoch：1–2 个轻/中度退化；
- 后 40% epoch：1–3 个退化，覆盖 JPEG30、blur2.0、resize0.25 等重度条件。

退化类型从 JPEG、Gaussian blur、resize roundtrip、Gaussian noise、brightness/contrast/saturation jitter、center crop 中无放回采样。退化在原始分辨率上完成，之后 clean 与 degraded 分别做相同的 448×448 模型预处理；这与真实的“先在平台传播/重编码，再进入检测器”顺序一致。

## 下一阶段的实验纪律

先确认 V0、V1、V2 的消融结果，再进入 V3：

1. V0：`augmentation.enabled: false`，`augmented_bce_weight: 0`，`lambda_consistency: 0`；
2. V1：`augmentation.enabled: true`，`augmented_bce_weight: 1`，`lambda_consistency: 0`；
3. V2：`augmentation.enabled: true`，`augmented_bce_weight: 1`，`lambda_consistency: 0.2`。

三组实验必须复用相同数据划分和随机种子，并同时比较 clean、mean transformed 与 worst-condition accuracy。之后才适合加入 global + local patch 分支。
