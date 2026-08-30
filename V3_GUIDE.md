# V3：DINOv2 全局 + 纹理局部 Patch

## 1. V3 改了什么

V3 保留 V2.1 已验证有效的多来源数据、复合退化和 clean/degraded
一致性训练，只改变视觉输入结构：

```text
448×448 global image ───────────────┐
                                    ├─ concat ─ MLP ─ binary logit
2 high-texture + 2 low-texture      │
native 224×224 patches ─ shared DINO┘
```

全局图与四个局部 patch 使用同一个 `facebook/dinov2-base`，不是五套
backbone。每个 patch 的 DINO 特征先做均值池化，再与 global 特征拼接。
局部位置由规则网格候选的 Laplacian variance 排序得到；训练时 clean 与
degraded 图复用相同位置，避免一致性损失比较不相干区域。

当前 V3 不加入 FFT、频谱支路、gating 或 worst-group loss。这样外部结果发生
变化时，能把差异归因于 local patch 分支。

## 2. 为什么先使用与 V2.1 相同的数据

组员当前实际训练方案是 SID、CIFAKE、WildFake 各约 30,000 张，而不是早期
指南中的抽样规模。第一次 V3 实验必须继续使用产生 V2.1 checkpoint 的同一份
manifest，才能回答“local patch 是否有效”。

建议保留两次独立实验：

1. `dinov2_v3`：与 V2.1 完全相同数据，只改变网络结构；
2. `dinov2_v3_wildfake_plus`：扩大 WildFake 后再训练，衡量数据扩展的额外收益。

不要覆盖旧 manifest、checkpoint 或输出目录，否则结构收益和数据收益会混在一起。

## 3. A100 80GB 首次运行

拉取代码并检查 V3 前向：

```bash
cd /workspace/ai_image_detector
git pull
source .venv/bin/activate
python cuda_preflight.py --config config.v3.yaml --run-forward
```

`config.v3.yaml` 默认 physical batch 为 8，梯度累计 4 次，有效 batch 为 32。
V3 每个样本需要处理 clean/degraded 的 global 与四个 local patches，计算量约为
V2.1 的两倍。先完成 smoke 或至少观察前 100 个 step 的显存和速度，再考虑把
physical batch 提高到 12 或 16；OOM 时先降到 4，不要改变有效 batch，改为相应
增加梯度累计。

正式训练：

```bash
mkdir -p outputs/dinov2_v3
python -u train.py --config config.v3.yaml 2>&1 | tee outputs/dinov2_v3/train.log
```

输出目录除 `best.pt`、`last.pt`、`history.json` 外，还会生成
`run_metadata.json`，记录配置哈希、Git commit、训练/验证数量、标签计数、
generator 计数和 manifest 哈希。

## 4. 扩大 WildFake 的实验

新数据先写成新的 manifest，例如：

```text
/workspace/ai_image_detector/data/mixed_v3_wildfake_plus.csv
```

复制 `config.v3.yaml` 为 `config.v3_wildfake_plus.yaml`，只修改：

```yaml
output_dir: /workspace/ai_image_detector/outputs/dinov2_v3_wildfake_plus
data:
  train:
    path: /workspace/ai_image_detector/data/mixed_v3_wildfake_plus.csv
  val:
    path: /workspace/ai_image_detector/data/mixed_v3_wildfake_plus.csv
```

必须继续排除演示验证集的 COCO val2017 与 DALL·E Advanced，并保持
generator-disjoint validation。增加 WildFake 时关注 generator 平衡，而不是只把
某一个容易识别的生成器扩到很大。

## 5. 验证

先 clean：

```bash
python evaluate.py \
  --config config.wildfake.yaml \
  --checkpoint outputs/dinov2_v3/best.pt \
  --source wildfake_demo \
  --conditions clean
```

确认 clean AUROC、balanced accuracy、Real/Fake recall 后，再去掉
`--conditions clean` 跑完整鲁棒性表。V3 是否成功必须同时比较：

- clean AUROC；
- balanced accuracy；
- Real recall / Fake recall；
- mean transformed accuracy；
- worst-condition accuracy；
- 同样本数、同 manifest 下相对 V2.1 的变化。

## 6. best.pt 能告诉我们什么

本仓库的 `train.py` 会把完整 YAML `config` 保存进 checkpoint，所以可信来源的
checkpoint 可以显示 backbone、结构、batch size、学习率、epoch 数、数据 manifest
路径等：

```bash
python inspect_checkpoint.py outputs/dinov2_v3/best.pt
```

但旧 checkpoint 不包含 manifest 内容统计，也不能证明实际磁盘上的 manifest 在
训练后没有被替换。因此，不能只凭旧 `best.pt` 推断“三个数据集各 30,000 张”。
V3 新增的 metadata 和 manifest SHA256 用于补上这一缺口。

只对自己或可信组员提供的 `.pt` 使用 `inspect_checkpoint.py`；PyTorch checkpoint
不应被当作安全的任意来源文件打开。
