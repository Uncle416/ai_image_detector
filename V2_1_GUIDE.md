# V2.1 扩展数据训练指南

V2.1 不改变 DINOv2-B global classifier 的主体结构。它针对 V2.0 暴露出的跨数据集泛化问题，增加多来源数据、生成器平衡、基准集去重、generator-disjoint validation、保持长宽比的预处理和更可靠的模型选择指标。

## 1. 文件说明

- `prepare_v21_data.py`：数据抽取、manifest 转换与防泄漏合并工具；
- `config.v2_1.yaml`：A100 80GB 推荐训练配置；
- `V2_REPORT.md`：V2.0 技术与结果报告。

所有值得保留的数据建议放在 RunPod 的 `/workspace`：

```text
/workspace/
├── ai_image_detector/
├── v21_data/
│   ├── sid/
│   └── wildfake/
└── wildfake_eval/
```

## 2. Pod 分工与存储边界

当前低成本验证 Pod 和后续 A100 训练 Pod 不承担相同任务：

| 阶段 | 建议资源 | 可以执行 | 不建议执行 |
| --- | --- | --- | --- |
| 数据准备/验证 | 当前 CPU 或低价 Pod | 下载、解压、完整性检查、生成 manifest、去重、统计样本、小规模 smoke | 完整 DINOv2 训练、全量多条件评估 |
| 正式训练 | A100 80GB | CUDA 预检、V2.1 全量训练、保存 checkpoint | 在 container disk 保存唯一副本 |
| 正式评估 | A100，或训练结束后短时保留同一 GPU Pod | generator-disjoint val、WildFake clean 与全条件评估 | 在 CPU 上长时间跑完整 16 条件评估 |

现在不需要为了准备数据而重新启用 A100。完成第 3–7 节、确认最终 manifest 正常后，再创建或启动 A100 Pod。A100 Pod 必须：

1. 与现有 network volume 位于同一个数据中心；
2. 把该卷挂载到 `/workspace`；
3. 从 GitHub 拉取最新代码；
4. 确认 `/workspace/v21_data`、`/workspace/wildfake_eval` 和已有 checkpoint 仍可见；
5. 将所有训练输出写到 `/workspace/ai_image_detector/outputs/dinov2_v2_1`。

`/workspace` 是持久卷时，停止或删除 A100 Pod 后其中的数据仍保留；Pod 的 container disk 则不是可靠的唯一保存位置。切换 Pod 前先检查：

```bash
df -h /workspace
du -sh /workspace/v21_data /workspace/wildfake_eval /workspace/ai_image_detector/outputs 2>/dev/null
```

## 3. 在当前验证 Pod 安装数据工具

当前 Pod 只需要数据处理依赖，不要在这里安装或替换 A100 使用的 CUDA Torch。建议给数据准备单独建环境：

```bash
cd /workspace/ai_image_detector
python -m venv .venv-data
source .venv-data/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets Pillow ImageHash
```

## 4. 流式抽取 SID_Set

完整 SID_Set 约 140GB，不应完整保存到容量有限的 network volume。下面的命令从 train split 流式抽取 10,000 张 Real 与 10,000 张 full-synthetic，排除 tampered 类别 2，并尽可能保存数据集中原始图片字节而不是重新编码：

```bash
python prepare_v21_data.py sid \
  --output-root /workspace/v21_data/sid \
  --manifest /workspace/ai_image_detector/data/sid_v2_1.csv \
  --per-class 10000
```

输出 manifest 中 SID 样本均属于 `split=train`。

## 5. 从 CIFAKE 保留少量样本

避免 32×32 CIFAKE 主导训练分布，每类只保留 2,500 张：

```bash
python prepare_v21_data.py imagefolder \
  --root /workspace/ai_image_detector/data/cifake/train \
  --manifest /workspace/ai_image_detector/data/cifake_v2_1.csv \
  --source cifake \
  --per-class 2500
```

## 6. 转换已选择并解压的 WildFake 子集

不要下载完整 WildFake。选择若干非保留生成器和真实来源，下载相应 ZIP 与标签 CSV，解压到 `/workspace/v21_data/wildfake`。

先查看标签中可用的 `Weight` 名称：

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

for path in Path("/workspace/v21_data/wildfake/labels").glob("*.csv"):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(path.name, Counter(row.get("Weight", "") for row in rows).most_common())
PY
```

然后明确指定训练生成器和完全隔离的验证生成器：

```bash
python prepare_v21_data.py wildfake \
  --csv /workspace/v21_data/wildfake/labels/*.csv \
  --image-root /workspace/v21_data/wildfake/images \
  --manifest /workspace/ai_image_detector/data/wildfake_v2_1.csv \
  --train-generators TRAIN_GENERATOR_A TRAIN_GENERATOR_B TRAIN_REAL_SOURCE \
  --val-generators HELDOUT_GENERATOR HELDOUT_REAL_SOURCE \
  --per-generator 2500
```

必须把占位符替换成 CSV 中真实存在的 `Weight` 值。脚本无条件排除：

- DALL·E Advanced / DALL·E 3；
- 路径属于 COCO val2017 的图片。

同一个 generator 不能同时出现在 train 与 val。

## 7. 合并、平衡并与 benchmark 去重

已有外部 benchmark manifest：

```text
/workspace/ai_image_detector/data/wildfake_demo.csv
```

合并命令：

```bash
python prepare_v21_data.py merge \
  --inputs \
    /workspace/ai_image_detector/data/sid_v2_1.csv \
    /workspace/ai_image_detector/data/cifake_v2_1.csv \
    /workspace/ai_image_detector/data/wildfake_v2_1.csv \
  --benchmark-manifest /workspace/ai_image_detector/data/wildfake_demo.csv \
  --output /workspace/ai_image_detector/data/mixed_v2_1.csv \
  --max-per-generator 10000 \
  --phash-distance 4 \
  --balance-classes
```

该步骤会：

1. 删除与 benchmark SHA-256 完全相同的训练图片；
2. 删除与 benchmark 感知哈希距离不超过 4 的近重复图片；
3. 删除输入 manifest 内的重复图片；
4. 限制单个 `split + label + generator` 的最大样本数；
5. 分别平衡 train 和 val 的 Real/Fake 数量；
6. 输出绝对图片路径、generator、split 和哈希。

如果暂时只想进行精确哈希去重，可设置：

```bash
--phash-distance -1
```

## 8. 检查最终 manifest

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path("/workspace/ai_image_detector/data/mixed_v2_1.csv")
with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print("samples:", len(rows))
print("split/label:", Counter((r["split"], r["label"]) for r in rows))
print("generators:", Counter((r["split"], r["generator"]) for r in rows))
print("missing:", sum(not Path(r["path"]).is_file() for r in rows))
print("forbidden holdout:", sum(r["holdout"] == "1" for r in rows))
PY
```

要求：

- `missing: 0`；
- `forbidden holdout: 0`；
- train 和 val 都同时包含标签 0/1；
- val 中的生成器不出现在 train。

## 9. 在 A100 上进行 CUDA 预检与训练

以下步骤不要在当前 CPU 验证 Pod 上执行。先启动挂载同一 network volume 的 A100 80GB Pod，再运行：

```bash
cd /workspace/ai_image_detector
git pull --ff-only
python -m venv .venv-a100 --system-site-packages
source .venv-a100/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python cuda_preflight.py --config config.v2_1.yaml --run-forward
```

`.venv-data` 与 `.venv-a100` 分离是有意设计：前者属于当前数据验证 Pod，后者继承 RunPod PyTorch 模板自带的 CUDA Torch，避免把 CPU 版 Torch 或不匹配的 CUDA wheel 带入训练环境。

确认成功后，新建 tmux 会话：

```bash
tmux new -s v21
```

启动训练：

```bash
cd /workspace/ai_image_detector
source .venv-a100/bin/activate
mkdir -p outputs/dinov2_v2_1
python -u train.py --config config.v2_1.yaml 2>&1 | tee outputs/dinov2_v2_1/train.log
```

V2.1 使用新的数据和预处理，不应从 V2.0 的 optimizer/scheduler 状态续训，因此默认 `resume: null`。模型仍从公开 DINOv2 预训练权重开始。

`batch_size: 32` 时，每步会将 clean 和 augmented 拼接为 64 张送入 backbone。如果显存不足，将 batch size 改为 16，不需要修改其他代码。

训练结束后确认权重确实位于持久卷：

```bash
pgrep -af "python.*train.py" || true
ls -lh /workspace/ai_image_detector/outputs/dinov2_v2_1
sync
```

至少应看到 `best.pt`、`last.pt`、`history.json` 和 `train.log`。确认后即可停止 A100 Pod，避免继续产生 GPU 费用。若训练意外中断，先保留 `last.pt`，将 `config.v2_1.yaml` 中的 `resume` 指向它后再重新启动训练；不要从 V2.0 checkpoint 恢复 V2.1 的 optimizer 状态。

## 10. 评估顺序

先在 generator-disjoint val 上训练并按 AUROC 选择 `best.pt`。随后只跑外部 benchmark 的 clean 条件：

```bash
python evaluate.py \
  --config config.wildfake.yaml \
  --checkpoint /workspace/ai_image_detector/outputs/dinov2_v2_1/best.pt \
  --source wildfake_demo \
  --conditions clean
```

只有当外部 clean AUROC 相比 V2.0 的约 0.59 有明确提升时，才运行全部变换条件。完整评估去掉 `--conditions clean` 即可。
