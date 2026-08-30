# 从零复现 WildFake 演示验证集

挑战演示集由 4,998 张 COCO val2017 真实图像和 8,843 张 DALL-E Advanced
图像组成。两类数据只用于验证，禁止加入训练集。

## 1. 获取代码并创建环境

建议在 Linux、RunPod 或 Windows WSL2 Ubuntu 中运行：

```bash
git clone https://github.com/Uncle416/ai_image_detector.git
cd ai_image_detector
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install modelscope
```

在 NVIDIA 电脑上先确认 PyTorch 能识别显卡：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出 `False`，需要根据 PyTorch 官网为本机驱动重新安装 CUDA 版 PyTorch，
不要继续跑完整验证。

## 2. 下载四个 WildFake 源文件

```bash
mkdir -p data/wildfake_raw
modelscope download --dataset hy2628982280/WildFake \
  Images/Diffusion_based/DALLE.zip \
  Images/Real/coco.zip \
  label_csv_files/dalle3.csv \
  label_csv_files/real_coco.csv \
  --local_dir data/wildfake_raw
```

预计下载约 28 GB。构建过程中还要保存解压后的图像，建议开始前至少准备
65–70 GB 可用空间。

## 3. 构建验证目录与清单

如果有当前项目已经使用过的 `wildfake_demo.csv`，先把它保存为
`data/reference_wildfake_demo.csv`，然后执行：

```bash
python prepare_wildfake_validation.py \
  --raw-root data/wildfake_raw \
  --output-root data/wildfake_eval \
  --manifest data/wildfake_demo.csv \
  --selection-manifest data/reference_wildfake_demo.csv
```

这样会按完整相对路径复现相同样本，避免 DALL-E 重名文件造成误配。

如果拿不到旧清单，也可以执行：

```bash
python prepare_wildfake_validation.py \
  --raw-root data/wildfake_raw \
  --output-root data/wildfake_eval \
  --manifest data/wildfake_demo.csv
```

此模式会使用全部 8,843 张 DALL-E 图片，并按稳定排序选 4,998 张 COCO
val2017 图片。COCO 官方 val2017 通常有 5,000 张，而题目只写 4,998 张；
没有主办方样本清单时，无法证明本地排除的两张与主办方完全一致。

## 4. 放置 checkpoint 并验证

例如把权重放到：

```text
outputs/dinov2_v2_1/best.pt
```

先只验证 clean，确认模型、路径和显存均正常：

```bash
python evaluate.py \
  --config config.wildfake.yaml \
  --checkpoint outputs/dinov2_v2_1/best.pt \
  --source wildfake_demo \
  --conditions clean
```

再跑完整鲁棒性表：

```bash
python evaluate.py \
  --config config.wildfake.yaml \
  --checkpoint outputs/dinov2_v2_1/best.pt \
  --source wildfake_demo
```

结果写入 checkpoint 同目录：

```text
outputs/dinov2_v2_1/wildfake_robustness.csv
outputs/dinov2_v2_1/wildfake_robustness.json
```

## 5. 完整性检查与清理

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

manifest = Path("data/wildfake_demo.csv")
with manifest.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
print("rows:", len(rows))
print("labels:", Counter(row["label"] for row in rows))
print("missing:", sum(not (Path("data/wildfake_eval") / row["path"]).is_file() for row in rows))
PY
```

期望输出总数 13,841、标签计数 `0: 4998` 与 `1: 8843`、缺失数 0。
确认无误后可以删除 `data/wildfake_raw` 中的两个 ZIP，回收约 28 GB；
CSV 和 `data/wildfake_eval` 必须保留。
