# V3 扩展训练集：SID_Set 40k + CIFAKE 40k + WildFake 60k

## 1. 默认实验设计

`prepare_v21_data.py compose-v3` 默认按以下目标抽样：

- SID_Set：40,000 张训练图，Real/Fake 各 20,000；
- CIFAKE：40,000 张训练图，Real/Fake 各 20,000；
- WildFake：60,000 张训练图，Real/Fake 各 30,000；
- WildFake generator-disjoint 内部验证：10,000 张，Real/Fake 各 5,000。

候选池充足时，最终 manifest 约 150,000 行，其中训练约 140,000、内部验证约
10,000。默认允许去重后每个目标最多短缺 10%，不会因为少量图片缺失而中断。
WildFake
在每个标签内部采用生成器轮转抽样，避免一个大生成器占满 30,000 张。所有抽样
使用固定随机种子，可重复生成相同结果。

`config.v3_expanded.yaml` 使用普通 `shuffle`，因此每个 epoch 会各看到 SID_Set
约 40,000、CIFAKE 约 40,000、WildFake 约 60,000。不要改回 `group_balanced`；后者会让
每个生成器等概率，WildFake 的生成器较多时会改变三来源的实际训练比例。

若希望三个来源更接近 1:1:1，把 `--wildfake-train-total` 改为 `40000`。
建议先使用默认 60,000，因为 V2.1 的主要问题是新生成器泛化，而不是 CIFAKE
内部准确率。

## 2. 数据隔离规则

以下图片只能用于最终演示验证，不能进入训练或内部验证：

- COCO val2017；
- DALL·E Advanced / DALL·E 3 Advanced。

WildFake 的 train generator 与 val generator 必须完全不同。工具会再次进行：

- holdout 行排除；
- benchmark SHA-256 精确去重；
- 可选 benchmark 感知哈希近重复排除；
- 三个来源之间的 SHA-256 去重；
- WildFake train/val generator 交集检查；
- 每个来源内部 Real/Fake 数量相等。

## 3. 准备环境

数据准备不需要 A100，可以在挂载同一 network volume 的 CPU Pod 完成：

```bash
cd /workspace/ai_image_detector
git pull --ff-only

python -m venv .venv-data --system-site-packages
source .venv-data/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets Pillow ImageHash modelscope

mkdir -p /workspace/v3_data/sid
mkdir -p /workspace/v3_data/wildfake
mkdir -p /workspace/ai_image_detector/data
```

## 4. SID_Set：先取 44,000 候选，最终保留 40,000

SID_Set 使用流式读取，不下载完整约 140GB 数据集。先保存 22,000 Real 和
22,000 full-synthetic 候选，忽略 tampered 类别。多出的 4,000 张是去重余量；
最终 `compose-v3` 仍只保留 40,000 张：

```bash
python prepare_v21_data.py sid \
  --output-root /workspace/v3_data/sid/images \
  --manifest /workspace/ai_image_detector/data/sid_v3_pool.csv \
  --per-class 22000 \
  --shuffle-buffer 20000 \
  --max-scanned 500000 \
  --seed 42
```

如果之前已经用相同输出目录保存了一部分 SID_Set，内容哈希命名会避免重复写入；
新候选 manifest 会包含 44,000 行。

## 5. CIFAKE：先取 50,000 候选，最终保留 40,000

```bash
python prepare_v21_data.py imagefolder \
  --root /workspace/ai_image_detector/data/cifake/train \
  --manifest /workspace/ai_image_detector/data/cifake_v3_pool.csv \
  --source cifake \
  --per-class 25000 \
  --seed 42
```

该命令只生成候选 manifest，不复制 CIFAKE 图片，因此不会额外占用图片空间。
最终 `compose-v3` 会在去重后以 20,000 Real 和 20,000 Fake 为目标抽样。

## 6. WildFake：选择训练与内部验证生成器

不要下载或解压完整 1TB WildFake。只准备计划使用的若干生成器 ZIP 和 CSV。
先从 WildFake 文件页面记下需要的 ZIP 和对应标签 CSV 的完整仓库路径，然后只下载
这些文件。例如：

```bash
python download_modelscope_subset.py \
  --dataset hy2628982280/WildFake \
  --local-dir /workspace/v3_data/wildfake/raw \
  --files \
    "Images/Diffusion_based/YOUR_GENERATOR_A.zip" \
    "Images/Diffusion_based/YOUR_GENERATOR_B.zip" \
    "Images/Real/YOUR_REAL_SOURCE.zip" \
    "label_csv_files/YOUR_GENERATOR_A.csv" \
    "label_csv_files/YOUR_GENERATOR_B.csv" \
    "label_csv_files/YOUR_REAL_SOURCE.csv" \
  --max-workers 4 \
  --international
```

RunPod 等中国大陆以外的主机使用 `--international`；位于中国大陆的电脑或云主机
通常应去掉这个参数。脚本使用 ModelScope 官方的按文件 pattern 下载接口，并在
结束时检查每个 pattern 是否真的匹配到本地文件。

下载后把所选 ZIP 解压到统一图片根目录。例如：

```bash
mkdir -p /workspace/v3_data/wildfake/images
unzip -q /workspace/v3_data/wildfake/raw/Images/Diffusion_based/YOUR_GENERATOR_A.zip \
  -d /workspace/v3_data/wildfake/images
```

只在 ZIP 完整性检查和解压成功后，才考虑移走原始 ZIP 释放空间。

先检查已经拥有的数据量：

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

root = Path("/workspace/v3_data/wildfake/raw/label_csv_files")
counts = Counter()
for path in root.glob("*.csv"):
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            generator = row.get("Weight") or row.get("Architecture") or row.get("Category")
            label = "fake" if row.get("IsFake") == "1" else "real"
            advanced = row.get("IsAdvanced") == "1"
            counts[(generator, label, advanced)] += 1
for key, count in counts.most_common():
    print(key, count)
PY
```

选择至少若干个 fake generator 和多个 real source。保留完全不同的 real/fake
generator 给内部验证，然后运行（把大写占位符替换成 CSV 中真实 `Weight`）：

```bash
python prepare_v21_data.py wildfake \
  --csv /workspace/v3_data/wildfake/raw/label_csv_files/*.csv \
  --image-root /workspace/v3_data/wildfake/images \
  --manifest /workspace/ai_image_detector/data/wildfake_v3_pool.csv \
  --train-generators \
    TRAIN_REAL_A TRAIN_REAL_B TRAIN_FAKE_A TRAIN_FAKE_B TRAIN_FAKE_C TRAIN_FAKE_D \
  --val-generators \
    HELDOUT_REAL HELDOUT_FAKE \
  --per-generator 10000 \
  --seed 42
```

`--per-generator` 是候选池上限，不是最终每个生成器一定取 10,000 张。下一步会从
候选池中按生成器轮转，以 60,000 train 和 10,000 val 为目标抽样。默认只要
去重后的可用数量达到目标的 90%，程序就继续并打印实际数量；低于 90% 才报错，
避免某个来源明显偏少。

## 7. 合成最终 manifest

如果训练与外部验证位于同一个 volume，可以传入外部 benchmark manifest 做额外的
图片哈希防泄漏检查：

```text
/workspace/ai_image_detector/data/wildfake_demo.csv
```

严格版本使用感知哈希距离 4：

```bash
python prepare_v21_data.py compose-v3 \
  --sid-manifest /workspace/ai_image_detector/data/sid_v3_pool.csv \
  --cifake-manifest /workspace/ai_image_detector/data/cifake_v3_pool.csv \
  --wildfake-manifest /workspace/ai_image_detector/data/wildfake_v3_pool.csv \
  --benchmark-manifest /workspace/ai_image_detector/data/wildfake_demo.csv \
  --benchmark-root /workspace/wildfake_eval \
  --output /workspace/ai_image_detector/data/mixed_v3_expanded.csv \
  --sid-train-total 40000 \
  --cifake-train-total 40000 \
  --wildfake-train-total 60000 \
  --wildfake-val-total 10000 \
  --minimum-quota-fraction 0.90 \
  --phash-distance 4 \
  --seed 42
```

感知哈希会解码候选图片，CPU 上可能较慢。仅用于先检查数量和路径时，可以使用
`--phash-distance -1` 只做 SHA-256 精确去重；最终正式 manifest 建议仍使用 4。

如果 A100 只负责训练，外部演示图片保存在另一个 A40 Pod，可以省略
`--benchmark-manifest` 和 `--benchmark-root`。`wildfake` 导出阶段仍会按元数据排除
COCO val2017 与 DALL·E Advanced；GitHub 中的
`data/reference_wildfake_demo.csv` 用于在验证 Pod 上复现固定的 13,841 张样本，不含
图片本身。

除 CSV 外还会生成：

```text
data/mixed_v3_expanded.csv.stats.json
```

其中记录来源、split、标签、生成器数量、去重原因以及最终引用图片的实际字节数。
`compose-v3` 只引用已有图片，不复制图片。

## 8. 容量检查

数据量不会增加 GPU 显存需求，但会线性增加每个 epoch 的训练时间。A100 80GB
仍使用 physical batch 8、梯度累计 4。140,000 张训练图相对 90,000 张约增加 56%
的每 epoch step 数。

75GB network volume 是否够不能只看图片数量，要看原始分辨率和是否同时保留 ZIP。
RunPod 共享挂载的 `df -h /workspace` 有时显示后端总容量，不等于账号卷配额。应结合
RunPod 控制台中的 volume size 与以下实际占用：

```bash
du -sh \
  /workspace/v3_data \
  /workspace/ai_image_detector/data/cifake \
  /workspace/wildfake_eval \
  /workspace/ai_image_detector/outputs \
  /workspace/hf-cache 2>/dev/null
```

训练前还要给 `best.pt`、`last.pt`、临时 checkpoint、日志和模型缓存至少预留
5GB，建议预留 10GB。若仍保留 25GB 的 DALL·E ZIP、解压副本和其他原始 ZIP，
75GB 很可能偏紧；100–150GB 更稳妥。

## 9. A100 训练

挂载同一 network volume 后：

```bash
cd /workspace/ai_image_detector
git pull --ff-only
source .venv-v3/bin/activate
export HF_HOME=/workspace/hf-cache

python cuda_preflight.py --config config.v3_expanded.yaml --run-forward
tmux new -s v3-expanded
```

在 tmux 内启动：

```bash
cd /workspace/ai_image_detector
source .venv-v3/bin/activate
export HF_HOME=/workspace/hf-cache
mkdir -p outputs/dinov2_v3_expanded

python -u train.py \
  --config config.v3_expanded.yaml \
  2>&1 | tee outputs/dinov2_v3_expanded/train.log
```

输出全部位于 `/workspace/ai_image_detector/outputs/dinov2_v3_expanded`，不会覆盖
现有 V2.1 或第一版 V3。
