# 施工区域 MaskDINO 训练流程

本文说明如何将现有 YOLO segmentation 多边形标签转换为 MaskDINO 可读取的
COCO instance segmentation 标注，并使用单类别 `construction_area` 配置训练模型。

## 数据约定

- 原始 YOLO 类别 `0` 表示 `construction_area`（施工区域）。
- COCO JSON 中该类别写为 `category_id: 1`。
- Detectron2 注册时将 COCO 类别 `1` 映射到模型内部连续类别 `0`。
- 空 `.txt` 标签文件保留为没有施工区域的负样本，并在自定义配置中启用其训练加载。
- 当前划分方式为固定随机种子 `42` 的随机划分，训练集占 `80%`，验证集占 `20%`。

期望目录结构如下：

```text
datasets/construction_area/
  image/                       # 原始图片
  label/                       # 原始 YOLO polygon 标签
  images/
    train/
    val/
  labels/
    train/
    val/
  annotations/                 # 转换脚本生成
    instances_train.json
    instances_val.json
  visualizations/              # 可视化脚本生成
```

## 1. 划分数据集

在项目根目录运行：

```bash
cd ~/autodl-tmp/disaster_maskdino
conda activate maskdino
python datasets/split_dataset.py
```

该步骤会将图片与 YOLO 标签复制到 `images/{train,val}` 与
`labels/{train,val}`。你已运行过一次；当原始图片或标注发生变化时再重新运行。

## 2. 转换为 COCO JSON

```bash
cd ~/autodl-tmp/disaster_maskdino
python datasets/yolo_to_coco.py
```

默认输出：

```text
datasets/construction_area/annotations/instances_train.json
datasets/construction_area/annotations/instances_val.json
```

转换脚本会执行以下检查：

- 只接受 YOLO 类别 id `0`。
- 多边形必须至少有 3 个点。
- 归一化坐标必须位于 `[0, 1]`。
- 多边形面积必须大于 `0`。
- 每张已划分图片必须存在对应 `.txt` 文件，空文件是合法负样本。

## 3. 抽样可视化检查

先查看验证集的随机样本：

```bash
python datasets/visualize_coco_annotations.py --split val --num-samples 12
```

也可检查训练集：

```bash
python datasets/visualize_coco_annotations.py --split train --num-samples 12
```

输出目录分别为：

```text
datasets/construction_area/visualizations/val/
datasets/construction_area/visualizations/train/
```

红色半透明多边形应覆盖施工区域。标注为 `instances: 0` 的图应确实不含施工区域；
若发现漏标，应先修正原始 `label/` 文件，再重新执行划分和转换。

## 4. 训练配置和数据集注册

本项目新增的数据集名称为：

```text
construction_area_train
construction_area_val
```

注册代码位于：

```text
MaskDINO/maskdino/data/datasets/register_construction_area.py
```

训练配置位于：

```text
MaskDINO/configs/construction_area/maskdino_R50_1class.yaml
```

该配置基于 COCO instance segmentation 的 ResNet-50 配置，关键参数为：

```yaml
MODEL:
  SEM_SEG_HEAD:
    NUM_CLASSES: 1
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 0.0000125
  MAX_ITER: 27000
TEST:
  EVAL_PERIOD: 1000
DATALOADER:
  FILTER_EMPTY_ANNOTATIONS: False
```

`27000` iterations 对当前 `1086` 张训练图、batch size `2` 约等于 50 epochs。
请在训练前确认空标签图片确实不含施工区域；否则这些漏标会被模型学习为背景。

## 5. 下载预训练权重

从项目根目录执行：

```bash
cd ~/autodl-tmp/disaster_maskdino/MaskDINO
mkdir -p pretrained
wget -O pretrained/maskdino_r50_coco_instance.pth \
  https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
```

该权重来自 MaskDINO 官方 README 中与基础 R50 配置匹配的 COCO 实例分割模型。
由于预训练模型有 80 个 COCO 类别而本任务只有 1 个类别，加载分类层时出现形状
不匹配提示是预期现象；骨干网络与分割特征仍可用于微调。

## 6. 先进行短跑训练测试

设置 Detectron2 数据目录并只训练 100 iterations：

```bash
PROJECT_ROOT=~/autodl-tmp/disaster_maskdino
cd "$PROJECT_ROOT/MaskDINO"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_coco_instance.pth \
  SOLVER.MAX_ITER 100 \
  TEST.EVAL_PERIOD 100 \
  OUTPUT_DIR output/construction_area_debug
```

短跑阶段应检查：

- 日志能识别 `construction_area_train` 与 `construction_area_val`。
- 数据加载和 COCO evaluation 没有报错。
- loss 数值正常输出。
- GPU 显存足够。

若发生 CUDA out of memory，先用较小 batch size 重跑：

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_coco_instance.pth \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.BASE_LR 0.00000625 \
  SOLVER.MAX_ITER 100 \
  TEST.EVAL_PERIOD 100 \
  OUTPUT_DIR output/construction_area_debug_bs1
```

## 7. 正式训练

短跑测试通过后，使用配置中的完整训练迭代数：

```bash
PROJECT_ROOT=~/autodl-tmp/disaster_maskdino
cd "$PROJECT_ROOT/MaskDINO"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_coco_instance.pth \
  OUTPUT_DIR output/construction_area_r50
```

若正式训练使用 `IMS_PER_BATCH 1`，同时覆盖学习率：

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_coco_instance.pth \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.BASE_LR 0.00000625 \
  OUTPUT_DIR output/construction_area_r50_bs1
```

## 8. 评估训练后的模型

以正式训练生成的最终模型为例：

```bash
PROJECT_ROOT=~/autodl-tmp/disaster_maskdino
cd "$PROJECT_ROOT/MaskDINO"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

python train_net.py \
  --eval-only \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class.yaml \
  MODEL.WEIGHTS output/construction_area_r50/model_final.pth \
  OUTPUT_DIR output/construction_area_r50_eval
```

实例分割主要关注验证集的 mask `AP`、`AP50` 和 `AP75`，并结合可视化结果检查
大面积施工区域漏检、道路/裸土误检以及边界质量。

## 9. 工具测试

数据处理工具的单元测试可在项目根目录执行：

```bash
python -m unittest datasets.tests.test_construction_area_tools -v
```
