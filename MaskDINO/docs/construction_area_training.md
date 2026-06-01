# Construction Area MaskDINO Training Guide

This document records the official training command for the `construction_area` one-class instance segmentation task based on MaskDINO-R50.

## 1. Project directory

Run all commands from the MaskDINO root directory:

```bash
cd /root/autodl-tmp/disaster_maskdino/MaskDINO
```

The expected project layout is:

```text
/root/autodl-tmp/disaster_maskdino/
├── MaskDINO/
│   ├── train_net.py
│   ├── configs/construction_area/
│   ├── pretrained/
│   └── output/
└── datasets/
    └── construction_area/
        ├── train/
        ├── val/
        └── annotations/
            ├── instances_train.json
            └── instances_val.json
```

## 2. Environment variables

```bash
export DETECTRON2_DATASETS=/root/autodl-tmp/disaster_maskdino/datasets
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

`DETECTRON2_DATASETS` tells Detectron2 where the dataset root is located.

`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` helps reduce CUDA memory fragmentation.

## 3. Recommended official training command

The simplified command uses the dedicated 24GB-friendly config file:

```bash
cd /root/autodl-tmp/disaster_maskdino/MaskDINO
export DETECTRON2_DATASETS=/root/autodl-tmp/disaster_maskdino/datasets
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class_24g.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
```

## 4. Background training with nohup

For long training jobs, use `nohup`:

```bash
cd /root/autodl-tmp/disaster_maskdino/MaskDINO
export DETECTRON2_DATASETS=/root/autodl-tmp/disaster_maskdino/datasets
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p output/construction_area_r50_bs1_img512_q100_p4096

nohup python -u train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class_24g.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth \
  > output/construction_area_r50_bs1_img512_q100_p4096/train.log 2>&1 &
```

Check logs:

```bash
tail -f output/construction_area_r50_bs1_img512_q100_p4096/train.log
```

Check GPU usage:

```bash
watch -n 2 nvidia-smi
```

Stop training if needed:

```bash
pkill -f train_net.py
```

## 5. Important training settings

The file `configs/construction_area/maskdino_R50_1class_24g.yaml` contains the following key settings:

```yaml
INPUT:
  IMAGE_SIZE: 512
  MIN_SCALE: 0.5
  MAX_SCALE: 1.0

SOLVER:
  IMS_PER_BATCH: 1
  BASE_LR: 0.00000625
  MAX_ITER: 54000
  STEPS: (44000, 50000)
  CHECKPOINT_PERIOD: 1000

MODEL:
  MaskDINO:
    NUM_OBJECT_QUERIES: 100
    TRAIN_NUM_POINTS: 4096
    DN_NUM: 50

TEST:
  EVAL_PERIOD: 1000
```

## 6. Why these settings are used

The original COCO MaskDINO-R50 configuration is designed for large-scale 80-class instance segmentation. It uses large training resolution, 300 object queries, 12544 mask sampling points, and 100 denoising queries. These settings can easily cause CUDA out-of-memory errors on a 24GB single GPU.

For the one-class `construction_area` task, the following reductions are used:

| Setting | Original COCO setting | Current setting | Purpose |
|---|---:|---:|---|
| Batch size | 2 or larger | 1 | Reduce GPU memory usage |
| Image size | 1024 | 512 | Reduce spatial feature memory |
| Object queries | 300 | 100 | Reduce decoder memory |
| Train points | 12544 | 4096 | Reduce mask loss memory |
| DN queries | 100 | 50 | Reduce denoising branch memory |
| Learning rate | 0.0000125 | 0.00000625 | Match batch size reduction |

## 7. Debug training command

Use this command only to verify the environment and dataset pipeline:

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/construction_area/maskdino_R50_1class_24g.yaml \
  MODEL.WEIGHTS pretrained/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth \
  SOLVER.MAX_ITER 100 \
  TEST.EVAL_PERIOD 100 \
  OUTPUT_DIR output/construction_area_debug_bs1_img512_q100_p4096
```

If this debug command completes training and evaluation, the full training command can be used.

## 8. Expected outputs

Training outputs are saved under:

```text
output/construction_area_r50_bs1_img512_q100_p4096/
```

Common files include:

```text
train.log
metrics.json
events.out.tfevents.*
model_0000999.pth
model_0001999.pth
...
model_final.pth
```

## 9. Common checks

Check whether training is running:

```bash
nvidia-smi
```

Check whether logs are updating:

```bash
tail -f output/construction_area_r50_bs1_img512_q100_p4096/train.log
```

Check whether checkpoints are generated:

```bash
ls -lh output/construction_area_r50_bs1_img512_q100_p4096/*.pth
```

## 10. Notes

Do not upload datasets, pretrained weights, checkpoints, or output folders to GitHub. They should be ignored by `.gitignore`.
