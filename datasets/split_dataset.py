import os
import random
import shutil
from pathlib import Path

random.seed(42)

root = Path(__file__).resolve().parent / "construction_area"
image_dir = root / "image"
label_dir = root / "label"

out_img_train = root / "images/train"
out_img_val = root / "images/val"
out_lbl_train = root / "labels/train"
out_lbl_val = root / "labels/val"

for p in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
    p.mkdir(parents=True, exist_ok=True)

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]

random.shuffle(images)

val_ratio = 0.2
val_num = int(len(images) * val_ratio)

val_set = set(images[:val_num])
train_set = set(images[val_num:])

def copy_pair(img_path, img_out_dir, lbl_out_dir):
    shutil.copy2(img_path, img_out_dir / img_path.name)

    label_path = label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, lbl_out_dir / label_path.name)
    else:
        # 允许负样本没有标签
        (lbl_out_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")

for img in train_set:
    copy_pair(img, out_img_train, out_lbl_train)

for img in val_set:
    copy_pair(img, out_img_val, out_lbl_val)

print(f"Total images: {len(images)}")
print(f"Train images: {len(train_set)}")
print(f"Val images: {len(val_set)}")