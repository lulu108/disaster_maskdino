import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = [
    "transverse_crack",
    "longitudinal_crack",
    "alligator_crack",
    "oblique_crack",
]


def clip(value, lower, upper):
    return min(max(value, lower), upper)


def parse_yolo_bbox(line, width, height, label_path, line_number, class_names):
    """
    将 UAV pavement crack 的 YOLO bbox 标注转换为 COCO annotation。

    输入行格式为 `class_id x_center y_center width height`，坐标均为归一化值。
    输出使用 COCO bbox，并用矩形 segmentation 作为 MaskDINO 的实例分割监督。
    """
    fields = line.split()
    if not fields:
        return None
    if len(fields) != 5:
        raise ValueError(
            f"{label_path}:{line_number}: expected 5 YOLO bbox fields, got {len(fields)}"
        )

    try:
        class_id = int(fields[0])
        x_center, y_center, box_width, box_height = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: invalid YOLO bbox value") from exc

    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(
            f"{label_path}:{line_number}: class id {class_id} outside 0..{len(class_names) - 1}"
        )
    if any(value < 0.0 or value > 1.0 for value in [x_center, y_center, box_width, box_height]):
        raise ValueError(f"{label_path}:{line_number}: YOLO bbox values must be in [0, 1]")
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError(f"{label_path}:{line_number}: bbox width/height must be positive")

    # 少量标注因小数精度会落到 1.0000005，这里裁剪到图像边界以保留有效样本。
    x_min = clip((x_center - box_width / 2.0) * width, 0.0, float(width))
    y_min = clip((y_center - box_height / 2.0) * height, 0.0, float(height))
    x_max = clip((x_center + box_width / 2.0) * width, 0.0, float(width))
    y_max = clip((y_center + box_height / 2.0) * height, 0.0, float(height))
    coco_width = round(x_max - x_min, 4)
    coco_height = round(y_max - y_min, 4)
    if coco_width <= 0.0 or coco_height <= 0.0:
        raise ValueError(f"{label_path}:{line_number}: clipped bbox has non-positive area")

    x_min = round(x_min, 4)
    y_min = round(y_min, 4)
    x_max = round(x_max, 4)
    y_max = round(y_max, 4)
    segmentation = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

    return {
        "category_id": class_id + 1,
        "segmentation": [segmentation],
        "bbox": [x_min, y_min, coco_width, coco_height],
        "area": round(coco_width * coco_height, 4),
        "iscrowd": 0,
    }


def image_paths_for_split(image_dir):
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def convert_split(dataset_root, split, class_names, missing_label_policy):
    """
    转换单个 split，保留没有 label 的背景图作为空 annotation 样本。

    `missing_label_policy=empty` 适合该数据集中的 background 图片；
    `error` 可用于检查训练集是否存在意外漏标。
    """
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = []
    annotations = []
    annotation_id = 1
    missing_labels = 0

    for image_id, image_path in enumerate(image_paths_for_split(image_dir), start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            if missing_label_policy == "error":
                raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")
            continue

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            annotation = parse_yolo_bbox(
                line.strip(), width, height, label_path, line_number, class_names
            )
            if annotation is None:
                continue
            annotation.update({"id": annotation_id, "image_id": image_id})
            annotations.append(annotation)
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "pavement_crack"}
            for index, name in enumerate(class_names)
        ],
    }, missing_labels


def convert_dataset(dataset_root, class_names, splits, missing_label_policy):
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for split in splits:
        coco, missing_labels = convert_split(
            dataset_root, split, class_names, missing_label_policy
        )
        output_path = annotations_dir / f"instances_{split}.json"
        output_path.write_text(json.dumps(coco, ensure_ascii=True, indent=2), encoding="utf-8")
        summaries.append(
            (split, len(coco["images"]), len(coco["annotations"]), missing_labels, output_path)
        )
    return summaries


def main():
    default_root = Path(__file__).resolve().parent / "Pavement_cracks _UAV_imagery-1000"
    parser = argparse.ArgumentParser(
        description="Convert UAV pavement crack YOLO bbox labels to COCO instance annotations."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES)
    parser.add_argument(
        "--missing-label-policy",
        choices=["empty", "error"],
        default="empty",
        help="Treat missing label files as negative samples or fail fast.",
    )
    args = parser.parse_args()

    summaries = convert_dataset(
        args.dataset_root, args.class_names, args.splits, args.missing_label_policy
    )
    for split, image_count, annotation_count, missing_labels, output_path in summaries:
        print(
            f"{split}: {image_count} images, {annotation_count} annotations, "
            f"{missing_labels} missing-label negatives -> {output_path}"
        )


if __name__ == "__main__":
    main()
