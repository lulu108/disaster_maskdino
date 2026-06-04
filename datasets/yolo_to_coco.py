import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
YOLO_CLASS_ID = 0
COCO_CATEGORY_ID = 1


def polygon_area(segmentation):
    points = list(zip(segmentation[0::2], segmentation[1::2]))
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_annotation(line, width, height, label_path, line_number):
    fields = line.split()
    if not fields:
        return None
    try:
        class_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: invalid class id") from exc
    if class_id != YOLO_CLASS_ID:
        raise ValueError(
            f"{label_path}:{line_number}: expected YOLO class id 0, got {class_id}"
        )

    try:
        normalized = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: invalid polygon coordinate") from exc
    if len(normalized) < 6 or len(normalized) % 2:
        raise ValueError(
            f"{label_path}:{line_number}: polygon must contain at least three x/y points"
        )
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError(
            f"{label_path}:{line_number}: polygon coordinates must be normalized in [0, 1]"
        )

    segmentation = []
    for index in range(0, len(normalized), 2):
        segmentation.extend(
            [round(normalized[index] * width, 4), round(normalized[index + 1] * height, 4)]
        )

    xs = segmentation[0::2]
    ys = segmentation[1::2]
    area = round(polygon_area(segmentation), 4)
    if area <= 0:
        raise ValueError(f"{label_path}:{line_number}: polygon area must be positive")

    return {
        "category_id": COCO_CATEGORY_ID,
        "segmentation": [segmentation],
        "bbox": [
            min(xs),
            min(ys),
            round(max(xs) - min(xs), 4),
            round(max(ys) - min(ys), 4),
        ],
        "area": area,
        "iscrowd": 0,
    }


def convert_split(image_dir, label_dir, class_name):
    images = []
    annotations = []
    annotation_id = 1
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_id, image_path in enumerate(image_paths, start=1):
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
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            annotation = parse_annotation(line.strip(), width, height, label_path, line_number)
            if annotation is None:
                continue
            annotation.update({"id": annotation_id, "image_id": image_id})
            annotations.append(annotation)
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": COCO_CATEGORY_ID,
                "name": class_name,
                "supercategory": class_name,
            }
        ],
    }


def convert_dataset(dataset_root, class_name, splits):
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in splits:
        coco = convert_split(
            dataset_root / "images" / split,
            dataset_root / "labels" / split,
            class_name,
        )
        output_path = annotations_dir / f"instances_{split}.json"
        output_path.write_text(
            json.dumps(coco, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        summaries.append((split, len(coco["images"]), len(coco["annotations"]), output_path))
    return summaries


def main():
    default_root = Path(__file__).resolve().parent / "construction_area"
    parser = argparse.ArgumentParser(
        description="Convert construction-area YOLO polygon labels to COCO instance annotations."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--class-name", default="construction_area")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    summaries = convert_dataset(args.dataset_root, args.class_name, args.splits)
    for split, image_count, annotation_count, output_path in summaries:
        print(
            f"{split}: {image_count} images, {annotation_count} annotations -> {output_path}"
        )


if __name__ == "__main__":
    main()
