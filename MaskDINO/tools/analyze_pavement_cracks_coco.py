import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_coco(json_file):
    """
    统计 UAV pavement crack COCO 标注的类别分布和每图类别。

    该脚本只读 COCO JSON，不修改原始数据或标注；用于判断细长裂缝类是否需要
    repeat/oversample，以及验证负样本数量是否合理。
    """
    coco = json.loads(json_file.read_text(encoding="utf-8"))
    id_to_name = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image = defaultdict(list)
    instance_counts = Counter()
    image_counts = Counter()

    for annotation in coco["annotations"]:
        category_name = id_to_name[annotation["category_id"]]
        instance_counts[category_name] += 1
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_class_sets = {}
    for image in coco["images"]:
        classes = {
            id_to_name[annotation["category_id"]]
            for annotation in annotations_by_image.get(image["id"], [])
        }
        image_class_sets[image["file_name"]] = sorted(classes)
        for category_name in classes:
            image_counts[category_name] += 1

    background_count = sum(1 for classes in image_class_sets.values() if not classes)
    return coco, instance_counts, image_counts, background_count, image_class_sets


def main():
    parser = argparse.ArgumentParser(
        description="Analyze UAV pavement crack COCO class and image-level distribution."
    )
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--show-images", action="store_true")
    args = parser.parse_args()

    coco, instance_counts, image_counts, background_count, image_class_sets = analyze_coco(
        args.json_file
    )
    num_images = len(coco["images"])
    print(f"json: {args.json_file}")
    print(f"images: {num_images}")
    print(f"annotations: {len(coco['annotations'])}")
    print(f"background images: {background_count}")
    print("classes:")
    for category in coco["categories"]:
        name = category["name"]
        image_count = image_counts[name]
        frequency = image_count / num_images if num_images else 0.0
        print(
            f"  {category['id']:>2} {name:<20} "
            f"instances={instance_counts[name]:>5} images={image_count:>5} "
            f"image_freq={frequency:.4f}"
        )

    if args.show_images:
        print("image_classes:")
        for file_name, classes in sorted(image_class_sets.items()):
            print(f"  {file_name}: {','.join(classes) if classes else 'background'}")


if __name__ == "__main__":
    main()
