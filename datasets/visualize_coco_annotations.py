import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def render_samples(json_file, image_dir, output_dir, num_samples, seed):
    coco = json.loads(json_file.read_text(encoding="utf-8"))
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    rng = random.Random(seed)
    images = coco["images"]
    selected = rng.sample(images, min(num_samples, len(images)))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for image_record in selected:
        source_path = image_dir / image_record["file_name"]
        with Image.open(source_path) as source:
            image = source.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        annotations = annotations_by_image[image_record["id"]]
        for annotation in annotations:
            for segmentation in annotation["segmentation"]:
                points = list(zip(segmentation[0::2], segmentation[1::2]))
                overlay_draw.polygon(
                    points, fill=(255, 50, 50, 70), outline=(255, 30, 30, 255)
                )
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        text = f"construction_area instances: {len(annotations)}"
        draw.rectangle((5, 5, 230, 26), fill=(0, 0, 0, 190))
        draw.text((9, 9), text, fill=(255, 255, 255, 255))
        output_path = output_dir / image_record["file_name"]
        image.convert("RGB").save(output_path)
        outputs.append(output_path)
    return outputs


def main():
    default_root = Path(__file__).resolve().parent / "construction_area"
    parser = argparse.ArgumentParser(
        description="Render random COCO annotation overlays for visual inspection."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.dataset_root / "visualizations" / args.split
    outputs = render_samples(
        args.dataset_root / "annotations" / f"instances_{args.split}.json",
        args.dataset_root / "images" / args.split,
        output_dir,
        args.num_samples,
        args.seed,
    )
    print(f"Rendered {len(outputs)} samples to {output_dir}")


if __name__ == "__main__":
    main()
