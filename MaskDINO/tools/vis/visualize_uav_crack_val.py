import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# MaskDINO config
from maskdino import add_maskdino_config


COCO_MEDIUM_MIN = 32 * 32
COCO_MEDIUM_MAX = 96 * 96


def setup_cfg(config_file, weights, score_thresh):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights

    # MaskDINO 主要看这个阈值，不是 ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = score_thresh

    cfg.freeze()
    return cfg


def coco_poly_or_rle_to_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return None

    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def draw_gt(image_bgr, record, metadata):
    image_rgb = image_bgr[:, :, ::-1]
    visualizer = Visualizer(
        image_rgb,
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )

    annos = record.get("annotations", [])
    labels = []
    masks = []
    boxes = []

    thing_classes = metadata.thing_classes
    id_map = metadata.thing_dataset_id_to_contiguous_id

    h, w = image_bgr.shape[:2]

    for ann in annos:
        category_id = ann["category_id"]
        contiguous_id = id_map.get(category_id, category_id)
        class_name = thing_classes[contiguous_id] if contiguous_id < len(thing_classes) else str(category_id)
        area = ann.get("area", 0)
        labels.append(f"{class_name} area={int(area)}")

        if "segmentation" in ann:
            mask = coco_poly_or_rle_to_mask(ann["segmentation"], h, w)
            if mask is not None:
                masks.append(mask)

        if "bbox" in ann:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])

    if len(masks) > 0:
        out = visualizer.overlay_instances(
            masks=masks,
            boxes=boxes if boxes else None,
            labels=labels,
        )
    else:
        out = visualizer.overlay_instances(
            boxes=boxes if boxes else None,
            labels=labels,
        )

    return out.get_image()[:, :, ::-1]


def draw_pred(image_bgr, predictor, metadata):
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    visualizer = Visualizer(
        image_bgr[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    out = visualizer.draw_instance_predictions(instances)
    return out.get_image()[:, :, ::-1], instances


def is_medium_record(record):
    for ann in record.get("annotations", []):
        area = ann.get("area", 0)
        if COCO_MEDIUM_MIN <= area < COCO_MEDIUM_MAX:
            return True
    return False


def has_class(record, target_category_ids):
    if not target_category_ids:
        return True
    for ann in record.get("annotations", []):
        if ann.get("category_id") in target_category_ids:
            return True
    return False


def make_canvas(gt_bgr, pred_bgr, title_text):
    h = max(gt_bgr.shape[0], pred_bgr.shape[0])
    w1, w2 = gt_bgr.shape[1], pred_bgr.shape[1]

    def pad_to_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        pad = np.full((target_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        return np.vstack([img, pad])

    gt_bgr = pad_to_h(gt_bgr, h)
    pred_bgr = pad_to_h(pred_bgr, h)
    canvas = np.hstack([gt_bgr, pred_bgr])

    bar = np.full((40, canvas.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        title_text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, canvas])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--json-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="output/vis_uav_crack_val")
    parser.add_argument("--dataset-name", default="pavement_cracks_uav_val")
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--only-medium", action="store_true")
    parser.add_argument(
        "--only-thin",
        action="store_true",
        help="只看 transverse/longitudinal/oblique 三类细长裂缝",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = setup_cfg(args.config_file, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(args.dataset_name)
    metadata.set(
        thing_classes=[
            "transverse_crack",
            "longitudinal_crack",
            "alligator_crack",
            "oblique_crack",
        ],
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3},
    )

    records = load_coco_json(args.json_file, args.image_root, args.dataset_name)

    target_category_ids = None
    if args.only_thin:
        target_category_ids = {1, 2, 4}

    selected = []
    for record in records:
        if args.only_medium and not is_medium_record(record):
            continue
        if not has_class(record, target_category_ids):
            continue
        selected.append(record)

    print(f"Total val images: {len(records)}")
    print(f"Selected images: {len(selected)}")
    print(f"Saving to: {output_dir}")

    for idx, record in enumerate(selected[: args.max_images]):
        image_path = record["file_name"]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[WARN] failed to read image: {image_path}")
            continue

        gt_bgr = draw_gt(image_bgr, record, metadata)
        pred_bgr, instances = draw_pred(image_bgr, predictor, metadata)

        title = (
            f"LEFT: GT | RIGHT: Prediction | "
            f"pred_instances={len(instances)} | "
            f"file={os.path.basename(image_path)}"
        )
        canvas = make_canvas(gt_bgr, pred_bgr, title)

        save_name = f"{idx:04d}_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(output_dir / save_name), canvas)
        print(f"[{idx + 1}/{min(len(selected), args.max_images)}] saved {save_name}")

    print("Done.")


if __name__ == "__main__":
    main()