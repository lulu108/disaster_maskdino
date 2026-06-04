import os

from detectron2.data.datasets import register_coco_instances


_PREDEFINED_SPLITS = {
    "construction_area_train": (
        "construction_area/images/train",
        "construction_area/annotations/instances_train.json",
    ),
    "construction_area_val": (
        "construction_area/images/val",
        "construction_area/annotations/instances_val.json",
    ),
}


def get_construction_area_meta():
    return {
        "thing_dataset_id_to_contiguous_id": {1: 0},
        "thing_classes": ["construction_area"],
    }


def register_all_construction_area(root):
    for dataset_name, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_coco_instances(
            dataset_name,
            get_construction_area_meta(),
            os.path.join(root, json_file),
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_construction_area(_root)
