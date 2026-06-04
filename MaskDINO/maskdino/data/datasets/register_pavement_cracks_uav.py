import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json


_PREDEFINED_SPLITS = {
    "pavement_cracks_uav_train": (
        "Pavement_cracks _UAV_imagery-1000/train/images",
        "Pavement_cracks _UAV_imagery-1000/annotations/instances_train.json",
    ),
    "pavement_cracks_uav_val": (
        "Pavement_cracks _UAV_imagery-1000/val/images",
        "Pavement_cracks _UAV_imagery-1000/annotations/instances_val.json",
    ),
    "pavement_cracks_uav_test": (
        "Pavement_cracks _UAV_imagery-1000/test/images",
        "Pavement_cracks _UAV_imagery-1000/annotations/instances_test.json",
    ),
}

_THIN_CRACK_COCO_CATEGORY_IDS = {1, 2, 4}


def get_pavement_cracks_uav_meta():
    """
    UAV pavement crack 数据集的 COCO 类别映射。

    原始 YOLO class id 为 0..3，COCO JSON 中写成 1..4；
    Detectron2 再映射回模型内部连续 id 0..3。
    """
    return {
        "thing_dataset_id_to_contiguous_id": {1: 0, 2: 1, 3: 2, 4: 3},
        "thing_classes": [
            "transverse_crack",
            "longitudinal_crack",
            "alligator_crack",
            "oblique_crack",
        ],
    }


def register_all_pavement_cracks_uav(root):
    for dataset_name, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_coco_instances(
            dataset_name,
            get_pavement_cracks_uav_meta(),
            os.path.join(root, json_file),
            os.path.join(root, image_root),
        )

    train_image_root, train_json_file = _PREDEFINED_SPLITS["pavement_cracks_uav_train"]
    repeated_train_name = "pavement_cracks_uav_train_repeat_thin"
    DatasetCatalog.register(
        repeated_train_name,
        lambda: load_pavement_cracks_uav_repeated(
            os.path.join(root, train_json_file),
            os.path.join(root, train_image_root),
        ),
    )
    MetadataCatalog.get(repeated_train_name).set(
        json_file=os.path.join(root, train_json_file),
        image_root=os.path.join(root, train_image_root),
        evaluator_type="coco",
        **get_pavement_cracks_uav_meta(),
    )


def load_pavement_cracks_uav_repeated(json_file, image_root):
    """
    在 dataset dict 层面对三类细长裂缝做温和重复采样。

    不修改 COCO 标注文件；只在训练集读取后，将包含 transverse、
    longitudinal 或 oblique crack 的图片额外重复一次，提升这些类的采样概率。
    """
    dataset_dicts = load_coco_json(json_file, image_root, "pavement_cracks_uav_train")
    repeated = []
    for record in dataset_dicts:
        repeated.append(record)
        category_ids = {ann["category_id"] for ann in record.get("annotations", [])}
        if category_ids & _THIN_CRACK_COCO_CATEGORY_IDS:
            repeated.append(record.copy())
    return repeated


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pavement_cracks_uav(_root)
