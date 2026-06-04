import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


DATASETS_DIR = Path(__file__).resolve().parents[1]


def load_script(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, DATASETS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PavementCracksYoloToCocoTest(unittest.TestCase):
    def test_convert_split_preserves_missing_label_as_negative(self):
        converter = load_script("pavement_cracks_yolo_to_coco", "pavement_cracks_yolo_to_coco.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            Image.new("RGB", (100, 50)).save(image_dir / "positive.jpg")
            Image.new("RGB", (80, 80)).save(image_dir / "background.jpg")
            (label_dir / "positive.txt").write_text(
                "2 0.50 0.50 0.40 0.20\n",
                encoding="utf-8",
            )

            coco, missing_labels = converter.convert_split(
                root, "train", converter.DEFAULT_CLASS_NAMES, "empty"
            )

        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(missing_labels, 1)
        self.assertEqual(coco["annotations"][0]["category_id"], 3)
        self.assertEqual(coco["annotations"][0]["bbox"], [30.0, 20.0, 40.0, 10.0])
        self.assertEqual(coco["annotations"][0]["segmentation"][0], [30.0, 20.0, 70.0, 20.0, 70.0, 30.0, 30.0, 30.0])

    def test_convert_dataset_writes_json(self):
        converter = load_script("pavement_cracks_yolo_to_coco", "pavement_cracks_yolo_to_coco.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ["train", "val"]:
                image_dir = root / split / "images"
                label_dir = root / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                Image.new("RGB", (10, 10)).save(image_dir / f"{split}.jpg")
                (label_dir / f"{split}.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

            summaries = converter.convert_dataset(
                root, converter.DEFAULT_CLASS_NAMES, ["train", "val"], "empty"
            )
            train_json = root / "annotations" / "instances_train.json"

            self.assertEqual(len(summaries), 2)
            self.assertTrue(train_json.exists())
            self.assertEqual(json.loads(train_json.read_text(encoding="utf-8"))["categories"][0]["name"], "transverse_crack")


if __name__ == "__main__":
    unittest.main()
