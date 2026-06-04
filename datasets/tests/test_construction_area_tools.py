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


class YoloToCocoTest(unittest.TestCase):
    def test_convert_split_scales_polygon_and_preserves_empty_images(self):
        converter = load_script("yolo_to_coco", "yolo_to_coco.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            Image.new("RGB", (100, 50)).save(image_dir / "positive.jpg")
            Image.new("RGB", (80, 80)).save(image_dir / "negative.jpg")
            (label_dir / "positive.txt").write_text(
                "0 0.10 0.20 0.90 0.20 0.90 0.80 0.10 0.80\n",
                encoding="utf-8",
            )
            (label_dir / "negative.txt").write_text("", encoding="utf-8")

            coco = converter.convert_split(image_dir, label_dir, "construction_area")

        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(coco["categories"], [{"id": 1, "name": "construction_area", "supercategory": "construction_area"}])
        annotation = coco["annotations"][0]
        self.assertEqual(annotation["category_id"], 1)
        self.assertEqual(annotation["segmentation"][0], [10.0, 10.0, 90.0, 10.0, 90.0, 40.0, 10.0, 40.0])
        self.assertEqual(annotation["bbox"], [10.0, 10.0, 80.0, 30.0])
        self.assertEqual(annotation["area"], 2400.0)

    def test_convert_split_rejects_unknown_class_id(self):
        converter = load_script("yolo_to_coco", "yolo_to_coco.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (10, 10)).save(image_dir / "bad.jpg")
            (label_dir / "bad.txt").write_text("1 0 0 1 0 1 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "class id 0"):
                converter.convert_split(image_dir, label_dir, "construction_area")


class VisualizeCocoTest(unittest.TestCase):
    def test_render_samples_creates_overlay_image(self):
        visualizer = load_script("visualize_coco_annotations", "visualize_coco_annotations.py")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            output_dir = root / "visualized"
            image_dir.mkdir()
            Image.new("RGB", (40, 40), "white").save(image_dir / "scene.jpg")
            coco = {
                "images": [{"id": 1, "file_name": "scene.jpg", "width": 40, "height": 40}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[5, 5, 35, 5, 35, 35, 5, 35]],
                        "bbox": [5, 5, 30, 30],
                        "area": 900,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "construction_area"}],
            }
            json_file = root / "instances.json"
            json_file.write_text(json.dumps(coco), encoding="utf-8")

            outputs = visualizer.render_samples(
                json_file, image_dir, output_dir, num_samples=1, seed=42
            )

            self.assertEqual(outputs, [output_dir / "scene.jpg"])
            self.assertTrue(outputs[0].exists())


if __name__ == "__main__":
    unittest.main()
