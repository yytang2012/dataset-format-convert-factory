import os
from collections import defaultdict
from pathlib import Path

import cv2
from pycocotools.coco import COCO

from visualize.draw_settings import CYAN
from visualize.draw_utils import apply_polygon, apply_text_left


class ViewDataset:

    def yolo(self, yolo_dir, ext='.jpg', class2id=None):
        data = []
        yolo_dir = Path(yolo_dir)
        for label_txt in yolo_dir.glob("*.txt"):
            image_base = str(label_txt.absolute())[:-4]
            image_path = image_base + ext
            if os.path.isfile(image_path) is True:
                image = cv2.imread(image_path)
                height, width = image.shape[:2]

                with open(str(label_txt.absolute()), 'r') as fp:
                    labels = []
                    for line in fp.readlines():
                        class_id, center_x, center_y, bbox_width, bbox_height = line.split(None)[:5]
                        class_name = class2id[class_id] if class2id else class_id
                        bbox_width = float(bbox_width) * width
                        bbox_height = float(bbox_height) * height
                        center_x = float(center_x) * width
                        center_y = float(center_y) * height
                        xmin = max(int(center_x - (bbox_width / 2)), 1)
                        ymin = max(int(center_y - (bbox_height / 2)), 1)
                        xmax = min(int(center_x + (bbox_width / 2)), width - 1)
                        ymax = min(int(center_y + (bbox_height / 2)), height - 1)
                        vertices = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                        labels.append([class_name, vertices])
                    data.append({
                        "image_path": image_path,
                        "labels": labels
                    })

        self.view_data(data, window_name="YOLO")

    def pascal_voc(self, voc_dir):
        pass

    def coco(self, coco_dir, coco_set_name):
        coco_dir = os.path.expanduser(coco_dir)
        _coco = COCO(os.path.join(coco_dir, 'annotations', 'instances_' + coco_set_name + '.json'))
        image_ids = _coco.getImgIds()
        annotations_ids = _coco.getAnnIds(imgIds=image_ids, iscrowd=False)
        coco_annotations = _coco.loadAnns(annotations_ids)
        class2id = {item["id"]: item["name"] for item in _coco.cats.values()}

        image2label = defaultdict(lambda: [])
        for ann in coco_annotations:
            xmin, ymin, b_width, b_height = ann["bbox"]
            xmax = xmin + b_width
            ymax = ymin + b_height
            vertices = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            class_name = class2id[ann["category_id"]]
            image2label[ann["image_id"]].append([class_name, vertices])
        # for image_id, labels in image2label.items():
        #     image_path =

        images = _coco.loadImgs(image_ids)
        data = []
        for image in images:
            image_id = image["id"]
            image_path = os.path.join(coco_dir, "images", coco_set_name, image["file_name"])
            data.append({
                "image_path": image_path,
                "labels": image2label[image_id]
            })
        self.view_data(data, window_name="COCO")

    def view_data(self, data, window_name="Data", show_name=True):
        ind = 0
        while True:
            image_info = data[ind]
            image_path = image_info["image_path"]
            image = cv2.imread(image_path)
            labels = image_info["labels"]
            for name, vertices in labels:
                image = apply_polygon(image, vertices)
                if show_name is True:
                    (x1, y1), _, (x2, y2), _ = vertices
                    image = apply_text_left(
                        image=image,
                        bbox=(y1, x1, y2, x2),
                        color=CYAN,
                        text=name,
                        font_scale=0.5
                    )
            cv2.imshow(window_name, image)

            key = cv2.waitKeyEx(0)
            key = key & 0xFF
            if key == ord('q'):
                break
            elif key == ord('j'):
                ind = max(0, ind - 1)
            elif key == ord('k'):
                ind = min(len(data), ind + 1)


if __name__ == "__main__":
    CLASS_MAPPING = {
        '1': 'person',
        '2': 'item'
        # Add your remaining classes here.
    }
    # yolo_dir = "/home/yytang/Documents/datasets/Real_Store_ItemDetection_Data/test/TEST_DATASET"
    view_data = ViewDataset()
    # view_data.yolo(yolo_dir, class2id=CLASS_MAPPING)

    coco_dir = "~/Documents/datasets/item/coco"
    view_data.coco(coco_dir, "test")
