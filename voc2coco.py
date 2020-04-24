import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm
import xml.etree.ElementTree as ET


class VOC2COCO:
    def __init__(self, voc_dir, coco_dir, coco_set_name, label2id):
        self.voc_dir = os.path.expanduser(voc_dir)
        self.coco_dir = os.path.expanduser(coco_dir)
        self.coco_set_name = coco_set_name
        self.label2id = label2id

    def convert(self, extension='.jpg'):
        self.prepare_images(no_copy=True, extension=extension)
        self.create_coco_annotation()

    def prepare_images(self, no_copy=True, extension='.jpg'):
        voc_image_dir = os.path.join(self.voc_dir, "JPEGImages")
        voc_image_dir = Path(voc_image_dir)
        coco_image_dir = os.path.join(self.coco_dir, "images", self.coco_set_name)
        if os.path.isdir(coco_image_dir) is False:
            os.makedirs(coco_image_dir, exist_ok=True)
        for image in voc_image_dir.glob("*{}".format(extension)):
            src_path = str(image.absolute())
            dst_path = os.path.join(coco_image_dir, str(image.name))
            if no_copy is True:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
        print("Done processing image files")

    def create_coco_annotation(self):
        coco_ann_json = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        voc_ann_dir = os.path.join(self.voc_dir, "Annotations")
        voc_ann_dir = Path(voc_ann_dir)
        ann_paths = [str(ann.absolute()) for ann in voc_ann_dir.glob("*.xml")]

        bnd_id = 1  # START_BOUNDING_BOX_ID
        image_id = 0
        print('Start converting !')
        for ann_path in tqdm(ann_paths):
            image_id += 1
            # Read annotation xml
            ann_tree = ET.parse(ann_path)
            ann_root = ann_tree.getroot()
            image_info = self.get_image_info(ann_root)
            image_info.update({"id": image_id})
            coco_ann_json["images"].append(image_info)

            for obj in ann_root.findall('object'):
                ann = self.get_coco_annotation_from_obj(obj=obj)
                ann.update({'image_id': image_id, 'id': bnd_id})
                coco_ann_json['annotations'].append(ann)
                bnd_id = bnd_id + 1

        for label, label_id in self.label2id.items():
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
            coco_ann_json['categories'].append(category_info)

        coco_ann_dir = os.path.join(self.coco_dir, "annotations")
        if os.path.isdir(coco_ann_dir) is False:
            os.makedirs(coco_ann_dir, exist_ok=True)
        coco_ann_path = os.path.join(coco_ann_dir, "instances_" + self.coco_set_name + '.json')

        with open(coco_ann_path, 'w') as fp:
            json.dump(coco_ann_json, fp, indent=4)

    def get_image_info(self, annotation_root):
        path = annotation_root.findtext('path')
        if path is None:
            filename = annotation_root.findtext('filename')
        else:
            filename = os.path.basename(path)

        size = annotation_root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))

        image_info = {
            'file_name': filename,
            'height': height,
            'width': width
        }
        return image_info

    def get_coco_annotation_from_obj(self, obj):
        label = obj.findtext('name')
        assert label in self.label2id, f"Error: {label} is not in label2id !"
        category_id = self.label2id[label]
        bndbox = obj.find('bndbox')
        xmin = max(float(bndbox.findtext('xmin')) - 1.0, 1.0)
        ymin = max(float(bndbox.findtext('ymin')) - 1.0, 1.0)
        xmax = float(bndbox.findtext('xmax'))
        ymax = float(bndbox.findtext('ymax'))
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        ann = {
            'area': o_width * o_height,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
        }
        return ann


if __name__ == '__main__':
    # train set
    VOC_DIR = "~/Documents/datasets/item/pascal_voc/train"
    COCO_DIR = "~/Documents/datasets/item/coco"
    coco_set_name = "train"

    # # test set
    # VOC_DIR = "~/Documents/datasets/item/pascal_voc/test"
    # COCO_DIR = "~/Documents/datasets/item/coco"
    # coco_set_name = "test"
    label2id = {
        "person": 1,
        "item": 2
    }
    voc2coco = VOC2COCO(VOC_DIR, COCO_DIR, coco_set_name, label2id)
    voc2coco.convert()
