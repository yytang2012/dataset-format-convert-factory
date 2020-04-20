# Script to convert yolo annotations to voc format
import os
import random
import shutil
import xml.etree.cElementTree as ET
from pathlib import Path

from PIL import Image


class YOLO2VOC:
    def __init__(self, yolo_dir, voc_dir, class_mapping):
        self.yolo_dir = os.path.expanduser(yolo_dir)
        voc_dir = os.path.expanduser(voc_dir)
        if os.path.isdir(voc_dir) is False:
            os.makedirs(voc_dir, exist_ok=True)

        voc_images_dir = os.path.join(voc_dir, "JPEGImages")
        if os.path.isdir(voc_images_dir) is False:
            os.makedirs(voc_images_dir, exist_ok=True)
        self.voc_images_dir = voc_images_dir

        voc_ann_dir = os.path.join(voc_dir, "Annotations")
        if os.path.isdir(voc_ann_dir) is False:
            os.makedirs(voc_ann_dir, exist_ok=True)
        self.voc_ann_dir = voc_ann_dir

        voc_imagesets_main_dir = os.path.join(voc_dir, "ImageSets", "Main")
        if os.path.isdir(voc_imagesets_main_dir) is False:
            os.makedirs(voc_imagesets_main_dir, exist_ok=True)
        self.voc_imagesets_main_dir = voc_imagesets_main_dir

        self.dataset_name = yolo_dir.split(os.sep)[-1]
        self.class_mapping = class_mapping

    def convert(self, extension='.jpg', train_val_percent=1, train_percent=0.8):
        yolo_dir = Path(self.yolo_dir)
        for yolo in yolo_dir.glob("*.txt"):
            yolo_file = str(yolo.absolute())
            print("Processing {}".format(yolo_file))
            prefix = yolo_file[:-4]
            image_path = prefix + extension
            voc_labels = self.extract_from_yolo_file(yolo_file, image_path)
            self.create_voc_dataset(voc_labels, image_path)

        # split train, validate, test dataset
        self.split_train_val_test(train_val_percent=train_val_percent, train_percent=train_percent)

    def extract_from_yolo_file(self, yolo_file, image_path):
        img = Image.open(image_path)
        width, height = img.size
        with open(yolo_file, 'r') as fp:
            voc_labels = []
            for line in fp.readlines():
                class_id, center_x, center_y, bbox_width, bbox_height = line.split(None)[:5]
                class_name = self.class_mapping[class_id]
                bbox_width = float(bbox_width) * width
                bbox_height = float(bbox_height) * height
                center_x = float(center_x) * width
                center_y = float(center_y) * height
                xmin = max(int(center_x - (bbox_width / 2)), 1)
                ymin = max(int(center_y - (bbox_height / 2)), 1)
                xmax = min(int(center_x + (bbox_width / 2)), width - 1)
                ymax = min(int(center_y + (bbox_height / 2)), height - 1)
                voc_labels.append([class_name, xmin, ymin, xmax, ymax])
            return voc_labels

    def create_voc_dataset(self, voc_labels, image_path):
        img = Image.open(image_path)
        width, height = img.size

        image_file = image_path.split(os.sep)[-1]
        new_image_path = os.path.join(self.voc_images_dir, image_file)

        image_id = image_file.split('.')[0]
        voc_ann_xml = os.path.join(self.voc_ann_dir, image_id + '.xml')

        # 1. Construct JPEGImages folder
        shutil.copy(image_path, new_image_path)

        # 2. Construct Annotations folder
        root = ET.Element("annotations")
        ET.SubElement(root, "filename").text = image_file
        ET.SubElement(root, "folder").text = self.dataset_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        for name, xmin, ymin, xmax, ymax in voc_labels:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = str(0)
            ET.SubElement(obj, "difficult").text = str(0)
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(xmin)
            ET.SubElement(bbox, "ymin").text = str(ymin)
            ET.SubElement(bbox, "xmax").text = str(xmax)
            ET.SubElement(bbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(root)
        tree.write(voc_ann_xml)

    def split_train_val_test(self, train_val_percent, train_percent):
        voc_ann_dir = Path(self.voc_ann_dir)
        total_samples = {voc_ann_xml.name[:-4] for voc_ann_xml in voc_ann_dir.glob("*.xml")}
        total_num = len(total_samples)
        train_val_num = int(total_num * train_val_percent)
        train_num = int(train_val_num * train_percent)

        train_val_samples = set(random.sample(total_samples, train_val_num))
        test_samples = total_samples.difference(train_val_samples)
        train_samples = set(random.sample(train_val_samples, train_num))
        val_samples = set(train_val_samples).difference(train_samples)

        if len(train_val_samples) > 0:
            train_val_path = os.path.join(self.voc_imagesets_main_dir, "trainval.txt")
            with open(train_val_path, 'w') as fp:
                for _sample in train_val_samples:
                    fp.write(_sample + '\n')

        if len(train_samples) > 0:
            train_path = os.path.join(self.voc_imagesets_main_dir, "train.txt")
            with open(train_path, 'w') as fp:
                for _sample in train_samples:
                    fp.write(_sample + '\n')

        if len(val_samples) > 0:
            val_path = os.path.join(self.voc_imagesets_main_dir, "val.txt")
            with open(val_path, 'w') as fp:
                for _sample in val_samples:
                    fp.write(_sample + '\n')

        if len(test_samples) > 0:
            test_path = os.path.join(self.voc_imagesets_main_dir, "test.txt")
            with open(test_path, 'w') as fp:
                for _sample in test_samples:
                    fp.write(_sample + '\n')


def main():
    YOLO_DIR = "/home/yytang/Documents/datasets/Real_Store_ItemDetection_Data/test/TEST_DATASET"

    VOC_DIR = "~/Documents/datasets/item/pascal_voc/test"

    CLASS_MAPPING = {
        '0': 'person',
        '1': 'item'
        # Add your remaining classes here.
    }

    yolo2voc = YOLO2VOC(YOLO_DIR, VOC_DIR, CLASS_MAPPING)
    yolo2voc.convert()
    print("Done")


if __name__ == "__main__":
    main()
