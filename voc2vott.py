import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid1
import json

COLORS = [
    "#808000", "#800000", "#FFFF00", "#FF0000",
    "#8F0080", "#8F0000", "#8FFF00", "#8F0000",
    "#8F0F80", "#8F0F00", "#8FFFF0", "#8F0F00",
]


class Voc2Vott:
    TAGS_LIST = ["person", "item"]

    def __init__(self, voc_dir, vott_dir):
        self.voc_dir = os.path.expanduser(voc_dir)
        self.vott_dir = os.path.expanduser(vott_dir)

    def convert(self):
        # analyze .vott
        vott_dir = Path(self.vott_dir)
        _tmp_vott = list(vott_dir.glob("*.vott"))
        if len(_tmp_vott) != 1:
            print("Error, please specify a correct destination directory")
            return False
        dst_vott = str(_tmp_vott[0].absolute())

        with open(dst_vott) as fp:
            vott_data = json.load(fp)
            vott_assets = {}
            vott_data['tags'] = [{
                "name": tag,
                "color": COLORS[ind % len(COLORS)]
            } for ind, tag in enumerate(Voc2Vott.TAGS_LIST)]

        # convert xml files
        voc_dir = Path(self.voc_dir)
        ann_dir = voc_dir.joinpath('Annotations')
        for ind, ann in enumerate(ann_dir.glob("*.xml")):
            ann_path = str(ann.absolute())
            asset_data = self.read_data_from_xml(ann_path)
            asset_id = asset_data['asset']['id']
            if ind == 0:
                vott_data['lastVisitedAssetId'] = asset_id
            vott_assets[asset_id] = asset_data['asset']
            tmp_name = "{}-asset.json".format(asset_id)
            tmp_path = str(vott_dir.joinpath(tmp_name))
            print(tmp_path)
            with open(tmp_path, 'w') as tfp:
                json.dump(asset_data, tfp, indent=4, sort_keys=True)

        vott_data['assets'] = vott_assets
        with open(dst_vott + '.json', "w") as ofp:
            json.dump(vott_data, ofp, indent=4, sort_keys=True)

    def read_data_from_xml(self, ann_path):
        image_dir = Path(ann_path).parent.parent.joinpath("JPEGImages")
        tree = ET.parse(ann_path)
        annotation_root = tree.getroot()

        path = annotation_root.findtext('path')
        if path is None:
            name = annotation_root.findtext('filename')
            path = os.path.join(image_dir, name)
        else:
            name = os.path.basename(path)

        size = annotation_root.find("size")
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        ext = name.split('.')[-1]

        results = {
            "asset": {
                "name": name,
                "path": "file:" + path,
                "size": {
                    "width": width,
                    "height": height
                },
                "id": str(uuid1()).replace("-", ""),
                "format": ext,
                "state": 2,
                "type": 1
            },
            "regions": [],
            "version": "2.1.0"
        }

        for obj in annotation_root.findall('object'):
            obj_bnd_box = obj.find('bndbox')
            print(obj_bnd_box.findtext('xmin'))
            xmin = float(obj_bnd_box.findtext('xmin'))
            ymin = int(float(obj_bnd_box.findtext('ymin')))
            xmax = int(float(obj_bnd_box.findtext('xmax')))
            ymax = int(float(obj_bnd_box.findtext('ymax')))

            temp_region = {
                'id': str(uuid1()).replace("-", ""),
                'type': "RECTANGLE",
                'tags': [str(obj.find('name').text)],
                "boundingBox": {
                    "height": ymax - ymin,
                    "width": xmax - xmin,
                    "left": xmin,
                    "top": ymin
                },
                'points': [
                    {
                        "x": xmin,
                        "y": ymin
                    },
                    {
                        "x": xmax,
                        "y": ymin
                    },
                    {
                        "x": xmax,
                        "y": ymax
                    },
                    {
                        "x": xmin,
                        "y": ymax
                    }
                ]
            }

            results['regions'].append(temp_region)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_dir", type=str)
    parser.add_argument("--vott_dir", type=str)
    args = parser.parse_args()

    voc2vott = Voc2Vott(
        voc_dir=args.voc_dir,  # "~/Documents/datasets/item/pascal_voc/test/",
        vott_dir=args.vott_dir  # "~/Documents/test/annotations"
    )
    voc2vott.convert()


if __name__ == "__main__":
    main()
