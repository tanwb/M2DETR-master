"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from torchvision.datasets import CocoDetection

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['IP102Detection']


@register
class IP102Detection(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_ip102_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_ip102_category=True):
        super(IP102Detection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_ip102_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_ip102_category = remap_ip102_category


    def __getitem__(self, idx):
        img, target = super(IP102Detection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1])  # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_ip102_category=True):
        self.return_masks = return_masks
        self.remap_ip102_category = remap_ip102_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_ip102_category:
            classes = [ip102_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


ip102_category2name = {
    1: 'rice leaf roller',
    2: 'rice leaf caterpillar',
    3: 'paddy stem maggot',
    4: 'asiatic rice borer',
    5: 'yellow rice borer',
    6: 'rice gall midge',
    7: 'Rice Stemfly',
    8: 'brown plant hopper',
    9: 'white backed plant hopper',
    10: 'small brown plant hopper',
    11: 'rice water weevil',
    12: 'rice leafhopper',
    13: 'grain spreader thrips',
    14: 'rice shell pest',
    15: 'grub',
    16: 'mole cricket',
    17: 'wireworm',
    18: 'white margined moth',
    19: 'black cutworm',
    20: 'large cutworm',
    21: 'yellow cutworm',
    22: 'red spider',
    23: 'corn borer',
    24: 'army worm',
    25: 'aphids',
    26: 'Potosiabre vitarsis',
    27: 'peach borer',
    28: 'english grain aphid',
    29: 'green bug',
    30: 'bird cherry-oataphid',
    31: 'wheat blossom midge',
    32: 'penthaleus major',
    33: 'longlegged spider mite',
    34: 'wheat phloeothrips',
    35: 'wheat sawfly',
    36: 'cerodonta denticornis',
    37: 'beet fly',
    38: 'flea beetle',
    39: 'cabbage army worm',
    40: 'beet army worm',
    41: 'Beet spot flies',
    42: 'meadow moth',
    43: 'beet weevil',
    44: 'sericaorient alismots chulsky',
    45: 'alfalfa weevil',
    46: 'flax budworm',
    47: 'alfalfa plant bug',
    48: 'tarnished plant bug',
    49: 'Locustoidea',
    50: 'lytta polita',
    51: 'legume blister beetle',
    52: 'blister beetle',
    53: 'therioaphis maculata Buckton',
    54: 'odontothrips loti',
    55: 'Thrips',
    56: 'alfalfa seed chalcid',
    57: 'Pieris canidia',
    58: 'Apolygus lucorum',
    59: 'Limacodidae',
    60: 'Viteus vitifoliae',
    61: 'Colomerus vitis',
    62: 'Brevipoalpus lewisi McGregor',
    63: 'oides decempunctata',
    64: 'Polyphagotars onemus latus',
    65: 'Pseudococcus comstocki Kuwana',
    66: 'parathrene regalis',
    67: 'Ampelophaga',
    68: 'Lycorma delicatula',
    69: 'Xylotrechus',
    70: 'Cicadella viridis',
    71: 'Miridae',
    72: 'Trialeurodes vaporariorum',
    73: 'Erythroneura apicalis',
    74: 'Papilio xuthus',
    75: 'Panonchus citri McGregor',
    76: 'Phyllocoptes oleiverus ashmead',
    77: 'Icerya purchasi Maskell',
    78: 'Unaspis yanonensis',
    79: 'Ceroplastes rubens',
    80: 'Chrysomphalus aonidum',
    81: 'Parlatoria zizyphus Lucus',
    82: 'Nipaecoccus vastalor',
    83: 'Aleurocanthus spiniferus',
    84: 'Tetradacus c Bactrocera minax',
    85: 'Dacus dorsalis(Hendel)',
    86: 'Bactrocera tsuneonis',
    87: 'Prodenia litura',
    88: 'Adristyrannus',
    89: 'Phyllocnistis citrella Stainton',
    90: 'Toxoptera citricidus',
    91: 'Toxoptera aurantii',
    92: 'Aphis citricola Vander Goot',
    93: 'Scirtothrips dorsalis Hood',
    94: 'Dasineura sp',
    95: 'Lawana imitata Melichar',
    96: 'Salurnis marginella Guerr',
    97: 'Deporaus marginatus Pascoe',
    98: 'Chlumetia transversa',
    99: 'Mango flat beak leafhopper',
    100: 'Rhytidodera bowrinii white',
    101: 'Sternochetus frigidus',
    102: 'Cicadellidae'
}

ip102_category2label = {k: i for i, k in enumerate(ip102_category2name.keys())}
ip102_label2category = {v: k for k, v in ip102_category2label.items()}
