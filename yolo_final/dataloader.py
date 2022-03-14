
import config
from torchvision.datasets import CocoDetection
import numpy as np
from PIL import Image, ImageFile
import os
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

scale = [13, 26, 52]
num_scales = 3
num_anchors_per_scale = 3
ignore_iou_threshold = 0.5
anchor_boxes = [
    [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
]


def centered_iou(boxes_target, boxes_prediction):
    """
    This intersection over union function assumes that the center of the boxes are aligned. Should only be used to
    calculate the correct anchor box for the bbox
    :param boxes_target:
    :param boxes_prediction:
    :return:
    """
    intersection = torch.min(boxes_target[..., 0], boxes_prediction[..., 0]) * torch.min(boxes_target[..., 1], boxes_prediction[..., 1])
    union = (boxes_target[..., 0] * boxes_target[..., 1] + boxes_prediction[..., 0] * boxes_prediction[..., 1] - intersection)

    return intersection / union


def bboxes_to_target(bboxes, targets, ignore_iou_threshold, num_anchors_per_scale, anchor_boxes, Scale):
    """
    This function transforms the bboxes (labels) into a target structure. The target is a list of different sized 4D
    tensors e.g. list(torch.shape(3, 13, 13, 6), torch.shape(3, 26, 26, 6), torch.shape(3, 52, 52, 6))
    :param bboxes:
    :param targets:
    :param ignore_iou_threshold:
    :param num_anchors_per_scale:
    :param anchors:
    :param Scale:
    :return:
    """
    for box in bboxes:
        iou_anchors = centered_iou(torch.tensor(box[2:4]), anchor_boxes)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height, class_label = box
        has_anchor = [False, False, False]

        for anchor_idx in anchor_indices:
            scale_idx = torch.div(anchor_idx, num_anchors_per_scale, rounding_mode='trunc')  # 0, 1, 2
            anchor_on_scale = anchor_idx % num_anchors_per_scale  # 0, 1, 2
            S = Scale[scale_idx]
            i, j = int(S * y), int(S * x)  # x = 0.5, S = 13 --> int(6.5) = 6
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if not anchor_taken and not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i  # Both are between [0, 1]
                width_cell, height_cell = (width * S, height * S)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                # COCO_MAPPING maps to sequential values
                targets[scale_idx][anchor_on_scale, i, j, 5] = config.COCO_MAPPING[int(class_label)]
                has_anchor[scale_idx] = True

            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_threshold:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignore this prediction

    return targets


class CustomDataset(CocoDetection):

    def __init__(self, root: str, annFile, transform=None, target_transform=None, transforms=None, catagory=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = []
        self.anchor_boxes = torch.tensor(anchor_boxes[0] + anchor_boxes[1] + anchor_boxes[2])

        if catagory is None:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            for key, imgToAnn in self.coco.imgToAnns.items():
                replAnn = []
                contains_cat = False
                for ann in imgToAnn:
                    if ann['category_id'] == catagory:
                        contains_cat = True
                        replAnn.append(ann)
                self.coco.imgToAnns[key] = replAnn
                if contains_cat:
                    self.ids.append(key)
        print()

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        labels = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        image = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))
        bboxes = [x['bbox'] + [x['category_id']] for x in labels]

        height, width = float(image.shape[0]), float(image.shape[1])
        normed_bboxes = []
        for bbox in bboxes:
            x, y, w, h, _ = list(bbox)
            bbox[0] = (x + (w / 2.0)) / width
            bbox[1] = (y + (h / 2.0)) / height
            bbox[2] = max(min(w / width, 0.9999), 0.0001)
            bbox[3] = max(min(h / height, 0.9999), 0.0001)
            normed_bboxes.append(bbox)
        bboxes = normed_bboxes

        transforms = self.transforms(image=image, bboxes=bboxes)
        x = transforms['image']
        bboxes = transforms['bboxes']

        # Create target datastructure
        y = [torch.zeros((num_scales, s, s, 6)) for s in scale]
        y = bboxes_to_target(bboxes, y, ignore_iou_threshold, num_anchors_per_scale, self.anchor_boxes, scale)

        return x, y
