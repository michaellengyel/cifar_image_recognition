import config
import torch
from torchvision_utils import draw_bounding_boxes

"""
def draw_y_on_x(x, y):

    # x1 = y[..., 0] - y[..., 2] / 2
    # y1 = y[..., 1] - y[..., 3] / 2
    # x2 = y[..., 0] + y[..., 2] / 2
    # y2 = y[..., 1] + y[..., 3] / 2
    # c = y[..., 4]

    x1 = y[..., 0]
    y1 = y[..., 1]
    x2 = y[..., 0] + y[..., 2]
    y2 = y[..., 1] + y[..., 3]
    c = y[..., 4]

    y_corner = torch.stack([x1, y1, x2, y2, c], dim=2)

    for b in range(y.shape[0]):

        boxes = []
        classes = []

        for i in range(y_corner.shape[1]):
            if torch.sum(y_corner[b, i, :]).item() != 0:
                boxes.append(y_corner[b, i, 0:4].tolist())
                classes.append(int(y_corner[b, i, 4]))

        boxes = torch.tensor(boxes)
        labels = [config.COCO_CLASSES[x] for x in classes]
        x[b, ...] = draw_bounding_boxes(image=x[b, ...].type(torch.uint8), boxes=boxes, colors=(255, 0, 255), labels=labels, width=2)

    return x
"""


def draw_y_on_x(x, y):

    for batch in range(x.shape[0]):

        boxes = []
        classes = []

        for scale in range(len(y)):
            # print(y[scale][batch, ...].shape)
            for a in range(y[scale][batch, ...].shape[0]):
                for w in range(y[scale][batch, ...].shape[1]):
                    for h in range(y[scale][batch, ...].shape[2]):
                        cell = y[scale][batch, a, w, h, ...]
                        if cell[0] == 1:
                            bbox = cell[1:5]
                            cell_number = y[scale][batch, ...].shape[1]
                            box_size = 416 / cell_number
                            bbox_yolo = torch.tensor([box_size, box_size, box_size, box_size]) * bbox + torch.tensor([box_size * h, box_size * w, 0, 0])
                            bbox_corners = torch.tensor([bbox_yolo[0] - bbox_yolo[2] / 2,
                                                         bbox_yolo[1] - bbox_yolo[3] / 2,
                                                         bbox_yolo[0] + bbox_yolo[2] / 2,
                                                         bbox_yolo[1] + bbox_yolo[3] / 2])
                            boxes.append(bbox_corners.tolist())
                            classes.append(int(cell[5]))

        boxes = torch.tensor(boxes)
        labels = [config.COCO_CLASSES[x] for x in classes]
        x[batch, ...] = draw_bounding_boxes(image=x[batch, ...].type(torch.uint8), boxes=boxes, colors=(255, 255, 255), labels=labels, width=2)

    return x