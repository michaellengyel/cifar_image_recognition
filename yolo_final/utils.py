import config
import torch
from torchvision_utils import draw_bounding_boxes


def draw_y_on_x(x, y):

    """
    x1 = y[..., 0] - y[..., 2] / 2
    y1 = y[..., 1] - y[..., 3] / 2
    x2 = y[..., 0] + y[..., 2] / 2
    y2 = y[..., 1] + y[..., 3] / 2
    c = y[..., 4]
    """

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
