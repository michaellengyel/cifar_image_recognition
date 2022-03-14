import config
import torch
import math
from torchvision_utils import draw_bounding_boxes


def intersection_over_union(boxes_predictions, boxes_labels, box_format):

    if box_format == "midpoints":
        box1_x1 = boxes_predictions[..., 0:1] - boxes_predictions[..., 2:3] / 2
        box1_y1 = boxes_predictions[..., 1:2] - boxes_predictions[..., 3:4] / 2
        box1_x2 = boxes_predictions[..., 0:1] + boxes_predictions[..., 2:3] / 2
        box1_y2 = boxes_predictions[..., 1:2] + boxes_predictions[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_predictions[..., 0:1]
        box1_y1 = boxes_predictions[..., 1:2]
        box1_x2 = boxes_predictions[..., 2:3]
        box1_y2 = boxes_predictions[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    else:
        assert "Invalid box_formate in iou()"

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


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
        labels = [config.COCO_CLASSES[config.COCO_MAPPING_INV[x]] for x in classes]
        x[batch, ...] = draw_bounding_boxes(image=x[batch, ...].type(torch.uint8), boxes=boxes, colors=(0, 255, 255), labels=labels, width=2)


def draw_yp_on_x(x, yp, probability_threshold, anchors):

    batch, channel, width, height = x.shape

    for b in range(batch):

        boxes = []

        for scale in range(len(yp)):

            yp_anchor, yp_w, yp_h, _ = yp[scale][b, ...].shape
            num_cells = yp[scale][b, ...].shape[1]  # 13, 26, 52
            box_size = width / num_cells  # width = 416

            for a in range(yp_anchor):
                for w in range(yp_w):
                    for h in range(yp_h):

                        # Current cell probability is above threshold
                        # Use sigmoid to turn BCEWithLogitsLoss into a probability
                        if torch.sigmoid(yp[scale][b, a, w, h, 0]) > probability_threshold:

                            # p, x, y, w, h, logit
                            cell = yp[scale][b, a, w, h, ...]

                            # p, x, y, w, h
                            cell_class = torch.argmax(cell[5:])

                            # p, x, y, w, h, c
                            cell_box = torch.cat((cell[0:5], torch.tensor([cell_class])), dim=0)

                            # Decode yp
                            bbox_yolo = torch.tensor([cell_box[0],
                                                      (torch.sigmoid(cell_box[1]) + h) * box_size,
                                                      (torch.sigmoid(cell_box[2]) + w) * box_size,
                                                      anchors[scale][a][0] * math.pow(math.e, cell_box[3]) * width,
                                                      anchors[scale][a][1] * math.pow(math.e, cell_box[4]) * height,
                                                      cell_box[5]])

                            # Convert bbox midpoints to corners
                            bbox_corner = torch.tensor([bbox_yolo[0],
                                                        bbox_yolo[1] - bbox_yolo[3] / 2,
                                                        bbox_yolo[2] - bbox_yolo[4] / 2,
                                                        bbox_yolo[1] + bbox_yolo[3] / 2,
                                                        bbox_yolo[2] + bbox_yolo[4] / 2,
                                                        bbox_yolo[5]])

                            boxes.append(bbox_corner)

        nms_bboxes = non_maximum_suppression(bboxes=boxes, iou_threshold=0.25, threshold=0.5, box_format="corners")

        boxes = [box[1:5].tolist() for box in nms_bboxes]
        boxes = torch.tensor(boxes)
        labels = [config.COCO_CLASSES[config.COCO_MAPPING_INV[int(x[5].item())]] + " " + str(int(torch.sigmoid(x[0]).item() * 100)) + "%" for x in nms_bboxes]
        x[b, ...] = draw_bounding_boxes(image=x[b, ...].type(torch.uint8), boxes=boxes, colors=(255, 0, 255), labels=labels, width=1)


def yp_cells_to_boxes(yp_scale, anchors, S):

    """
    Turns a grid of SxS prediction cells into bboxes
    :param yp_scale: Grid of cells size Scale x Scale
    :param anchors:
    :param S:
    :return:
    """

    batch_size = yp_scale.shape[0]
    num_anchors = len(anchors)
    box_predictions = yp_scale[..., 1:5]

    anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
    box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
    scores = torch.sigmoid(yp_scale[..., 0:1])
    best_class = torch.argmax(yp_scale[..., 5:], dim=-1).unsqueeze(-1)

    cell_indices = (torch.arange(S).repeat(yp_scale.shape[0], 3, S, 1).unsqueeze(-1).to(yp_scale.device))
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * S * S, 6)

    return converted_bboxes.tolist()


def y_cells_to_boxes(y_scale, anchors, S):

    """
    Turns a grid of SxS targets cells into bboxes
    :param y_scale: Grid of cells size Scale x Scale
    :param anchors:
    :param S:
    :return:
    """

    batch_size = y_scale.shape[0]
    num_anchors = len(anchors)
    box_predictions = y_scale[..., 1:5]

    scores = y_scale[..., 0:1]
    best_class = y_scale[..., 5:6]

    cell_indices = (torch.arange(S).repeat(y_scale.shape[0], 3, S, 1).unsqueeze(-1).to(y_scale.device))
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * S * S, 6)

    return converted_bboxes.tolist()


def non_maximum_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    # bboxes = [bbox for bbox in bboxes if bbox[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    nms_bboxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[5] != chosen_box[5] or intersection_over_union(chosen_box[1:5], box[1:5], box_format=box_format) < iou_threshold]
        nms_bboxes.append(chosen_box)
    return nms_bboxes


def boxes_from_yp(yp, iou_threshold, threshold):

    batch_size = yp[0].shape[0]
    bboxes = [[] for _ in range(batch_size)]
    all_pred_boxes = []
    train_idx = 0

    for i in range(3):
        S = yp[i].shape[2]
        anchor = torch.tensor([*config.ANCHORS[i]]) * S
        boxes_scale_i = yp_cells_to_boxes(yp[i], anchor, S)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    for idx in range(batch_size):
        nms_boxes = non_maximum_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoints")
        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)
        train_idx += 1

    return all_pred_boxes


def boxes_from_y(y):

    batch_size = y[0].shape[0]
    S = y[2].shape[2]
    all_true_boxes = []
    train_idx = 0
    anchor = torch.tensor([*config.ANCHORS[2]]) * S

    # we just want one bbox for each label, not one for each scale
    true_bboxes = y_cells_to_boxes(y[2], anchor, S)

    for idx in range(batch_size):
        for box in true_bboxes[idx]:
            if box[1] > 0.6:
                all_true_boxes.append([train_idx] + box)
        train_idx += 1

    return all_true_boxes


def save_checkpoint(model, optimizer, cycle, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cycle": cycle,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint,
    # and it will lead to many hours of debugging \:
    #for param_group in optimizer.param_groups:
    #    param_group["lr"] = lr


def denormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
