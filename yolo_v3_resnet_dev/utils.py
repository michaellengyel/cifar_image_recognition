import torch
import time
import math
import numpy as np

from scipy.optimize import linear_sum_assignment
from torchvision_utils import draw_bounding_boxes

import config


def intersection_over_union(boxes_predictions, boxes_labels, box_format):

    if box_format == "midpoint":
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


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


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
        nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
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


def calc_batch_precision_recall(y_boxes, yp_boxes, iou_threshold):

    state_matrix = torch.zeros((16, 20, 3))  # TP, FP, FN

    for i in range(state_matrix.shape[0]):
        for c in range(state_matrix.shape[1]):

            ys = []
            yps = []

            for y in y_boxes:
                if (i, c) == (y[0], y[1]):
                    ys.append(y)

            for yp in yp_boxes:
                if (i, c) == (yp[0], yp[1]):
                    yps.append(yp)

            # No labels or predictions of this class and image
            if ys is False and yps is False:
                state_matrix[i, c, :] = torch.tensor([0, 0, 0])
            # No labels for this class and image
            elif ys is False:
                state_matrix[i, c, :] = torch.tensor([0, len(yps), 0])
            # No predictions for this class and image
            elif yps is False:
                state_matrix[i, c, :] = torch.tensor([0, 0, len(ys)])
            # Labels and prediction for class and image
            else:
                iou_matrix = np.zeros((len(ys), len(yps)))
                for y_idx, y in enumerate(ys):
                    for yp_idx, yp in enumerate(yps):
                        iou = intersection_over_union(torch.tensor(yp[3:]), torch.tensor(y[3:]), "midpoint")
                        iou_matrix[y_idx, yp_idx] = iou

                row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

                values = []
                for row, col in zip(row_ind, col_ind):
                    if iou_matrix[row, col] >= iou_threshold:
                        values.append(iou_matrix[row, col])

                state_matrix[i, c, :] = torch.tensor([len(values), len(yps) - len(values), len(ys) - len(values)])

    # Calculate batch precision and recall from state_matrix
    # TODO: Parallelize

    prec = []
    rec = []
    epsilon = 1e-6
    for i in range(state_matrix.shape[0]):
        for c in range(state_matrix.shape[1]):
            if torch.sum(state_matrix[i, c, :]).item() != 0:
                if state_matrix[i, c, 0].item() == 0:
                    p = 0
                    r = 0
                    prec.append(p)
                    rec.append(r)
                else:
                    p = state_matrix[i, c, 0] / (state_matrix[i, c, 0] + state_matrix[i, c, 1] + epsilon)
                    r = state_matrix[i, c, 0] / (state_matrix[i, c, 0] + state_matrix[i, c, 2] + epsilon)
                    prec.append(p)
                    rec.append(r)

    precision = sum(prec) / len(prec)
    recall = sum(rec) / len(rec)

    return precision, recall


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint,
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def time_function(current_time):
    delta_time = time.time() - current_time
    current_time = time.time()
    return delta_time, current_time


def non_maximum_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    # bboxes = [bbox for bbox in bboxes if bbox[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    nms_bboxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[5] != chosen_box[5] or intersection_over_union(chosen_box[1:5], box[1:5], box_format=box_format) < iou_threshold]
        nms_bboxes.append(chosen_box)
    return nms_bboxes


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
                        if yp[scale][b, a, w, h, 0] > probability_threshold:

                            # p, x, y, w, h, logit
                            cell = yp[scale][b, a, w, h, ...]

                            # p, x, y, w, h
                            cell_class = torch.argmax(cell[5:])

                            # p, x, y, w, h, c
                            cell_box = torch.cat((cell[0:5], torch.tensor([cell_class])), dim=0)

                            # Decode yp
                            bbox_yolo = torch.tensor([cell_box[0],
                                                      torch.sigmoid(cell_box[1]) + (h + 0.5) * box_size,
                                                      torch.sigmoid(cell_box[2]) + (w + 0.5) * box_size,
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
        labels = [config.PASCAL_CLASSES[int(x[5].item())] for x in nms_bboxes]
        x[b, ...] = draw_bounding_boxes(image=x[b, ...].type(torch.uint8), boxes=boxes, colors=(255, 0, 255), labels=labels, width=2)

    return x


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
        labels = [config.PASCAL_CLASSES[x] for x in classes]
        x[batch, ...] = draw_bounding_boxes(image=x[batch, ...].type(torch.uint8), boxes=boxes, colors=(255, 255, 255), labels=labels, width=2)

    return x