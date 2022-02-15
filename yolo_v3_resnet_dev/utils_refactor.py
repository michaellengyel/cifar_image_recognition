import torch
import config
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_midpoints(boxes_predictions, boxes_labels):
    """
    Calculates intersection over union (x, y, w, h)
    :param boxes_predictions: (tensor) Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels: (tensor) Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :return: Intersection over union for all examples
    """

    box1_x1 = boxes_predictions[..., 0:1] - boxes_predictions[..., 2:3] / 2
    box1_y1 = boxes_predictions[..., 1:2] - boxes_predictions[..., 3:4] / 2
    box1_x2 = boxes_predictions[..., 0:1] + boxes_predictions[..., 2:3] / 2
    box1_y2 = boxes_predictions[..., 1:2] + boxes_predictions[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def iou_corners(boxes_predictions, boxes_labels):
    """
    Calculates intersection over union for box format corners: (x1,y1,x2,y2)
    :param boxes_predictions: (tensor) Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels: (tensor) Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :return: Intersection over union for all examples
    """

    box1_x1 = boxes_predictions[..., 0:1]
    box1_y1 = boxes_predictions[..., 1:2]
    box1_x2 = boxes_predictions[..., 2:3]
    box1_y2 = boxes_predictions[..., 3:4]  # (N, 1)
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

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


from utils import non_max_suppression
def boxes_from_yp(yp):

    # batch_size = yp.shape[0][0]
    batch_size = 16
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
        nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=0.45, threshold=0.6, box_format="midpoint")
        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)
        train_idx += 1

    return all_pred_boxes


def boxes_from_y(y):

    batch_size = 16
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


def mean_average_precision(yp_boxes, y_boxes):

    # Sort predictions based on their source images
    yp_images = defaultdict(list)
    for box in yp_boxes:
        yp_images[box[0]].append(box)

    # Sort targets based on their source images
    y_images = defaultdict(list)
    for box in y_boxes:
        y_images[box[0]].append(box)

    # For the targets in a specific image
    for img_id, value in y_images.items():

        yp_image_class = defaultdict(list)
        for box in yp_images[img_id]:
            yp_image_class[box[1]].append(box)

        y_image_class = defaultdict(list)
        for box in value:
            y_image_class[box[1]].append(box)

        for class_id, value in y_image_class.items():

            # List of given image, given class target
            target_boxes = value
            # List of given image, given class prediction
            prediction_boxes = yp_image_class[class_id]

            image_class_precision = 0.0
            image_class_recall = 0.0
            matrix = np.zeros((len(target_boxes), len(prediction_boxes)))

            for yp_idx, yp in enumerate(prediction_boxes):
                for y_idx, y in enumerate(target_boxes):
                    iou = iou_midpoints(torch.tensor(yp[3:]), torch.tensor(y[3:]))
                    if iou > 0.5:
                        matrix[y_idx, yp_idx] = 1000
                    else:
                        matrix[y_idx, yp_idx] = iou

            row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

            """
            print("pred", len(prediction_boxes))
            print("targ", len(target_boxes))
            print(matrix.shape)
            print(matrix)
            print(row_ind)
            print(col_ind)
            print()
            """

    batch_map = 0.0
    return batch_map


def calc_batch_precision(yp_boxes, y_boxes, iou_threshold):
    return 0.0


def calc_batch_recall(yp_boxes, y_boxes, iou_threshold):
    return 0.0


def calc_mAP(yp_boxes, y_boxes, iou_threshold):

    epsilon = 1e-6

    # Sort predictions based on their source images
    yp_images = defaultdict(list)
    for box in yp_boxes:
        yp_images[(box[0], box[1])].append(box)

    # Sort targets based on their source images
    y_images = defaultdict(list)
    for box in y_boxes:
        y_images[(box[0], box[1])].append(box)

    image_precisions = []
    image_recalls = []

    for b in range(16):

        for c in range(20):

            TP = 0.0
            TN = 0.0
            FP = 0.0
            FN = 0.0

            current_y = y_images.get((b, c))
            current_yp = yp_images.get((b, c))

            if current_y is None and current_yp is None:
                continue
            elif current_y is None:
                FP = len(current_yp)
            elif current_yp is None:
                FN = len(current_y)
            else:
                matrix = np.zeros((len(current_y), len(current_yp)))
                for yp_idx, yp in enumerate(current_yp):
                    for y_idx, y in enumerate(current_y):
                        iou = iou_midpoints(torch.tensor(yp[3:]), torch.tensor(y[3:]))
                        if iou > iou_threshold:
                            matrix[y_idx, yp_idx] = iou
                        else:
                            matrix[y_idx, yp_idx] = 0
                row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
                if len(current_y) > len(current_yp):
                    TP = len(row_ind)
                    FN = len(current_y) - len(current_yp)
                elif len(current_yp) > len(current_y):
                    TP = len(row_ind)
                    FP = len(current_yp) - len(current_y)
                else:
                    TP = len(row_ind)

            precision = TP / (TP + FP + epsilon)
            recall = TP / (TP + FN + epsilon)

            image_precisions.append(precision)
            image_recalls.append(recall)

    prec = sum(image_precisions) / len(image_precisions)
    rec = sum(image_recalls) / len(image_recalls)
    return prec





