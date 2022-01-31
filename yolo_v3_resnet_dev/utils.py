import torch
import time
import math
from collections import Counter

from torchvision_utils import draw_bounding_boxes

import config


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


def time_function(current_time):
    delta_time = time.time() - current_time
    current_time = time.time()
    return delta_time, current_time


def check_class_accuracy(model, loader, threshold):

    total_class_predictions, correct_class = 0, 0
    total_noobj, correct_noobj = 0, 0
    total_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(loader):
        x = x.float()
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):

            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1  # In the paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # In the paper this is Oobj_i

            correct_class += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj])
            total_class_predictions += torch.sum(obj)

            obj_predictions = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_predictions[obj] == y[i][..., 0][obj])
            total_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_predictions[noobj] == y[i][..., 0][noobj])
            total_noobj += torch.sum(noobj)

    class_accuracy = correct_class / (total_class_predictions + 1e-16) * 100
    no_object_accuracy = correct_noobj / (total_noobj + 1e-16) * 100
    obj_accuracy = correct_obj / (total_obj + 1e-16) * 100

    return class_accuracy, no_object_accuracy, obj_accuracy


def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):

    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.float()
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_boxes(predictions[i], anchor, S=S, is_pred=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # We just want one box for each label, not one for each scale
        true_boxes = cells_to_boxes(y[2], anchor, S=S, is_pred=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_boxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

        return all_pred_boxes, all_true_boxes


def cells_to_boxes(predictions, anchors, S, is_pred=True):

    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_pred:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device))
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)

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

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.float()
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


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


def draw_y_on_x(x, y):

    for batch in range(x.shape[0]):

        boxes = []
        classes = []

        for scale in range(len(y)):
            # print(y[scale][batch, ...].shape)
            for i in range(y[scale][batch, ...].shape[0]):
                for j in range(y[scale][batch, ...].shape[1]):
                    for k in range(y[scale][batch, ...].shape[2]):
                        cell = y[scale][batch, i, j, k, ...]
                        if cell[0] == 1:
                            bbox = cell[1:5]
                            cell_number = y[scale][batch, ...].shape[1]
                            box_size = 416 / cell_number
                            bbox_yolo = torch.tensor(
                                [box_size, box_size, box_size, box_size]) * bbox + torch.tensor(
                                [box_size * k, box_size * j, 0, 0])
                            bbox_corners = torch.tensor([bbox_yolo[0] - bbox_yolo[2] / 2,
                                                         bbox_yolo[1] - bbox_yolo[3] / 2,
                                                         bbox_yolo[0] + bbox_yolo[2] / 2,
                                                         bbox_yolo[1] + bbox_yolo[3] / 2])
                            boxes.append(bbox_corners.tolist())
                            classes.append(int(cell[5]))

        boxes = torch.tensor(boxes)
        labels = [config.PASCAL_CLASSES[x] for x in classes]
        x[batch, ...] = draw_bounding_boxes(image=x[batch, ...].type(torch.uint8), boxes=boxes, colors=(255, 0, 255), labels=labels, width=2)

    return x


def draw_yp_on_x(x, yp, class_threshold, anchors):

    batch, channel, width, height = x.shape

    for b in range(batch):

        boxes = []
        classes = []

        for scale in range(len(yp)):
            yp_anchor, yp_w, yp_h, _ = yp[scale][b, ...].shape
            for a in range(yp_anchor):
                for w in range(yp_w):
                    for h in range(yp_h):
                        cell = yp[scale][b, a, w, h, ...]
                        if cell[0] > class_threshold:
                            bbox_yp = cell[1:5]
                            num_cells = yp[scale][b, ...].shape[1]  # 13, 26, 52
                            box_size = width / num_cells

                            # Decode yp
                            bbox_yolo = torch.tensor([torch.sigmoid(bbox_yp[0]) + h * box_size,
                                                      torch.sigmoid(bbox_yp[1]) + w * box_size,
                                                      anchors[scale][a][0] * math.pow(math.e, bbox_yp[2]),
                                                      anchors[scale][a][1] * math.pow(math.e, bbox_yp[3])])
                            bbox_yolo_scaled = bbox_yolo * torch.tensor([1, 1, width, height])
                            bbox_corner = torch.tensor([bbox_yolo_scaled[0] - bbox_yolo_scaled[2] / 2,
                                                        bbox_yolo_scaled[1] - bbox_yolo_scaled[3] / 2,
                                                        bbox_yolo_scaled[0] + bbox_yolo_scaled[2] / 2,
                                                        bbox_yolo_scaled[1] + bbox_yolo_scaled[3] / 2])

                            boxes.append(bbox_corner.tolist())
                            class_args = cell[5:].argsort(descending=True, dim=0)
                            classes.append(int(class_args[0]))

        boxes = torch.tensor(boxes)
        labels = [config.PASCAL_CLASSES[x] for x in classes]
        x[b, ...] = draw_bounding_boxes(image=x[b, ...].type(torch.uint8), boxes=boxes, colors=(255, 0, 255), labels=labels, width=1)

    return x
