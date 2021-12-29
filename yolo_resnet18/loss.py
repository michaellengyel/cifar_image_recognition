import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, S, C, B):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, y_p, y):

        iou_b1 = intersection_over_union(y_p[..., 21:25], y[..., 21:25])
        iou_b2 = intersection_over_union(y_p[..., 26:30], y[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = y[..., 20].unsqueeze(3)  # Identity of object i (is there an object in cell i?)

        # FOR BOX COORDINATES LOSS

        box_predictions = exists_box * (best_box * y_p[..., 26:30] + (1 - best_box) * y_p[..., 21:25])
        box_targets = exists_box * y[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        # (M, S, S, 25) This is why ... is needed
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (M, S, S, 25) -> (M*S*S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        # FOR OBJECT LOSS

        pred_box = (best_box * y_p[..., 25:26] + (1 - best_box) * y_p[..., 20:21])

        # (N*S*S, 1)
        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * y[..., 20:21]))

        # FOR NO OBJECT LOSS

        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * y_p[..., 20:21], start_dim=1),
                                  torch.flatten((1 - exists_box) * y[..., 20:21], start_dim=1))

        no_object_loss += self.mse(torch.flatten((1 - exists_box) * y_p[..., 25:26], start_dim=1),
                                   torch.flatten((1 - exists_box) * y[..., 20:21], start_dim=1))

        # FOR CLASS LOSS

        # (N, S, S, 20) -> (N * S * S, 20)
        class_loss = self.mse(torch.flatten(exists_box * y_p[..., :20],  end_dim=-2),
                              torch.flatten(exists_box * y[..., :20], end_dim=-2))

        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)

        return loss


def intersection_over_union(boxes_y_p, boxes_y):

    box1_x1 = boxes_y_p[..., 0:1] - boxes_y_p[..., 2:3] / 2
    box1_y1 = boxes_y_p[..., 1:2] - boxes_y_p[..., 3:4] / 2
    box1_x2 = boxes_y_p[..., 0:1] + boxes_y_p[..., 2:3] / 2
    box1_y2 = boxes_y_p[..., 1:2] + boxes_y_p[..., 3:4] / 2
    box2_x1 = boxes_y[..., 0:1] - boxes_y[..., 2:3] / 2
    box2_y1 = boxes_y[..., 1:2] - boxes_y[..., 3:4] / 2
    box2_x2 = boxes_y[..., 0:1] + boxes_y[..., 2:3] / 2
    box2_y2 = boxes_y[..., 1:2] + boxes_y[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)