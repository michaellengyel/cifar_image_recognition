import torch
import torch.nn as nn

from utils import iou_midpoints


class NoObjectLoss(nn.Module):
    def __init__(self):
        super(NoObjectLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, noobj, predictions, target):
        return self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))


class ObjectLoss(nn.Module):
    def __init__(self):
        super(ObjectLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()

    def forward(self, obj, predictions, target, anchors):
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = iou_midpoints(box_preds[obj], target[..., 1:5][obj]).detach()
        return self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])


class BoxCoordinateLoss(nn.Module):
    def __init__(self):
        super(BoxCoordinateLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, obj, predictions, target, anchors):
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x, y to be between [0, 1]
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        return self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, obj, predictions, target):
        return self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.no_object_loss = NoObjectLoss()
        self.object_loss = ObjectLoss()
        self.box_coordinate_loss = BoxCoordinateLoss()
        self.class_loss = ClassLoss()

    def forward(self, predictions, target, anchors):
        obj = (target[..., 0] == 1)
        noobj = (target[..., 0] == 0)
        no_object_loss = self.no_object_loss(noobj, predictions, target)
        object_loss = self.object_loss(obj, predictions, target, anchors)
        box_coordinate_loss = self.box_coordinate_loss(obj, predictions, target, anchors)
        class_loss = self.class_loss(obj, predictions, target)
        return no_object_loss + object_loss + box_coordinate_loss + class_loss

