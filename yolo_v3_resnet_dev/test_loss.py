from loss import NoObjectLoss
from loss import ObjectLoss
from loss import BoxCoordinateLoss
from loss import ClassLoss
from loss import YoloLoss

import torch
import unittest


class NoObjectLossTest(unittest.TestCase):

    def test_no_object_loss_0(self):
        # IF
        predictions = torch.zeros((16, 3, 13, 13, 25))
        predictions[0, 0, 0, 0, 0:5] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
        predictions[0, 0, 0, 0, 5] = torch.tensor([1.0])
        target = torch.zeros((16, 3, 13, 13, 6))
        target[0, 0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        noobj = (target[..., 0] == 0)
        loss_expected = torch.tensor([1.0])
        # WHEN
        loss = NoObjectLoss()
        loss_actual = loss(noobj=noobj, predictions=predictions, target=target)
        # THEN
        self.assertEqual(loss_expected.item(), loss_actual.item())


class ObjectLossTest(unittest.TestCase):

    def test_object_loss_0(self):
        # IF
        predictions = torch.zeros((16, 3, 13, 13, 25))
        predictions[0, 0, 0, 0, 0:5] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
        predictions[0, 0, 0, 0, 5] = torch.tensor([1.0])
        target = torch.zeros((16, 3, 13, 13, 6))
        target[0, 0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
        ]
        S = [13, 26, 52]
        scaled_anchors = (torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        obj = (target[..., 0] == 1)
        loss_expected = torch.tensor([1.0])
        # WHEN
        loss = ObjectLoss()
        loss_actual = loss(obj=obj, predictions=predictions, target=target, anchors=scaled_anchors[0])
        # THEN
        self.assertEqual(loss_expected.item(), loss_actual.item())


class BoxCoordinateLossTest(unittest.TestCase):

    def test_box_coordinate_loss_0(self):
        # IF
        predictions = torch.zeros((16, 3, 13, 13, 25))
        predictions[0, 0, 0, 0, 0:5] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
        predictions[0, 0, 0, 0, 5] = torch.tensor([1.0])
        target = torch.zeros((16, 3, 13, 13, 6))
        target[0, 0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
        ]
        S = [13, 26, 52]
        scaled_anchors = (torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        obj = (target[..., 0] == 1)
        loss_expected = torch.tensor([1.0])
        # WHEN
        loss = BoxCoordinateLoss()
        loss_actual = loss(obj=obj, predictions=predictions, target=target, anchors=scaled_anchors[0])
        # THEN
        self.assertEqual(loss_expected.item(), loss_actual.item())


class ClassLossTest(unittest.TestCase):

    def test_class_loss_0(self):
        # IF
        predictions = torch.zeros((16, 3, 13, 13, 25))
        predictions[0, 0, 0, 0, 0:5] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        predictions[0, 0, 0, 0, 5] = torch.tensor([1.0])
        target = torch.zeros((16, 3, 13, 13, 6))
        target[0, 0, 0, 0, ...] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        obj = (target[..., 0] == 1)
        loss_expected = torch.tensor([1.0])
        # WHEN
        loss = ClassLoss()
        loss_actual = loss(obj=obj, predictions=predictions, target=target)
        # THEN
        self.assertEqual(loss_expected.item(), loss_actual.item())


class YoloLossTest(unittest.TestCase):

    def test_yolo_loss_0(self):
        # IF
        predictions = torch.zeros((16, 3, 13, 13, 25))
        predictions[0, 0, 0, 0, 0:5] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
        predictions[0, 0, 0, 0, 6] = torch.tensor([1.0])
        target = torch.zeros((16, 3, 13, 13, 6))
        target[0, 0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
        ]
        S = [13, 26, 52]
        scaled_anchors = (torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        loss_expected = torch.tensor([1.0])
        # WHEN
        loss = YoloLoss()
        loss_actual = loss(predictions=predictions, target=target, anchors=scaled_anchors[0])
        # THEN
        self.assertEqual(loss_expected.item(), loss_actual.item())


if __name__ == '__main__':
    unittest.main()
