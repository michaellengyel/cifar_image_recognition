from utils import intersection_over_union
from utils import calc_batch_precision_recall

import unittest
import torch


class IouMidpointsTest(unittest.TestCase):

    def test_iou_midpoints_0(self):
        """Checked the iou of two coinciding squares"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([1.0],)
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_1(self):
        """Checked the iou of two identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([0.25, 0.25, 0.5, 0.5])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([(1/16)/(7/16)])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_2(self):
        """Checked the iou of two identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        boxes_labels = torch.tensor([0.25, 0.25, 0.5, 0.5])
        ious_expected = torch.tensor([(1/16)/(7/16)])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_3(self):
        """Checked the iou of two nested squares"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 1.0, 1.0])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([0.25])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_4(self):
        """Checked the iou of multiple identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        boxes_labels = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5]])
        ious_expected = torch.tensor([[(1/16)/(7/16)], [(1/16)/(7/16)]])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected[0].item(), ious_actual[0].item(), places=5)
        self.assertAlmostEqual(ious_expected[0].item(), ious_actual[0].item(), places=5)


class IouCenterTest(unittest.TestCase):

    def test_iou_center_0(self):
        # IF
        boxes_predictions = torch.tensor([0.25, 0.25, 0.75, 0.75])
        boxes_labels = torch.tensor([0.25, 0.25, 0.75, 0.75])
        ious_expected = torch.tensor([1.0],)
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "corners")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)


class PrecisionRecallTest(unittest.TestCase):

    # Box format: [image_id, class, prob, x, y, w, h]

    def test_p_r_0(self):
        """Checks precision recall in one image"""
        # IF
        y_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],
            [0, 0, 1.0, 0.75, 0.25, 0.2, 0.2]
        ]
        yp_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],  # TP
            [0, 0, 1.0, 0.75, 0.25, 0.2, 0.2],  # TP
            [0, 0, 1.0, 0.75, 0.75, 0.2, 0.2]   # FP
        ]
        iou_threshold = 0.5
        precision_expected = 2 / (2 + 1)
        recall_expected = 2 / (2 + 0)
        # WHEN
        precision_actual, recall_actual = calc_batch_precision_recall(y_boxes, yp_boxes, iou_threshold)
        # THEN
        self.assertAlmostEqual(precision_expected, precision_actual, places=5)
        self.assertAlmostEqual(recall_expected, recall_actual, places=5)

    def test_p_r_1(self):
        """Checks precision recall in one image"""
        # IF
        y_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],
            [0, 0, 1.0, 0.75, 0.25, 0.2, 0.2],  # FN
            [0, 0, 1.0, 0.25, 0.75, 0.2, 0.2]   # FN
        ]
        yp_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],  # TP
            [0, 0, 1.0, 0.75, 0.75, 0.2, 0.2]   # FP
        ]
        iou_threshold = 0.5
        precision_expected = 1 / (1 + 1)
        recall_expected = 1 / (1 + 2)
        # WHEN
        precision_actual, recall_actual = calc_batch_precision_recall(y_boxes, yp_boxes, iou_threshold)
        # THEN
        self.assertAlmostEqual(precision_expected, precision_actual, places=5)
        self.assertAlmostEqual(recall_expected, recall_actual, places=5)

    def test_p_r_2(self):
        """Checks precision recall in different images"""
        # IF
        y_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],
            [0, 0, 1.0, 0.75, 0.25, 0.2, 0.2],  # FN
            [0, 0, 1.0, 0.25, 0.75, 0.2, 0.2]   # FN
        ]
        yp_boxes = [
            [1, 0, 1.0, 0.25, 0.25, 0.2, 0.2],  # TP
            [1, 0, 1.0, 0.75, 0.75, 0.2, 0.2]   # FP
        ]
        iou_threshold = 0.5
        precision_expected = 0.0
        recall_expected = 0.0
        # WHEN
        precision_actual, recall_actual = calc_batch_precision_recall(y_boxes, yp_boxes, iou_threshold)
        # THEN
        self.assertAlmostEqual(precision_expected, precision_actual, places=5)
        self.assertAlmostEqual(recall_expected, recall_actual, places=5)

    def test_p_r_3(self):
        """Checks precision recall in one image for different classes"""
        # IF
        y_boxes = [
            [0, 0, 1.0, 0.25, 0.25, 0.2, 0.2],
            [0, 0, 1.0, 0.75, 0.25, 0.2, 0.2],  # FN
            [0, 0, 1.0, 0.25, 0.75, 0.2, 0.2]   # FN
        ]
        yp_boxes = [
            [0, 1, 1.0, 0.25, 0.25, 0.2, 0.2],  # TP
            [0, 1, 1.0, 0.75, 0.75, 0.2, 0.2]   # FP
        ]
        iou_threshold = 0.5
        precision_expected = 0.0
        recall_expected = 0.0
        # WHEN
        precision_actual, recall_actual = calc_batch_precision_recall(y_boxes, yp_boxes, iou_threshold)
        # THEN
        self.assertAlmostEqual(precision_expected, precision_actual, places=5)
        self.assertAlmostEqual(recall_expected, recall_actual, places=5)


if __name__ == '__main__':
    unittest.main()
